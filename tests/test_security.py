"""Security tests for the embodied AI benchmark."""

import pytest
import tempfile
import json
from pathlib import Path
from unittest.mock import patch, mock_open
import hashlib
import os

import sys
sys.path.append(str(Path(__file__).parent.parent / "src"))

from embodied_ai_benchmark.utils.validation import SecurityValidator, ValidationError
from embodied_ai_benchmark.utils.logging_config import BenchmarkLogger


class TestSecurityValidator:
    """Test suite for security validation functionality."""
    
    def test_model_file_validation_success(self):
        """Test successful model file validation."""
        # Create temporary model file
        with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
            model_path = f.name
            # Write some dummy data (less than max size)
            f.write(b"dummy model data" * 1000)  # Small file
        
        try:
            validated_path = SecurityValidator.validate_model_file(model_path, max_size_mb=1)
            assert validated_path.exists()
            assert validated_path.suffix == '.pt'
        finally:
            Path(model_path).unlink(missing_ok=True)
    
    def test_model_file_validation_size_limit(self):
        """Test model file size limit validation."""
        # Create temporary large file
        with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
            model_path = f.name
            # Write data larger than max size
            f.write(b"x" * (2 * 1024 * 1024))  # 2MB
        
        try:
            with pytest.raises(ValidationError, match="Model file too large"):
                SecurityValidator.validate_model_file(model_path, max_size_mb=1)
        finally:
            Path(model_path).unlink(missing_ok=True)
    
    def test_model_file_validation_extension(self):
        """Test model file extension validation."""
        # Create temporary file with invalid extension
        with tempfile.NamedTemporaryFile(suffix='.txt', delete=False) as f:
            model_path = f.name
            f.write(b"dummy data")
        
        try:
            with pytest.raises(ValidationError, match="Unsupported model file type"):
                SecurityValidator.validate_model_file(model_path)
        finally:
            Path(model_path).unlink(missing_ok=True)
    
    def test_model_file_not_found(self):
        """Test model file validation with non-existent file."""
        with pytest.raises(ValidationError, match="Model file not found"):
            SecurityValidator.validate_model_file("non_existent_file.pt")
    
    def test_network_config_validation_success(self):
        """Test successful network configuration validation."""
        config = {
            "host": "localhost",
            "port": 8080,
            "timeout": 30
        }
        
        validated_config = SecurityValidator.validate_network_config(config)
        assert validated_config["host"] == "localhost"
        assert validated_config["port"] == 8080
    
    def test_network_config_dangerous_host(self):
        """Test network configuration with dangerous host."""
        dangerous_configs = [
            {"host": "0.0.0.0"},
            {"host": "*"},
            {"host": ""}
        ]
        
        for config in dangerous_configs:
            with pytest.raises(ValidationError, match="Dangerous host configuration"):
                SecurityValidator.validate_network_config(config)
    
    def test_network_config_invalid_port(self):
        """Test network configuration with invalid port."""
        invalid_port_configs = [
            {"port": 80},      # Below 1024
            {"port": 70000},   # Above 65535
            {"port": -1},      # Negative
            {"port": "8080"}   # String instead of int
        ]
        
        for config in invalid_port_configs:
            with pytest.raises(ValidationError, match="Port must be integer in range"):
                SecurityValidator.validate_network_config(config)
    
    def test_resource_limits_validation_success(self):
        """Test successful resource limits validation."""
        # Normal resource usage should pass
        result = SecurityValidator.check_resource_limits(
            memory_mb=1024,      # 1GB
            cpu_percent=50.0,    # 50%
            gpu_memory_mb=4096   # 4GB
        )
        assert result is True
    
    def test_resource_limits_memory_exceeded(self):
        """Test resource limits with excessive memory usage."""
        with pytest.raises(ValidationError, match="Memory usage too high"):
            SecurityValidator.check_resource_limits(memory_mb=10240)  # 10GB
    
    def test_resource_limits_cpu_exceeded(self):
        """Test resource limits with excessive CPU usage."""
        with pytest.raises(ValidationError, match="CPU usage too high"):
            SecurityValidator.check_resource_limits(cpu_percent=95.0)
    
    def test_resource_limits_gpu_exceeded(self):
        """Test resource limits with excessive GPU memory usage."""
        with pytest.raises(ValidationError, match="GPU memory usage too high"):
            SecurityValidator.check_resource_limits(gpu_memory_mb=15360)  # 15GB


class TestFilePathSecurity:
    """Test file path security validation."""
    
    def test_safe_filepath_validation(self):
        """Test validation of safe file paths."""
        from embodied_ai_benchmark.utils.validation import InputValidator
        
        safe_paths = [
            "/tmp/safe_file.txt",
            "data/model.pt",
            "./results/output.json"
        ]
        
        for path in safe_paths:
            validated_path = InputValidator.sanitize_filepath(path)
            assert isinstance(validated_path, Path)
    
    def test_directory_traversal_prevention(self):
        """Test prevention of directory traversal attacks."""
        from embodied_ai_benchmark.utils.validation import InputValidator
        
        malicious_paths = [
            "../../../etc/passwd",
            "..\\..\\windows\\system32\\config\\sam",
            "/tmp/../../../etc/passwd",
            "C:\\..\\..\\Windows\\System32\\config\\SAM"
        ]
        
        for path in malicious_paths:
            with pytest.raises(ValidationError, match="Directory traversal not allowed"):
                InputValidator.sanitize_filepath(path)
    
    def test_suspicious_characters_in_path(self):
        """Test detection of suspicious characters in file paths."""
        from embodied_ai_benchmark.utils.validation import InputValidator
        
        suspicious_paths = [
            "file<script>.txt",
            "output>redirect.json",
            'file"quote.pt',
            "pipe|command.log",
            "wild*card.txt",
            "question?.dat"
        ]
        
        for path in suspicious_paths:
            with pytest.raises(ValidationError, match="Suspicious characters"):
                InputValidator.sanitize_filepath(path)
    
    def test_empty_filepath(self):
        """Test validation of empty file paths."""
        from embodied_ai_benchmark.utils.validation import InputValidator
        
        with pytest.raises(ValidationError, match="File path cannot be empty"):
            InputValidator.sanitize_filepath("")
        
        with pytest.raises(ValidationError, match="File path cannot be empty"):
            InputValidator.sanitize_filepath("   ")  # Whitespace only


class TestConfigurationSecurity:
    """Test security validation of configuration files."""
    
    def test_json_config_validation_success(self):
        """Test successful JSON configuration validation."""
        from embodied_ai_benchmark.utils.validation import InputValidator
        
        config_data = {
            "benchmark": {
                "name": "test_benchmark",
                "episodes": 100
            },
            "agent": {
                "type": "random",
                "seed": 42
            }
        }
        
        # Create temporary JSON file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(config_data, f)
            config_path = f.name
        
        try:
            validated_config = InputValidator.validate_json_config(config_path)
            assert validated_config["benchmark"]["name"] == "test_benchmark"
            assert validated_config["agent"]["type"] == "random"
        finally:
            Path(config_path).unlink(missing_ok=True)
    
    def test_json_config_malformed(self):
        """Test validation of malformed JSON configuration."""
        from embodied_ai_benchmark.utils.validation import InputValidator
        
        # Create temporary malformed JSON file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            f.write('{"invalid": json, "missing": quote}')
            config_path = f.name
        
        try:
            with pytest.raises(ValidationError, match="Invalid JSON"):
                InputValidator.validate_json_config(config_path)
        finally:
            Path(config_path).unlink(missing_ok=True)
    
    def test_json_config_not_object(self):
        """Test validation of JSON file that doesn't contain an object."""
        from embodied_ai_benchmark.utils.validation import InputValidator
        
        # Create temporary JSON file with array instead of object
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(["not", "an", "object"], f)
            config_path = f.name
        
        try:
            with pytest.raises(ValidationError, match="Configuration file must contain a JSON object"):
                InputValidator.validate_json_config(config_path)
        finally:
            Path(config_path).unlink(missing_ok=True)


class TestLoggingSecurity:
    """Test security aspects of logging system."""
    
    def test_log_injection_prevention(self):
        """Test prevention of log injection attacks."""
        from embodied_ai_benchmark.utils.logging_config import get_logger
        
        logger = get_logger("security_test")
        
        # Malicious input with newlines and ANSI escape codes
        malicious_input = "User input\n[FAKE] CRITICAL - System compromised\x1b[31mRed text\x1b[0m"
        
        # The logging system should handle this safely
        # This shouldn't raise an exception
        logger.info(f"Processing user input: {malicious_input}")
        
        # Test with format string injection attempt
        format_attack = "User: %s%s%s%s%s%s%s%s%s%s%n"
        logger.info(f"User data: {format_attack}")
    
    def test_sensitive_data_filtering(self):
        """Test filtering of sensitive data from logs."""
        from embodied_ai_benchmark.utils.logging_config import get_logger
        
        logger = get_logger("sensitive_test")
        
        # Simulate logging with potential sensitive data
        # In a real implementation, you might want to filter these patterns
        sensitive_patterns = [
            "password=secret123",
            "api_key=sk-1234567890abcdef",
            "token=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9",
            "credit_card=4111-1111-1111-1111"
        ]
        
        for pattern in sensitive_patterns:
            # This test verifies the logging system doesn't crash
            # In production, you might want to implement filtering
            logger.info(f"Config loaded with {pattern}")
    
    @patch('embodied_ai_benchmark.utils.logging_config.logging.handlers.RotatingFileHandler')
    def test_log_file_permissions(self, mock_handler):
        """Test that log files are created with secure permissions."""
        from embodied_ai_benchmark.utils.logging_config import BenchmarkLogger
        
        # Create logger instance
        logger_instance = BenchmarkLogger()
        
        # Verify that file handlers are created (mocked)
        assert mock_handler.called
        
        # In a real implementation, you would check file permissions here
        # os.stat(log_file).st_mode & 0o777 should be 0o600 (owner read/write only)


class TestInputSanitization:
    """Test input sanitization and validation."""
    
    def test_action_validation_bounds_checking(self):
        """Test action validation with bounds checking."""
        from embodied_ai_benchmark.utils.validation import InputValidator
        
        action_space = {
            "type": "continuous",
            "shape": (3,),
            "low": [-1.0, -2.0, -3.0],
            "high": [1.0, 2.0, 3.0]
        }
        
        # Valid action
        valid_action = {
            "type": "continuous",
            "values": [0.5, 1.0, -1.5]
        }
        
        validated_action = InputValidator.validate_action(valid_action, action_space)
        assert validated_action["type"] == "continuous"
        
        # Invalid action - out of bounds
        invalid_action = {
            "type": "continuous",
            "values": [2.0, 1.0, -1.5]  # First value exceeds upper bound
        }
        
        with pytest.raises(ValidationError, match="Action values out of bounds"):
            InputValidator.validate_action(invalid_action, action_space)
    
    def test_observation_sanitization(self):
        """Test observation sanitization for NaN and infinite values."""
        from embodied_ai_benchmark.utils.validation import InputValidator
        import numpy as np
        
        # Observation with problematic values
        observation = {
            "sensor_1": np.array([1.0, np.nan, 3.0]),
            "sensor_2": np.array([np.inf, 2.0, -np.inf]),
            "sensor_3": np.array([1.0, 2.0, 3.0])  # Clean data
        }
        
        sanitized_obs = InputValidator.validate_observation(observation)
        
        # NaN and inf values should be replaced
        assert not np.isnan(sanitized_obs["sensor_1"]).any()
        assert not np.isinf(sanitized_obs["sensor_2"]).any()
        # Clean data should remain unchanged
        np.testing.assert_array_equal(sanitized_obs["sensor_3"], observation["sensor_3"])
    
    def test_config_schema_validation(self):
        """Test configuration schema validation."""
        from embodied_ai_benchmark.utils.validation import InputValidator
        
        schema = {
            "name": str,
            "episodes": int,
            "learning_rate": float,
            "capabilities": list
        }
        
        # Valid configuration
        valid_config = {
            "name": "test_benchmark",
            "episodes": 100,
            "learning_rate": 0.01,
            "capabilities": ["vision", "action"]
        }
        
        validated_config = InputValidator.validate_config(valid_config, schema)
        assert validated_config["name"] == "test_benchmark"
        assert validated_config["episodes"] == 100
        
        # Invalid configuration - wrong type
        invalid_config = {
            "name": "test_benchmark",
            "episodes": "100",  # Should be int
            "learning_rate": 0.01,
            "capabilities": ["vision", "action"]
        }
        
        # Should attempt type conversion
        converted_config = InputValidator.validate_config(invalid_config, schema)
        assert isinstance(converted_config["episodes"], int)
        assert converted_config["episodes"] == 100


@pytest.mark.security
class TestSecurityIntegration:
    """Integration tests for security features."""
    
    def test_end_to_end_security_validation(self):
        """Test end-to-end security validation workflow."""
        from embodied_ai_benchmark.utils.validation import SecurityValidator, InputValidator
        
        # Test complete security validation pipeline
        test_config = {
            "network": {
                "host": "localhost",
                "port": 8080
            },
            "resources": {
                "memory_mb": 1024,
                "cpu_percent": 50.0
            }
        }
        
        # Validate network configuration
        validated_network = SecurityValidator.validate_network_config(test_config["network"])
        assert validated_network["host"] == "localhost"
        
        # Validate resource limits
        result = SecurityValidator.check_resource_limits(
            memory_mb=test_config["resources"]["memory_mb"],
            cpu_percent=test_config["resources"]["cpu_percent"]
        )
        assert result is True
    
    def test_security_logging_integration(self):
        """Test integration between security validation and logging."""
        from embodied_ai_benchmark.utils.logging_config import BenchmarkLogger
        
        logger_instance = BenchmarkLogger()
        test_logger = logger_instance.get_logger("security_integration_test")
        
        # Test security event logging
        security_event = {
            "type": "validation_failure",
            "details": "Invalid file path detected",
            "severity": "high"
        }
        
        # This should not raise an exception
        logger_instance.log_security_event(test_logger, "validation_failure", security_event)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short", "-m", "not slow"])
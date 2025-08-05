"""Input validation utilities for the embodied AI benchmark."""

import numpy as np
from typing import Any, Dict, List, Optional, Union, Callable, Tuple
import logging
from pathlib import Path
import json

logger = logging.getLogger(__name__)


class ValidationError(Exception):
    """Custom exception for validation errors."""
    pass


class InputValidator:
    """Comprehensive input validation for benchmark components."""
    
    @staticmethod
    def validate_config(config: Dict[str, Any], schema: Dict[str, Any]) -> Dict[str, Any]:
        """Validate configuration against schema.
        
        Args:
            config: Configuration dictionary to validate
            schema: Schema defining expected structure and types
            
        Returns:
            Validated and sanitized configuration
            
        Raises:
            ValidationError: If configuration is invalid
        """
        if not isinstance(config, dict):
            raise ValidationError(f"Configuration must be a dictionary, got {type(config)}")
        
        validated_config = {}
        
        for key, expected_type in schema.items():
            if isinstance(expected_type, dict):
                # Nested validation
                if key in config:
                    validated_config[key] = InputValidator.validate_config(
                        config[key], expected_type
                    )
                else:
                    validated_config[key] = {}
            else:
                # Type validation
                if key in config:
                    value = config[key]
                    if not isinstance(value, expected_type):
                        try:
                            # Attempt type conversion
                            validated_config[key] = expected_type(value)
                        except (ValueError, TypeError):
                            raise ValidationError(
                                f"Invalid type for '{key}': expected {expected_type.__name__}, "
                                f"got {type(value).__name__}"
                            )
                    else:
                        validated_config[key] = value
        
        return validated_config
    
    @staticmethod
    def validate_action(action: Dict[str, Any], action_space: Dict[str, Any]) -> Dict[str, Any]:
        """Validate agent action against action space.
        
        Args:
            action: Action dictionary from agent
            action_space: Action space specification
            
        Returns:
            Validated action dictionary
            
        Raises:
            ValidationError: If action is invalid
        """
        if not isinstance(action, dict):
            raise ValidationError(f"Action must be a dictionary, got {type(action)}")
        
        action_type = action.get("type", "unknown")
        
        if action_type == "continuous":
            values = action.get("values", [])
            if not isinstance(values, (list, np.ndarray)):
                raise ValidationError("Continuous action values must be list or numpy array")
            
            values = np.array(values)
            
            # Check shape
            expected_shape = action_space.get("shape", ())
            if values.shape != expected_shape:
                raise ValidationError(
                    f"Action shape mismatch: expected {expected_shape}, got {values.shape}"
                )
            
            # Check bounds
            low = np.array(action_space.get("low", -np.inf))
            high = np.array(action_space.get("high", np.inf))
            
            if np.any(values < low) or np.any(values > high):
                raise ValidationError(
                    f"Action values out of bounds: {values} not in [{low}, {high}]"
                )
        
        elif action_type == "discrete":
            value = action.get("value", 0)
            if not isinstance(value, int):
                raise ValidationError("Discrete action value must be integer")
            
            n_actions = action_space.get("n", 1)
            if not (0 <= value < n_actions):
                raise ValidationError(f"Discrete action {value} out of range [0, {n_actions})")
        
        return action
    
    @staticmethod
    def validate_observation(observation: Dict[str, Any]) -> Dict[str, Any]:
        """Validate observation dictionary.
        
        Args:
            observation: Observation from environment
            
        Returns:
            Validated observation
            
        Raises:
            ValidationError: If observation is invalid
        """
        if not isinstance(observation, dict):
            raise ValidationError(f"Observation must be a dictionary, got {type(observation)}")
        
        validated_obs = {}
        
        for key, value in observation.items():
            if isinstance(value, np.ndarray):
                # Check for invalid values
                if np.any(np.isnan(value)):
                    logger.warning(f"NaN values detected in observation '{key}'")
                    value = np.nan_to_num(value)
                
                if np.any(np.isinf(value)):
                    logger.warning(f"Infinite values detected in observation '{key}'")
                    value = np.nan_to_num(value)
            
            validated_obs[key] = value
        
        return validated_obs
    
    @staticmethod
    def validate_agent_config(config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate agent configuration.
        
        Args:
            config: Agent configuration dictionary
            
        Returns:
            Validated configuration
            
        Raises:
            ValidationError: If configuration is invalid
        """
        schema = {
            "agent_id": str,
            "capabilities": list,
            "max_episode_length": int,
            "learning_rate": float,
            "batch_size": int
        }
        
        # Validate basic structure
        validated_config = {}
        for key, value in config.items():
            if key in schema:
                expected_type = schema[key]
                if not isinstance(value, expected_type):
                    try:
                        validated_config[key] = expected_type(value)
                    except (ValueError, TypeError):
                        raise ValidationError(f"Invalid {key}: expected {expected_type}, got {type(value)}")
                else:
                    validated_config[key] = value
            else:
                validated_config[key] = value
        
        # Specific validations
        if "learning_rate" in validated_config:
            lr = validated_config["learning_rate"]
            if not (0.0 < lr <= 1.0):
                raise ValidationError(f"Learning rate must be in (0, 1], got {lr}")
        
        if "batch_size" in validated_config:
            batch_size = validated_config["batch_size"]
            if batch_size <= 0:
                raise ValidationError(f"Batch size must be positive, got {batch_size}")
        
        return validated_config
    
    @staticmethod
    def validate_episode_results(results: Dict[str, Any]) -> Dict[str, Any]:
        """Validate episode results dictionary.
        
        Args:
            results: Episode results from evaluator
            
        Returns:
            Validated results
            
        Raises:
            ValidationError: If results are invalid
        """
        if not isinstance(results, dict):
            raise ValidationError(f"Results must be a dictionary, got {type(results)}")
        
        required_fields = ["episode_id", "total_steps", "total_reward", "success"]
        for field in required_fields:
            if field not in results:
                raise ValidationError(f"Missing required field '{field}' in episode results")
        
        # Validate types and ranges
        if not isinstance(results["episode_id"], int) or results["episode_id"] < 0:
            raise ValidationError("Episode ID must be non-negative integer")
        
        if not isinstance(results["total_steps"], int) or results["total_steps"] < 0:
            raise ValidationError("Total steps must be non-negative integer")
        
        if not isinstance(results["total_reward"], (int, float)):
            raise ValidationError("Total reward must be numeric")
        
        if not isinstance(results["success"], bool):
            raise ValidationError("Success must be boolean")
        
        return results
    
    @staticmethod
    def sanitize_filepath(filepath: str) -> Path:
        """Sanitize and validate file path.
        
        Args:
            filepath: File path string
            
        Returns:
            Sanitized Path object
            
        Raises:
            ValidationError: If path is invalid or unsafe
        """
        if not isinstance(filepath, str):
            raise ValidationError(f"File path must be string, got {type(filepath)}")
        
        if not filepath.strip():
            raise ValidationError("File path cannot be empty")
        
        path = Path(filepath).resolve()
        
        # Check for directory traversal attempts
        if ".." in str(path):
            raise ValidationError("Directory traversal not allowed in file paths")
        
        # Check for suspicious characters
        suspicious_chars = ['<', '>', ':', '"', '|', '?', '*']
        if any(char in str(path) for char in suspicious_chars):
            raise ValidationError(f"Suspicious characters in file path: {filepath}")
        
        return path
    
    @staticmethod
    def validate_json_config(config_path: str) -> Dict[str, Any]:
        """Validate and load JSON configuration file.
        
        Args:
            config_path: Path to JSON configuration file
            
        Returns:
            Loaded configuration dictionary
            
        Raises:
            ValidationError: If file is invalid or cannot be loaded
        """
        path = InputValidator.sanitize_filepath(config_path)
        
        if not path.exists():
            raise ValidationError(f"Configuration file not found: {config_path}")
        
        if not path.is_file():
            raise ValidationError(f"Configuration path is not a file: {config_path}")
        
        try:
            with open(path, 'r') as f:
                config = json.load(f)
            
            if not isinstance(config, dict):
                raise ValidationError("Configuration file must contain a JSON object")
            
            return config
            
        except json.JSONDecodeError as e:
            raise ValidationError(f"Invalid JSON in configuration file: {e}")
        except Exception as e:
            raise ValidationError(f"Error loading configuration file: {e}")


class SecurityValidator:
    """Security-focused validation for benchmark components."""
    
    @staticmethod
    def validate_model_file(model_path: str, max_size_mb: int = 100) -> Path:
        """Validate model file for security risks.
        
        Args:
            model_path: Path to model file
            max_size_mb: Maximum allowed file size in MB
            
        Returns:
            Validated Path object
            
        Raises:
            ValidationError: If model file is unsafe
        """
        path = InputValidator.sanitize_filepath(model_path)
        
        if not path.exists():
            raise ValidationError(f"Model file not found: {model_path}")
        
        # Check file size
        size_mb = path.stat().st_size / (1024 * 1024)
        if size_mb > max_size_mb:
            raise ValidationError(f"Model file too large: {size_mb:.1f}MB > {max_size_mb}MB")
        
        # Check file extension
        allowed_extensions = {'.pkl', '.pth', '.pt', '.h5', '.onnx', '.pb'}
        if path.suffix.lower() not in allowed_extensions:
            raise ValidationError(f"Unsupported model file type: {path.suffix}")
        
        return path
    
    @staticmethod
    def validate_network_config(config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate network configuration for security.
        
        Args:
            config: Network configuration dictionary
            
        Returns:
            Validated configuration
            
        Raises:
            ValidationError: If configuration is unsafe
        """
        if "host" in config:
            host = config["host"]
            # Block potentially dangerous hosts
            dangerous_hosts = ["0.0.0.0", "*", ""]
            if host in dangerous_hosts:
                raise ValidationError(f"Dangerous host configuration: {host}")
        
        if "port" in config:
            port = config["port"]
            if not isinstance(port, int) or not (1024 <= port <= 65535):
                raise ValidationError(f"Port must be integer in range 1024-65535, got {port}")
        
        return config
    
    @staticmethod
    def check_resource_limits(memory_mb: Optional[int] = None, 
                            cpu_percent: Optional[float] = None,
                            gpu_memory_mb: Optional[int] = None) -> bool:
        """Check if resource usage is within safe limits.
        
        Args:
            memory_mb: Memory usage in MB
            cpu_percent: CPU usage percentage
            gpu_memory_mb: GPU memory usage in MB
            
        Returns:
            True if within limits
            
        Raises:
            ValidationError: If resource usage is excessive
        """
        if memory_mb is not None and memory_mb > 8192:  # 8GB limit
            raise ValidationError(f"Memory usage too high: {memory_mb}MB")
        
        if cpu_percent is not None and cpu_percent > 90.0:
            raise ValidationError(f"CPU usage too high: {cpu_percent}%")
        
        if gpu_memory_mb is not None and gpu_memory_mb > 12288:  # 12GB GPU limit
            raise ValidationError(f"GPU memory usage too high: {gpu_memory_mb}MB")
        
        return True
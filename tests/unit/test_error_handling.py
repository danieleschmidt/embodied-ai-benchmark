"""Tests for comprehensive error handling and recovery systems."""

import pytest
from unittest.mock import Mock, patch, MagicMock
import asyncio
from concurrent.futures import ThreadPoolExecutor
import time

from embodied_ai_benchmark.utils.error_handling import (
    ErrorHandler,
    ErrorRecoveryStrategy,
    NetworkErrorRecovery,
    ResourceErrorRecovery,
    ComputationErrorRecovery
)


class TestErrorRecoveryStrategy:
    """Test base error recovery strategy functionality."""
    
    def test_base_strategy_interface(self):
        """Test base strategy interface."""
        strategy = ErrorRecoveryStrategy()
        
        # Should raise NotImplementedError for abstract methods
        with pytest.raises(NotImplementedError):
            strategy.can_recover(Exception("test"))
        
        with pytest.raises(NotImplementedError):
            strategy.recover(Exception("test"), {})
    
    def test_strategy_registration(self):
        """Test strategy registration and retrieval."""
        custom_strategy = ErrorRecoveryStrategy()
        custom_strategy.can_recover = Mock(return_value=True)
        custom_strategy.recover = Mock(return_value={"status": "recovered"})
        
        # Should be able to register custom strategies
        assert custom_strategy is not None


class TestNetworkErrorRecovery:
    """Test network error recovery functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.recovery = NetworkErrorRecovery()
    
    def test_network_error_detection(self):
        """Test detection of network-related errors."""
        # Test connection errors
        connection_error = ConnectionError("Connection failed")
        assert self.recovery.can_recover(connection_error)
        
        # Test timeout errors
        timeout_error = TimeoutError("Request timed out")
        assert self.recovery.can_recover(timeout_error)
        
        # Test non-network errors
        value_error = ValueError("Invalid value")
        assert not self.recovery.can_recover(value_error)
    
    @patch('time.sleep')
    @patch('requests.get')
    def test_connection_retry_recovery(self, mock_get, mock_sleep):
        """Test connection retry recovery strategy."""
        # Mock initial failures followed by success
        mock_get.side_effect = [
            ConnectionError("Connection failed"),
            ConnectionError("Connection failed"), 
            Mock(status_code=200, json=lambda: {"status": "ok"})
        ]
        
        connection_error = ConnectionError("Connection failed")
        context = {
            "operation": "api_call",
            "url": "http://example.com",
            "method": "GET"
        }
        
        result = self.recovery.recover(connection_error, context)
        
        assert result["status"] == "recovered"
        assert result["attempts"] == 3
        assert mock_get.call_count == 3
        assert mock_sleep.call_count == 2  # Sleep between retries
    
    def test_exponential_backoff(self):
        """Test exponential backoff in retry logic."""
        with patch('time.sleep') as mock_sleep:
            with patch('requests.get', side_effect=ConnectionError("Failed")):
                try:
                    self.recovery.recover(ConnectionError("test"), {
                        "operation": "api_call",
                        "url": "http://example.com"
                    })
                except Exception:
                    pass  # Expected to fail after max retries
                
                # Check that sleep times increase exponentially
                sleep_calls = [call[0][0] for call in mock_sleep.call_args_list]
                assert len(sleep_calls) > 1
                assert sleep_calls[1] > sleep_calls[0]  # Exponential increase
    
    def test_max_retry_limit(self):
        """Test maximum retry limit enforcement."""
        with patch('requests.get', side_effect=ConnectionError("Always fails")):
            connection_error = ConnectionError("Persistent failure")
            context = {"operation": "api_call", "url": "http://example.com"}
            
            result = self.recovery.recover(connection_error, context)
            
            assert result["status"] == "failed"
            assert result["attempts"] == 3  # Default max retries


class TestResourceErrorRecovery:
    """Test resource error recovery functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.recovery = ResourceErrorRecovery()
    
    def test_resource_error_detection(self):
        """Test detection of resource-related errors."""
        # Test memory errors
        memory_error = MemoryError("Out of memory")
        assert self.recovery.can_recover(memory_error)
        
        # Test permission errors
        permission_error = PermissionError("Access denied")
        assert self.recovery.can_recover(permission_error)
        
        # Test file not found errors
        file_error = FileNotFoundError("File missing")
        assert self.recovery.can_recover(file_error)
        
        # Test non-resource errors
        type_error = TypeError("Type mismatch")
        assert not self.recovery.can_recover(type_error)
    
    @patch('gc.collect')
    @patch('psutil.Process')
    def test_memory_recovery(self, mock_process, mock_gc):
        """Test memory error recovery through garbage collection."""
        # Mock memory usage information
        mock_memory_info = Mock()
        mock_memory_info.rss = 1024 * 1024 * 512  # 512MB
        mock_process.return_value.memory_info.return_value = mock_memory_info
        
        memory_error = MemoryError("Out of memory")
        context = {"operation": "data_processing", "data_size": "large"}
        
        result = self.recovery.recover(memory_error, context)
        
        assert result["status"] == "recovered"
        assert result["recovery_action"] == "memory_cleanup"
        assert mock_gc.call_count >= 1  # Garbage collection should be called
    
    @patch('os.makedirs')
    @patch('os.path.exists')
    def test_file_system_recovery(self, mock_exists, mock_makedirs):
        """Test file system error recovery."""
        # Mock missing directory
        mock_exists.return_value = False
        
        file_error = FileNotFoundError("Directory not found: /tmp/missing")
        context = {
            "operation": "file_write",
            "path": "/tmp/missing/file.txt"
        }
        
        result = self.recovery.recover(file_error, context)
        
        assert result["status"] == "recovered"
        assert result["recovery_action"] == "create_directory"
        mock_makedirs.assert_called_once_with("/tmp/missing", exist_ok=True)
    
    def test_permission_error_handling(self):
        """Test handling of permission errors."""
        permission_error = PermissionError("Access denied to /protected/file")
        context = {"operation": "file_read", "path": "/protected/file"}
        
        result = self.recovery.recover(permission_error, context)
        
        # Should suggest alternative approaches but not actually recover
        assert result["status"] == "failed"
        assert "suggestions" in result
        assert len(result["suggestions"]) > 0


class TestComputationErrorRecovery:
    """Test computation error recovery functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.recovery = ComputationErrorRecovery()
    
    def test_computation_error_detection(self):
        """Test detection of computation-related errors."""
        # Test numerical errors
        overflow_error = OverflowError("Numerical overflow")
        assert self.recovery.can_recover(overflow_error)
        
        zero_div_error = ZeroDivisionError("Division by zero")
        assert self.recovery.can_recover(zero_div_error)
        
        # Test non-computation errors
        import_error = ImportError("Module not found")
        assert not self.recovery.can_recover(import_error)
    
    def test_numerical_overflow_recovery(self):
        """Test recovery from numerical overflow errors."""
        overflow_error = OverflowError("Result too large")
        context = {
            "operation": "matrix_multiplication",
            "parameters": {"precision": "double"}
        }
        
        result = self.recovery.recover(overflow_error, context)
        
        assert result["status"] == "recovered"
        assert result["recovery_action"] == "reduce_precision"
        assert "adjusted_parameters" in result
    
    def test_zero_division_recovery(self):
        """Test recovery from division by zero errors."""
        zero_div_error = ZeroDivisionError("Division by zero")
        context = {
            "operation": "normalization",
            "parameters": {"denominator": 0.0}
        }
        
        result = self.recovery.recover(zero_div_error, context)
        
        assert result["status"] == "recovered"  
        assert result["recovery_action"] == "add_epsilon"
        assert result["adjusted_parameters"]["denominator"] > 0.0
    
    def test_algorithm_fallback(self):
        """Test fallback to alternative algorithms."""
        computation_error = RuntimeError("Algorithm convergence failed")
        context = {
            "operation": "optimization",
            "algorithm": "gradient_descent",
            "parameters": {"learning_rate": 0.1}
        }
        
        result = self.recovery.recover(computation_error, context)
        
        assert result["status"] == "recovered"
        assert result["recovery_action"] == "algorithm_fallback"
        assert result["fallback_algorithm"] != "gradient_descent"


class TestErrorHandler:
    """Test main error handler functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.handler = ErrorHandler()
    
    def test_handler_initialization(self):
        """Test error handler initialization."""
        assert len(self.handler.recovery_strategies) == 3  # Network, Resource, Computation
        assert self.handler.max_recovery_attempts == 3
        assert len(self.handler.error_history) == 0
    
    def test_strategy_registration(self):
        """Test custom strategy registration."""
        custom_strategy = Mock()
        custom_strategy.can_recover = Mock(return_value=True)
        custom_strategy.recover = Mock(return_value={"status": "custom_recovered"})
        
        self.handler.register_strategy(custom_strategy)
        
        assert custom_strategy in self.handler.recovery_strategies
    
    def test_error_handling_with_recovery(self):
        """Test successful error handling with recovery."""
        # Mock a network error that can be recovered
        with patch.object(self.handler.recovery_strategies[0], 'can_recover', return_value=True):
            with patch.object(self.handler.recovery_strategies[0], 'recover', 
                            return_value={"status": "recovered", "attempts": 2}):
                
                error = ConnectionError("Network failure")
                context = {"operation": "data_fetch"}
                
                result = self.handler.handle_error(error, context)
                
                assert result["recovered"] is True
                assert result["recovery_result"]["status"] == "recovered"
                assert len(self.handler.error_history) == 1
    
    def test_error_handling_without_recovery(self):
        """Test error handling when no recovery is possible."""
        # Create error that no strategy can recover
        unrecoverable_error = RuntimeError("Unrecoverable error")
        context = {"operation": "critical_task"}
        
        result = self.handler.handle_error(unrecoverable_error, context)
        
        assert result["recovered"] is False
        assert result["error_type"] == "RuntimeError"
        assert len(self.handler.error_history) == 1
    
    def test_multiple_recovery_attempts(self):
        """Test multiple recovery attempts for persistent errors."""
        recovery_strategy = Mock()
        recovery_strategy.can_recover = Mock(return_value=True)
        
        # First two attempts fail, third succeeds
        recovery_strategy.recover = Mock(side_effect=[
            {"status": "failed", "attempts": 1},
            {"status": "failed", "attempts": 2}, 
            {"status": "recovered", "attempts": 3}
        ])
        
        self.handler.register_strategy(recovery_strategy)
        
        error = Exception("Persistent error")
        context = {"operation": "retry_test"}
        
        result = self.handler.handle_error(error, context)
        
        assert result["recovered"] is True
        assert recovery_strategy.recover.call_count == 3
    
    def test_max_recovery_attempts_limit(self):
        """Test enforcement of maximum recovery attempts."""
        recovery_strategy = Mock()
        recovery_strategy.can_recover = Mock(return_value=True)
        recovery_strategy.recover = Mock(return_value={"status": "failed"})
        
        self.handler.register_strategy(recovery_strategy)
        
        error = Exception("Always failing error")
        context = {"operation": "fail_test"}
        
        result = self.handler.handle_error(error, context)
        
        assert result["recovered"] is False
        assert recovery_strategy.recover.call_count == self.handler.max_recovery_attempts
    
    def test_error_history_tracking(self):
        """Test error history tracking and analysis."""
        # Generate multiple errors
        errors = [
            ConnectionError("Network error 1"),
            MemoryError("Memory error 1"),
            ConnectionError("Network error 2"),
            OverflowError("Computation error 1")
        ]
        
        for error in errors:
            self.handler.handle_error(error, {"operation": "test"})
        
        history = self.handler.get_error_history()
        
        assert len(history) == 4
        
        # Check history entries have required fields
        for entry in history:
            assert "timestamp" in entry
            assert "error_type" in entry
            assert "error_message" in entry
            assert "context" in entry
            assert "recovered" in entry
    
    def test_error_statistics(self):
        """Test error statistics collection."""
        # Simulate various error scenarios
        errors = [
            (ConnectionError("Net 1"), True),   # Recovered
            (ConnectionError("Net 2"), False),  # Not recovered
            (MemoryError("Mem 1"), True),       # Recovered
            (ValueError("Val 1"), False),       # Not recovered
        ]
        
        for error, should_recover in errors:
            # Mock recovery based on should_recover flag
            for strategy in self.handler.recovery_strategies:
                strategy.can_recover = Mock(return_value=should_recover)
                strategy.recover = Mock(return_value={
                    "status": "recovered" if should_recover else "failed"
                })
            
            self.handler.handle_error(error, {"operation": "stats_test"})
        
        stats = self.handler.get_error_statistics()
        
        assert isinstance(stats, dict)
        assert "total_errors" in stats
        assert "recovery_rate" in stats
        assert "error_types" in stats
        assert "most_common_errors" in stats
        
        assert stats["total_errors"] == 4
        assert stats["recovery_rate"] == 0.5  # 2 recovered out of 4
    
    def test_concurrent_error_handling(self):
        """Test thread-safe concurrent error handling."""
        import threading
        import concurrent.futures
        
        errors_handled = []
        
        def handle_error_task(error_id):
            error = ConnectionError(f"Concurrent error {error_id}")
            result = self.handler.handle_error(error, {"operation": f"task_{error_id}"})
            errors_handled.append(result)
        
        # Handle multiple errors concurrently
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(handle_error_task, i) for i in range(10)]
            concurrent.futures.wait(futures)
        
        assert len(errors_handled) == 10
        assert len(self.handler.error_history) == 10
        
        # Verify all errors were handled (no race conditions)
        for result in errors_handled:
            assert "error_type" in result
            assert "recovered" in result
    
    def test_context_preservation(self):
        """Test preservation of error context through recovery."""
        error = ConnectionError("Context test error")
        rich_context = {
            "operation": "data_sync",
            "user_id": "user123",
            "session_id": "session456", 
            "retry_count": 1,
            "metadata": {"source": "test", "priority": "high"}
        }
        
        result = self.handler.handle_error(error, rich_context)
        
        # Context should be preserved in result and history
        assert result["context"] == rich_context
        
        history_entry = self.handler.error_history[-1]
        assert history_entry["context"] == rich_context
    
    @patch('logging.Logger.error')
    @patch('logging.Logger.info') 
    def test_error_logging(self, mock_info, mock_error):
        """Test proper error logging."""
        error = RuntimeError("Test logging error")
        context = {"operation": "logging_test"}
        
        self.handler.handle_error(error, context)
        
        # Should log the error
        mock_error.assert_called()
        
        # Check that error details are in the log message
        log_call = mock_error.call_args[0][0]
        assert "RuntimeError" in log_call
        assert "Test logging error" in log_call
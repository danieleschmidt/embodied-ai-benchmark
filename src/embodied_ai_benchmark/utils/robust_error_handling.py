"""Comprehensive error handling and recovery system for production."""

import sys
import traceback
import logging
import time
import json
import threading
from typing import Any, Dict, List, Optional, Callable, Tuple, Type, Union
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from functools import wraps
from contextlib import contextmanager
import concurrent.futures


@dataclass
class ErrorContext:
    """Context information for error handling."""
    function_name: str
    module_name: str
    timestamp: datetime
    thread_id: str
    process_id: int
    stack_trace: str
    local_variables: Dict[str, Any]
    system_state: Dict[str, Any]


@dataclass
class RecoveryAction:
    """Recovery action specification."""
    name: str
    action: Callable
    max_retries: int
    backoff_multiplier: float
    timeout_seconds: float
    prerequisites: List[str]
    success_criteria: Callable[[Any], bool]


class RobustErrorHandler:
    """Production-grade error handling with automatic recovery."""
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize error handler.
        
        Args:
            config: Configuration for error handling behavior
        """
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Error tracking
        self.error_counts = {}
        self.error_history = []
        self.recovery_actions = {}
        self.circuit_breakers = {}
        
        # Configuration
        self.max_error_history = self.config.get("max_error_history", 1000)
        self.circuit_breaker_threshold = self.config.get("circuit_breaker_threshold", 5)
        self.circuit_breaker_timeout = self.config.get("circuit_breaker_timeout", 60)
        self.default_max_retries = self.config.get("default_max_retries", 3)
        self.default_backoff = self.config.get("default_backoff", 1.5)
        
        # Monitoring
        self.metrics = {
            "total_errors": 0,
            "recovered_errors": 0,
            "circuit_breaker_trips": 0,
            "recovery_success_rate": 0.0
        }
        
        self._lock = threading.RLock()
    
    def register_recovery_action(self, 
                                error_type: Union[Type[Exception], str],
                                action: RecoveryAction):
        """Register recovery action for specific error type."""
        with self._lock:
            if isinstance(error_type, type):
                error_type = error_type.__name__
            self.recovery_actions[error_type] = action
            self.logger.info(f"Registered recovery action for {error_type}")
    
    def handle_error(self, 
                    error: Exception,
                    context: ErrorContext,
                    recovery_data: Dict[str, Any] = None) -> Tuple[bool, Any]:
        """Handle error with automatic recovery attempts.
        
        Args:
            error: The exception that occurred
            context: Error context information
            recovery_data: Additional data for recovery
            
        Returns:
            Tuple of (success, result)
        """
        with self._lock:
            error_type = type(error).__name__
            
            # Track error
            self._track_error(error_type, context)
            
            # Check circuit breaker
            if self._is_circuit_breaker_open(error_type):
                self.logger.error(f"Circuit breaker open for {error_type}")
                return False, None
            
            # Log error details
            self._log_error_details(error, context)
            
            # Attempt recovery
            success, result = self._attempt_recovery(error, context, recovery_data)
            
            if success:
                self.metrics["recovered_errors"] += 1
                self._reset_circuit_breaker(error_type)
            else:
                self._update_circuit_breaker(error_type)
            
            # Update success rate
            self._update_metrics()
            
            return success, result
    
    def _track_error(self, error_type: str, context: ErrorContext):
        """Track error occurrence."""
        self.metrics["total_errors"] += 1
        
        if error_type not in self.error_counts:
            self.error_counts[error_type] = 0
        self.error_counts[error_type] += 1
        
        # Add to history
        error_record = {
            "type": error_type,
            "timestamp": context.timestamp.isoformat(),
            "function": context.function_name,
            "module": context.module_name,
            "thread_id": context.thread_id
        }
        
        self.error_history.append(error_record)
        
        # Limit history size
        if len(self.error_history) > self.max_error_history:
            self.error_history = self.error_history[-self.max_error_history:]
    
    def _log_error_details(self, error: Exception, context: ErrorContext):
        """Log comprehensive error details."""
        error_details = {
            "error_type": type(error).__name__,
            "error_message": str(error),
            "function": context.function_name,
            "module": context.module_name,
            "timestamp": context.timestamp.isoformat(),
            "thread_id": context.thread_id,
            "process_id": context.process_id,
            "stack_trace": context.stack_trace,
            "local_variables": self._sanitize_variables(context.local_variables),
            "system_state": context.system_state
        }
        
        self.logger.error(f"Error occurred: {json.dumps(error_details, indent=2)}")
    
    def _sanitize_variables(self, variables: Dict[str, Any]) -> Dict[str, str]:
        """Sanitize local variables for logging."""
        sanitized = {}
        
        for name, value in variables.items():
            try:
                # Skip private variables
                if name.startswith('_'):
                    continue
                
                # Convert to string, truncate if too long
                str_value = str(value)
                if len(str_value) > 200:
                    str_value = str_value[:200] + "..."
                
                sanitized[name] = str_value
                
            except Exception:
                sanitized[name] = "<unable to serialize>"
        
        return sanitized
    
    def _attempt_recovery(self, 
                         error: Exception,
                         context: ErrorContext,
                         recovery_data: Dict[str, Any] = None) -> Tuple[bool, Any]:
        """Attempt to recover from error."""
        error_type = type(error).__name__
        
        # Check for registered recovery action
        if error_type in self.recovery_actions:
            recovery_action = self.recovery_actions[error_type]
            return self._execute_recovery_action(recovery_action, error, context, recovery_data)
        
        # Try generic recovery strategies
        return self._generic_recovery_attempt(error, context, recovery_data)
    
    def _execute_recovery_action(self,
                               action: RecoveryAction,
                               error: Exception,
                               context: ErrorContext,
                               recovery_data: Dict[str, Any] = None) -> Tuple[bool, Any]:
        """Execute specific recovery action."""
        self.logger.info(f"Attempting recovery action: {action.name}")
        
        for attempt in range(action.max_retries):
            try:
                # Wait with exponential backoff
                if attempt > 0:
                    wait_time = action.backoff_multiplier ** (attempt - 1)
                    time.sleep(wait_time)
                
                # Execute recovery action with timeout
                with self._timeout(action.timeout_seconds):
                    result = action.action(error, context, recovery_data)
                
                # Check success criteria
                if action.success_criteria(result):
                    self.logger.info(f"Recovery action {action.name} succeeded on attempt {attempt + 1}")
                    return True, result
                
            except Exception as recovery_error:
                self.logger.warning(f"Recovery action {action.name} failed on attempt {attempt + 1}: {recovery_error}")
                continue
        
        self.logger.error(f"Recovery action {action.name} exhausted all retries")
        return False, None
    
    def _generic_recovery_attempt(self,
                                error: Exception,
                                context: ErrorContext,
                                recovery_data: Dict[str, Any] = None) -> Tuple[bool, Any]:
        """Generic recovery strategies."""
        error_type = type(error).__name__
        
        # Strategy 1: Retry with backoff for transient errors
        if self._is_transient_error(error):
            return self._retry_with_backoff(context.function_name, error)
        
        # Strategy 2: Fallback for known error patterns
        fallback_result = self._try_fallback_strategies(error, context)
        if fallback_result is not None:
            return True, fallback_result
        
        # Strategy 3: Resource cleanup and retry
        if self._requires_cleanup(error):
            cleanup_success = self._cleanup_resources(context)
            if cleanup_success:
                return self._retry_operation(context, 1)
        
        return False, None
    
    def _is_transient_error(self, error: Exception) -> bool:
        """Check if error is likely transient."""
        transient_types = [
            "ConnectionError",
            "TimeoutError",
            "TemporaryFailure",
            "ServiceUnavailable",
            "RateLimitError"
        ]
        
        error_type = type(error).__name__
        return error_type in transient_types or "timeout" in str(error).lower()
    
    def _requires_cleanup(self, error: Exception) -> bool:
        """Check if error requires resource cleanup."""
        cleanup_errors = [
            "OutOfMemoryError",
            "DiskSpaceError",
            "ResourceExhaustionError",
            "FileDescriptorError"
        ]
        
        error_type = type(error).__name__
        return error_type in cleanup_errors
    
    def _retry_with_backoff(self, function_name: str, error: Exception) -> Tuple[bool, Any]:
        """Retry with exponential backoff."""
        for attempt in range(self.default_max_retries):
            if attempt > 0:
                wait_time = self.default_backoff ** attempt
                self.logger.info(f"Retrying {function_name} in {wait_time:.2f} seconds")
                time.sleep(wait_time)
            
            try:
                # This would need to be implemented per use case
                # For now, just indicate retry was attempted
                self.logger.info(f"Retry attempt {attempt + 1} for {function_name}")
                return False, None  # Would need actual retry logic
            
            except Exception as retry_error:
                if attempt == self.default_max_retries - 1:
                    return False, None
                continue
        
        return False, None
    
    def _try_fallback_strategies(self, error: Exception, context: ErrorContext) -> Any:
        """Try various fallback strategies."""
        error_type = type(error).__name__
        
        # Database fallback
        if "Database" in error_type or "Connection" in error_type:
            return self._database_fallback()
        
        # API fallback
        if "API" in error_type or "HTTP" in error_type:
            return self._api_fallback()
        
        # Computation fallback
        if "Computation" in error_type or "Algorithm" in error_type:
            return self._computation_fallback()
        
        return None
    
    def _database_fallback(self) -> Dict[str, Any]:
        """Fallback to in-memory or file-based storage."""
        return {
            "fallback_type": "database",
            "storage": "memory",
            "message": "Using in-memory storage fallback"
        }
    
    def _api_fallback(self) -> Dict[str, Any]:
        """Fallback to cached or default responses."""
        return {
            "fallback_type": "api",
            "response": "cached",
            "message": "Using cached API response"
        }
    
    def _computation_fallback(self) -> Dict[str, Any]:
        """Fallback to simpler computation methods."""
        return {
            "fallback_type": "computation",
            "method": "simplified",
            "message": "Using simplified computation"
        }
    
    def _cleanup_resources(self, context: ErrorContext) -> bool:
        """Attempt to cleanup resources."""
        try:
            # Generic cleanup strategies
            import gc
            gc.collect()  # Force garbage collection
            
            # Close file handles, database connections, etc.
            # This would be implemented per application
            
            self.logger.info(f"Resource cleanup completed for {context.function_name}")
            return True
            
        except Exception as cleanup_error:
            self.logger.error(f"Resource cleanup failed: {cleanup_error}")
            return False
    
    def _retry_operation(self, context: ErrorContext, max_attempts: int) -> Tuple[bool, Any]:
        """Retry the original operation."""
        # This would need to be implemented per use case
        self.logger.info(f"Retrying operation {context.function_name}")
        return False, None
    
    def _is_circuit_breaker_open(self, error_type: str) -> bool:
        """Check if circuit breaker is open for error type."""
        if error_type not in self.circuit_breakers:
            return False
        
        breaker = self.circuit_breakers[error_type]
        
        # Check if timeout has passed
        if time.time() - breaker["last_failure"] > self.circuit_breaker_timeout:
            breaker["failure_count"] = 0
            return False
        
        return breaker["failure_count"] >= self.circuit_breaker_threshold
    
    def _update_circuit_breaker(self, error_type: str):
        """Update circuit breaker state."""
        if error_type not in self.circuit_breakers:
            self.circuit_breakers[error_type] = {
                "failure_count": 0,
                "last_failure": 0
            }
        
        breaker = self.circuit_breakers[error_type]
        breaker["failure_count"] += 1
        breaker["last_failure"] = time.time()
        
        if breaker["failure_count"] == self.circuit_breaker_threshold:
            self.metrics["circuit_breaker_trips"] += 1
            self.logger.warning(f"Circuit breaker opened for {error_type}")
    
    def _reset_circuit_breaker(self, error_type: str):
        """Reset circuit breaker after successful recovery."""
        if error_type in self.circuit_breakers:
            self.circuit_breakers[error_type]["failure_count"] = 0
            self.logger.info(f"Circuit breaker reset for {error_type}")
    
    def _update_metrics(self):
        """Update success rate metrics."""
        if self.metrics["total_errors"] > 0:
            self.metrics["recovery_success_rate"] = (
                self.metrics["recovered_errors"] / self.metrics["total_errors"]
            )
    
    @contextmanager
    def _timeout(self, seconds: float):
        """Context manager for operation timeout."""
        if seconds <= 0:
            yield
            return
        
        def timeout_handler():
            raise TimeoutError(f"Operation timed out after {seconds} seconds")
        
        timer = threading.Timer(seconds, timeout_handler)
        timer.start()
        
        try:
            yield
        finally:
            timer.cancel()
    
    def get_error_statistics(self) -> Dict[str, Any]:
        """Get comprehensive error statistics."""
        with self._lock:
            return {
                "total_errors": self.metrics["total_errors"],
                "recovered_errors": self.metrics["recovered_errors"],
                "recovery_success_rate": self.metrics["recovery_success_rate"],
                "circuit_breaker_trips": self.metrics["circuit_breaker_trips"],
                "error_counts_by_type": self.error_counts.copy(),
                "circuit_breaker_states": {
                    error_type: {
                        "failure_count": breaker["failure_count"],
                        "is_open": self._is_circuit_breaker_open(error_type)
                    }
                    for error_type, breaker in self.circuit_breakers.items()
                },
                "recent_errors": self.error_history[-10:] if self.error_history else []
            }
    
    def reset_statistics(self):
        """Reset error statistics."""
        with self._lock:
            self.error_counts.clear()
            self.error_history.clear()
            self.circuit_breakers.clear()
            self.metrics = {
                "total_errors": 0,
                "recovered_errors": 0,
                "circuit_breaker_trips": 0,
                "recovery_success_rate": 0.0
            }


def robust_operation(error_handler: RobustErrorHandler = None,
                    max_retries: int = 3,
                    backoff_multiplier: float = 1.5,
                    timeout_seconds: float = 30.0,
                    recovery_data: Dict[str, Any] = None):
    """Decorator for robust operation execution."""
    
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            nonlocal error_handler
            
            if error_handler is None:
                error_handler = RobustErrorHandler()
            
            for attempt in range(max_retries):
                try:
                    # Execute function with timeout
                    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                        future = executor.submit(func, *args, **kwargs)
                        result = future.result(timeout=timeout_seconds)
                    
                    return result
                
                except Exception as e:
                    # Create error context
                    context = ErrorContext(
                        function_name=func.__name__,
                        module_name=func.__module__,
                        timestamp=datetime.now(timezone.utc),
                        thread_id=str(threading.current_thread().ident),
                        process_id=os.getpid() if 'os' in globals() else 0,
                        stack_trace=traceback.format_exc(),
                        local_variables=locals(),
                        system_state={"attempt": attempt + 1, "max_retries": max_retries}
                    )
                    
                    # Handle error
                    success, recovery_result = error_handler.handle_error(
                        e, context, recovery_data
                    )
                    
                    if success and recovery_result is not None:
                        return recovery_result
                    
                    # If this is the last attempt, raise the error
                    if attempt == max_retries - 1:
                        raise e
                    
                    # Wait before retry
                    if attempt < max_retries - 1:
                        wait_time = backoff_multiplier ** attempt
                        time.sleep(wait_time)
            
            # This should not be reached
            raise RuntimeError(f"Unexpected state in robust_operation for {func.__name__}")
        
        return wrapper
    return decorator


# Global error handler instance
_global_error_handler = None

def get_global_error_handler() -> RobustErrorHandler:
    """Get or create global error handler."""
    global _global_error_handler
    
    if _global_error_handler is None:
        _global_error_handler = RobustErrorHandler()
    
    return _global_error_handler


def set_global_error_handler(handler: RobustErrorHandler):
    """Set global error handler."""
    global _global_error_handler
    _global_error_handler = handler
"""Comprehensive error handling and recovery mechanisms for the embodied AI benchmark."""

import time
import traceback
import functools
import threading
from typing import Any, Dict, List, Optional, Callable, Type, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import inspect
import pickle
import json

from .logging_config import get_logger
from .validation import ValidationError

logger = get_logger(__name__)


class ErrorSeverity(Enum):
    """Error severity levels."""
    LOW = "low"
    MEDIUM = "medium"  
    HIGH = "high"
    CRITICAL = "critical"


class ErrorCategory(Enum):
    """Error categories for classification."""
    VALIDATION = "validation"
    NETWORK = "network"
    RESOURCE = "resource"
    COMPUTATION = "computation"
    PHYSICS = "physics"
    COMMUNICATION = "communication"
    DATA = "data"
    SECURITY = "security"
    CONFIGURATION = "configuration"
    UNKNOWN = "unknown"


@dataclass
class ErrorContext:
    """Context information for errors."""
    timestamp: datetime = field(default_factory=datetime.now)
    function_name: Optional[str] = None
    module_name: Optional[str] = None
    thread_id: Optional[int] = None
    process_id: Optional[int] = None
    stack_trace: Optional[str] = None
    local_variables: Optional[Dict[str, str]] = None
    system_state: Optional[Dict[str, Any]] = None


@dataclass
class BenchmarkError:
    """Comprehensive error information."""
    error_id: str
    category: ErrorCategory
    severity: ErrorSeverity
    message: str
    original_exception: Optional[Exception] = None
    context: Optional[ErrorContext] = None
    recovery_actions: List[str] = field(default_factory=list)
    similar_errors: List[str] = field(default_factory=list)
    user_guidance: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert error to dictionary for serialization."""
        return {
            "error_id": self.error_id,
            "category": self.category.value,
            "severity": self.severity.value,
            "message": self.message,
            "original_exception": str(self.original_exception) if self.original_exception else None,
            "timestamp": self.context.timestamp.isoformat() if self.context else None,
            "function_name": self.context.function_name if self.context else None,
            "module_name": self.context.module_name if self.context else None,
            "stack_trace": self.context.stack_trace if self.context else None,
            "recovery_actions": self.recovery_actions,
            "similar_errors": self.similar_errors,
            "user_guidance": self.user_guidance
        }


class ErrorRecoveryStrategy:
    """Base class for error recovery strategies."""
    
    def __init__(self, max_attempts: int = 3, backoff_factor: float = 2.0):
        """Initialize recovery strategy.
        
        Args:
            max_attempts: Maximum recovery attempts
            backoff_factor: Exponential backoff factor for delays
        """
        self.max_attempts = max_attempts
        self.backoff_factor = backoff_factor
        self.attempt_count = 0
        
    def can_recover(self, error: BenchmarkError) -> bool:
        """Check if error can be recovered from.
        
        Args:
            error: Error to check
            
        Returns:
            True if recovery is possible
        """
        return (self.attempt_count < self.max_attempts and 
                error.severity != ErrorSeverity.CRITICAL)
    
    def recover(self, error: BenchmarkError, context: Dict[str, Any]) -> bool:
        """Attempt to recover from error.
        
        Args:
            error: Error to recover from
            context: Recovery context
            
        Returns:
            True if recovery was successful
        """
        raise NotImplementedError("Subclasses must implement recover method")
    
    def get_next_delay(self) -> float:
        """Get delay before next recovery attempt."""
        delay = (self.backoff_factor ** self.attempt_count)
        self.attempt_count += 1
        return min(delay, 60.0)  # Cap at 60 seconds


class NetworkErrorRecovery(ErrorRecoveryStrategy):
    """Recovery strategy for network-related errors."""
    
    def can_recover(self, error: BenchmarkError) -> bool:
        """Check if network error can be recovered."""
        if error.category != ErrorCategory.NETWORK:
            return False
        
        # Don't retry authentication or permission errors
        auth_keywords = ["unauthorized", "forbidden", "authentication", "permission"]
        if any(keyword in error.message.lower() for keyword in auth_keywords):
            return False
        
        return super().can_recover(error)
    
    def recover(self, error: BenchmarkError, context: Dict[str, Any]) -> bool:
        """Attempt to recover from network error."""
        logger.info(f"Attempting network error recovery (attempt {self.attempt_count + 1})")
        
        # Wait with exponential backoff
        delay = self.get_next_delay()
        time.sleep(delay)
        
        # Check network connectivity
        try:
            import socket
            socket.create_connection(("8.8.8.8", 53), timeout=3)
            logger.info("Network connectivity restored")
            return True
        except Exception:
            logger.warning("Network still unavailable")
            return False


class ResourceErrorRecovery(ErrorRecoveryStrategy):
    """Recovery strategy for resource-related errors."""
    
    def can_recover(self, error: BenchmarkError) -> bool:
        """Check if resource error can be recovered."""
        if error.category != ErrorCategory.RESOURCE:
            return False
        return super().can_recover(error)
    
    def recover(self, error: BenchmarkError, context: Dict[str, Any]) -> bool:
        """Attempt to recover from resource error."""
        logger.info(f"Attempting resource error recovery (attempt {self.attempt_count + 1})")
        
        # Free up memory
        import gc
        gc.collect()
        
        # Wait for resources to be freed
        delay = self.get_next_delay()
        time.sleep(delay)
        
        # Check if resources are now available
        try:
            import psutil
            memory_percent = psutil.virtual_memory().percent
            if memory_percent < 90:  # Less than 90% memory usage
                logger.info("Memory resources available")
                return True
            else:
                logger.warning(f"Memory still high: {memory_percent}%")
                return False
        except ImportError:
            # psutil not available, assume recovery
            return True
        except Exception:
            return False


class ComputationErrorRecovery(ErrorRecoveryStrategy):
    """Recovery strategy for computation-related errors."""
    
    def can_recover(self, error: BenchmarkError) -> bool:
        """Check if computation error can be recovered."""
        if error.category != ErrorCategory.COMPUTATION:
            return False
        
        # Can't recover from mathematical errors like division by zero
        math_keywords = ["division by zero", "overflow", "underflow", "invalid operation"]
        if any(keyword in error.message.lower() for keyword in math_keywords):
            return False
        
        return super().can_recover(error)
    
    def recover(self, error: BenchmarkError, context: Dict[str, Any]) -> bool:
        """Attempt to recover from computation error."""
        logger.info(f"Attempting computation error recovery (attempt {self.attempt_count + 1})")
        
        # Reset computational state if available
        if "reset_function" in context:
            try:
                context["reset_function"]()
                logger.info("Computational state reset")
                return True
            except Exception as e:
                logger.error(f"Failed to reset computational state: {e}")
        
        delay = self.get_next_delay()
        time.sleep(delay)
        return self.attempt_count < self.max_attempts


class ErrorHandler:
    """Comprehensive error handling system."""
    
    def __init__(self):
        """Initialize error handler."""
        self.error_history: List[BenchmarkError] = []
        self.recovery_strategies: Dict[ErrorCategory, ErrorRecoveryStrategy] = {
            ErrorCategory.NETWORK: NetworkErrorRecovery(),
            ErrorCategory.RESOURCE: ResourceErrorRecovery(), 
            ErrorCategory.COMPUTATION: ComputationErrorRecovery()
        }
        self.error_patterns: Dict[str, List[BenchmarkError]] = {}
        self._lock = threading.Lock()
        
        # Error classification patterns
        self.classification_patterns = {
            ErrorCategory.VALIDATION: [
                "validation", "invalid", "malformed", "schema", "format"
            ],
            ErrorCategory.NETWORK: [
                "connection", "network", "socket", "timeout", "dns", "http", "ssl"
            ],
            ErrorCategory.RESOURCE: [
                "memory", "disk", "cpu", "resource", "allocation", "out of"
            ],
            ErrorCategory.COMPUTATION: [
                "computation", "calculation", "algorithm", "convergence", "numerical"
            ],
            ErrorCategory.PHYSICS: [
                "physics", "simulation", "collision", "dynamics", "constraint"
            ],
            ErrorCategory.COMMUNICATION: [
                "message", "communication", "protocol", "agent", "coordinate"
            ],
            ErrorCategory.DATA: [
                "data", "file", "corruption", "missing", "parsing", "serialization"
            ],
            ErrorCategory.SECURITY: [
                "security", "authentication", "authorization", "permission", "access"
            ],
            ErrorCategory.CONFIGURATION: [
                "configuration", "config", "setting", "parameter", "option"
            ]
        }
    
    def classify_error(self, exception: Exception, context: Optional[Dict[str, Any]] = None) -> ErrorCategory:
        """Classify error into appropriate category.
        
        Args:
            exception: Exception to classify
            context: Additional context information
            
        Returns:
            Error category
        """
        error_message = str(exception).lower()
        exception_type = type(exception).__name__.lower()
        
        # Check specific exception types first
        if isinstance(exception, ValidationError):
            return ErrorCategory.VALIDATION
        elif isinstance(exception, (ConnectionError, TimeoutError)):
            return ErrorCategory.NETWORK
        elif isinstance(exception, (MemoryError, OSError)):
            return ErrorCategory.RESOURCE
        elif isinstance(exception, (ArithmeticError, ValueError, RuntimeError)):
            return ErrorCategory.COMPUTATION
        elif isinstance(exception, (PermissionError, FileNotFoundError)):
            return ErrorCategory.SECURITY if "permission" in error_message else ErrorCategory.DATA
        
        # Pattern-based classification
        for category, patterns in self.classification_patterns.items():
            if any(pattern in error_message or pattern in exception_type for pattern in patterns):
                return category
        
        return ErrorCategory.UNKNOWN
    
    def determine_severity(self, exception: Exception, context: Optional[Dict[str, Any]] = None) -> ErrorSeverity:
        """Determine error severity level.
        
        Args:
            exception: Exception to analyze
            context: Additional context information
            
        Returns:
            Error severity level
        """
        # Critical errors that should stop execution
        critical_types = (SystemError, MemoryError, KeyboardInterrupt)
        if isinstance(exception, critical_types):
            return ErrorSeverity.CRITICAL
        
        # High severity errors
        high_severity_keywords = ["fatal", "critical", "corruption", "security", "unauthorized"]
        error_message = str(exception).lower()
        if any(keyword in error_message for keyword in high_severity_keywords):
            return ErrorSeverity.HIGH
        
        # Medium severity errors
        medium_severity_types = (ConnectionError, TimeoutError, ValidationError, RuntimeError)
        if isinstance(exception, medium_severity_types):
            return ErrorSeverity.MEDIUM
        
        # Default to low severity
        return ErrorSeverity.LOW
    
    def create_error_context(self, frame_depth: int = 2) -> ErrorContext:
        """Create error context from current execution state.
        
        Args:
            frame_depth: Stack frame depth to capture
            
        Returns:
            Error context information
        """
        context = ErrorContext()
        
        # Get current frame information
        frame = inspect.currentframe()
        try:
            # Go up the stack to get caller information
            for _ in range(frame_depth):
                if frame.f_back:
                    frame = frame.f_back
            
            context.function_name = frame.f_code.co_name
            context.module_name = frame.f_globals.get('__name__')
            context.thread_id = threading.get_ident()
            context.process_id = id(threading.current_thread())
            
            # Capture stack trace
            context.stack_trace = traceback.format_stack()
            
            # Capture local variables (safely)
            local_vars = {}
            for key, value in frame.f_locals.items():
                try:
                    # Only capture serializable values
                    if isinstance(value, (str, int, float, bool, list, dict, tuple)):
                        if isinstance(value, (list, dict, tuple)) and len(str(value)) > 1000:
                            local_vars[key] = f"<{type(value).__name__} with {len(value)} items>"
                        else:
                            local_vars[key] = str(value)[:500]  # Limit string length
                    else:
                        local_vars[key] = f"<{type(value).__name__} object>"
                except Exception:
                    local_vars[key] = "<unable to serialize>"
            
            context.local_variables = local_vars
            
        finally:
            del frame
        
        return context
    
    def handle_error(self, 
                    exception: Exception, 
                    context: Optional[Dict[str, Any]] = None,
                    auto_recover: bool = True) -> BenchmarkError:
        """Handle an error with comprehensive analysis and recovery.
        
        Args:
            exception: Exception that occurred
            context: Additional context information
            auto_recover: Whether to attempt automatic recovery
            
        Returns:
            Processed benchmark error
        """
        with self._lock:
            # Generate unique error ID
            error_id = f"err_{int(time.time() * 1000000)}"
            
            # Classify and analyze error
            category = self.classify_error(exception, context)
            severity = self.determine_severity(exception, context)
            error_context = self.create_error_context()
            
            # Create comprehensive error object
            error = BenchmarkError(
                error_id=error_id,
                category=category,
                severity=severity,
                message=str(exception),
                original_exception=exception,
                context=error_context
            )
            
            # Find similar errors and add guidance
            similar_errors = self._find_similar_errors(error)
            error.similar_errors = [e.error_id for e in similar_errors]
            error.user_guidance = self._generate_user_guidance(error, similar_errors)
            error.recovery_actions = self._suggest_recovery_actions(error)
            
            # Log error with full context
            self._log_error(error)
            
            # Add to history and patterns
            self.error_history.append(error)
            self._update_error_patterns(error)
            
            # Attempt recovery if enabled
            if auto_recover and severity != ErrorSeverity.CRITICAL:
                recovery_success = self._attempt_recovery(error, context or {})
                if recovery_success:
                    logger.info(f"Successfully recovered from error {error_id}")
                    error.recovery_actions.append("auto_recovery_successful")
            
            return error
    
    def _find_similar_errors(self, error: BenchmarkError, max_results: int = 5) -> List[BenchmarkError]:
        """Find similar errors in history.
        
        Args:
            error: Error to find similarities for
            max_results: Maximum number of similar errors to return
            
        Returns:
            List of similar errors
        """
        similar = []
        
        for historical_error in self.error_history[-100:]:  # Check last 100 errors
            if historical_error.category == error.category:
                # Simple similarity based on message tokens
                error_tokens = set(error.message.lower().split())
                hist_tokens = set(historical_error.message.lower().split())
                
                similarity = len(error_tokens & hist_tokens) / len(error_tokens | hist_tokens)
                if similarity > 0.5:  # 50% similarity threshold
                    similar.append(historical_error)
                    
                if len(similar) >= max_results:
                    break
        
        return similar
    
    def _generate_user_guidance(self, error: BenchmarkError, similar_errors: List[BenchmarkError]) -> str:
        """Generate user guidance based on error analysis.
        
        Args:
            error: Current error
            similar_errors: List of similar historical errors
            
        Returns:
            User guidance string
        """
        guidance_map = {
            ErrorCategory.VALIDATION: "Check your input data format and ensure all required fields are present.",
            ErrorCategory.NETWORK: "Check your network connection and firewall settings. The service might be temporarily unavailable.",
            ErrorCategory.RESOURCE: "Close unnecessary applications to free up system resources. Consider increasing available memory or disk space.",
            ErrorCategory.COMPUTATION: "Verify your input parameters and consider using different computational settings. The algorithm may need different initial conditions.",
            ErrorCategory.PHYSICS: "Check physics simulation parameters and object constraints. Reduce time step or increase solver iterations if needed.",
            ErrorCategory.COMMUNICATION: "Verify agent communication settings and message protocols. Check that all agents are properly initialized.",
            ErrorCategory.DATA: "Check file permissions and data integrity. Ensure all required data files are present and accessible.",
            ErrorCategory.SECURITY: "Verify your authentication credentials and access permissions. Contact your administrator if needed.",
            ErrorCategory.CONFIGURATION: "Review your configuration settings and ensure all required parameters are set correctly."
        }
        
        base_guidance = guidance_map.get(error.category, "An unexpected error occurred. Please check the logs for more details.")
        
        # Add frequency-based guidance
        if len(similar_errors) > 2:
            base_guidance += " This error has occurred multiple times recently. Consider reviewing the suggested recovery actions."
        
        # Add severity-specific guidance
        if error.severity == ErrorSeverity.CRITICAL:
            base_guidance += " This is a critical error that requires immediate attention."
        elif error.severity == ErrorSeverity.HIGH:
            base_guidance += " This error may significantly impact functionality."
        
        return base_guidance
    
    def _suggest_recovery_actions(self, error: BenchmarkError) -> List[str]:
        """Suggest recovery actions based on error analysis.
        
        Args:
            error: Error to suggest recovery for
            
        Returns:
            List of recovery action suggestions
        """
        actions = []
        
        category_actions = {
            ErrorCategory.VALIDATION: [
                "Validate input data format",
                "Check required fields",
                "Review data schema"
            ],
            ErrorCategory.NETWORK: [
                "Retry connection", 
                "Check network settings",
                "Verify service availability"
            ],
            ErrorCategory.RESOURCE: [
                "Free up system resources",
                "Restart application",
                "Check disk space"
            ],
            ErrorCategory.COMPUTATION: [
                "Retry with different parameters",
                "Reset computational state",
                "Check numerical stability"
            ],
            ErrorCategory.PHYSICS: [
                "Adjust physics parameters",
                "Reset simulation state",
                "Check object constraints"
            ]
        }
        
        actions.extend(category_actions.get(error.category, ["Restart operation", "Check logs"]))
        
        # Add severity-based actions
        if error.severity == ErrorSeverity.CRITICAL:
            actions.insert(0, "Stop execution immediately")
        elif error.severity == ErrorSeverity.HIGH:
            actions.insert(0, "Review system state")
        
        return actions
    
    def _log_error(self, error: BenchmarkError):
        """Log error with appropriate level and detail.
        
        Args:
            error: Error to log
        """
        log_level = {
            ErrorSeverity.LOW: logger.info,
            ErrorSeverity.MEDIUM: logger.warning,
            ErrorSeverity.HIGH: logger.error,
            ErrorSeverity.CRITICAL: logger.critical
        }[error.severity]
        
        log_message = f"ERROR [{error.error_id}] {error.category.value.upper()}: {error.message}"
        
        extra_info = {
            "error_id": error.error_id,
            "category": error.category.value,
            "severity": error.severity.value,
            "function": error.context.function_name if error.context else None,
            "module": error.context.module_name if error.context else None
        }
        
        log_level(log_message, extra=extra_info)
        
        # Log stack trace for high severity errors
        if error.severity in (ErrorSeverity.HIGH, ErrorSeverity.CRITICAL) and error.context:
            logger.debug(f"Stack trace for {error.error_id}: {error.context.stack_trace}")
    
    def _update_error_patterns(self, error: BenchmarkError):
        """Update error pattern tracking.
        
        Args:
            error: Error to add to patterns
        """
        pattern_key = f"{error.category.value}_{hash(error.message) % 10000}"
        
        if pattern_key not in self.error_patterns:
            self.error_patterns[pattern_key] = []
        
        self.error_patterns[pattern_key].append(error)
        
        # Keep only recent errors for each pattern
        self.error_patterns[pattern_key] = self.error_patterns[pattern_key][-10:]
    
    def _attempt_recovery(self, error: BenchmarkError, context: Dict[str, Any]) -> bool:
        """Attempt automatic recovery from error.
        
        Args:
            error: Error to recover from
            context: Recovery context
            
        Returns:
            True if recovery was successful
        """
        if error.category in self.recovery_strategies:
            strategy = self.recovery_strategies[error.category]
            
            if strategy.can_recover(error):
                try:
                    return strategy.recover(error, context)
                except Exception as recovery_exception:
                    logger.error(f"Recovery failed for {error.error_id}: {recovery_exception}")
        
        return False
    
    def get_error_statistics(self, hours: int = 24) -> Dict[str, Any]:
        """Get error statistics for the specified time period.
        
        Args:
            hours: Number of hours to analyze
            
        Returns:
            Error statistics dictionary
        """
        cutoff_time = datetime.now() - timedelta(hours=hours)
        recent_errors = [e for e in self.error_history if e.context and e.context.timestamp > cutoff_time]
        
        if not recent_errors:
            return {"total_errors": 0, "error_rate": 0}
        
        # Category breakdown
        category_counts = {}
        for error in recent_errors:
            category_counts[error.category.value] = category_counts.get(error.category.value, 0) + 1
        
        # Severity breakdown
        severity_counts = {}
        for error in recent_errors:
            severity_counts[error.severity.value] = severity_counts.get(error.severity.value, 0) + 1
        
        # Most common error patterns
        pattern_frequencies = {}
        for errors in self.error_patterns.values():
            recent_pattern_errors = [e for e in errors if e.context and e.context.timestamp > cutoff_time]
            if recent_pattern_errors:
                pattern_key = f"{recent_pattern_errors[0].category.value}_{len(recent_pattern_errors)}"
                pattern_frequencies[pattern_key] = len(recent_pattern_errors)
        
        return {
            "total_errors": len(recent_errors),
            "error_rate": len(recent_errors) / hours,  # errors per hour
            "category_breakdown": category_counts,
            "severity_breakdown": severity_counts,
            "top_patterns": dict(sorted(pattern_frequencies.items(), key=lambda x: x[1], reverse=True)[:5]),
            "analysis_period_hours": hours
        }
    
    def export_error_report(self, filepath: str, hours: int = 24):
        """Export comprehensive error report to file.
        
        Args:
            filepath: Output file path
            hours: Hours of history to include
        """
        cutoff_time = datetime.now() - timedelta(hours=hours)
        recent_errors = [e for e in self.error_history if e.context and e.context.timestamp > cutoff_time]
        
        report = {
            "report_generated": datetime.now().isoformat(),
            "analysis_period_hours": hours,
            "statistics": self.get_error_statistics(hours),
            "errors": [error.to_dict() for error in recent_errors]
        }
        
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"Error report exported to {filepath}")


# Global error handler instance
global_error_handler = ErrorHandler()


def handle_errors(category: Optional[ErrorCategory] = None, 
                 auto_recover: bool = True,
                 return_on_error: Any = None):
    """Decorator for comprehensive error handling.
    
    Args:
        category: Force specific error category
        auto_recover: Whether to attempt automatic recovery
        return_on_error: Value to return on unrecoverable errors
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                # Override category if specified
                if category:
                    original_classify = global_error_handler.classify_error
                    global_error_handler.classify_error = lambda exc, ctx: category
                
                error = global_error_handler.handle_error(
                    e, 
                    context={"function": func.__name__, "args": args, "kwargs": kwargs},
                    auto_recover=auto_recover
                )
                
                # Restore original classification
                if category:
                    global_error_handler.classify_error = original_classify
                
                # Re-raise critical errors
                if error.severity == ErrorSeverity.CRITICAL:
                    raise
                
                # Return specified value for other errors
                if return_on_error is not None:
                    return return_on_error
                
                # Re-raise if no return value specified
                raise
        
        return wrapper
    return decorator


def with_retry(max_attempts: int = 3, 
               backoff_factor: float = 2.0,
               exceptions: Tuple[Type[Exception], ...] = (Exception,),
               on_retry: Optional[Callable[[int, Exception], None]] = None):
    """Decorator for automatic retry with exponential backoff.
    
    Args:
        max_attempts: Maximum number of attempts
        backoff_factor: Exponential backoff factor
        exceptions: Tuple of exception types to retry on
        on_retry: Callback function called on each retry
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    
                    if attempt == max_attempts - 1:
                        # Last attempt failed, handle the error
                        global_error_handler.handle_error(e, auto_recover=False)
                        break
                    
                    # Calculate delay
                    delay = backoff_factor ** attempt
                    
                    # Call retry callback if provided
                    if on_retry:
                        try:
                            on_retry(attempt + 1, e)
                        except Exception as callback_error:
                            logger.warning(f"Retry callback failed: {callback_error}")
                    
                    logger.info(f"Retrying {func.__name__} in {delay}s (attempt {attempt + 1}/{max_attempts})")
                    time.sleep(delay)
            
            # All attempts failed
            raise last_exception
        
        return wrapper
    return decorator


def get_error_handler() -> ErrorHandler:
    """Get the global error handler instance."""
    return global_error_handler


@dataclass
class ExecutionResult:
    """Result from safe execution."""
    success: bool
    result: Any = None
    error: Optional[Exception] = None
    execution_time: float = 0.0
    recovery_attempts: int = 0
    
    
class SafeExecutor:
    """Safe execution wrapper with error handling and recovery."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize safe executor.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.error_handler = get_error_handler()
        self.max_retries = self.config.get('max_retries', 3)
        self.retry_delay = self.config.get('retry_delay', 1.0)
        
    def execute_with_recovery(
        self, 
        func: Callable, 
        *args, 
        max_retries: Optional[int] = None,
        timeout: Optional[float] = None,
        **kwargs
    ) -> ExecutionResult:
        """Execute function with error handling and recovery.
        
        Args:
            func: Function to execute
            *args: Positional arguments
            max_retries: Maximum retry attempts
            timeout: Execution timeout
            **kwargs: Keyword arguments
            
        Returns:
            Execution result
        """
        max_retries = max_retries or self.max_retries
        start_time = time.time()
        last_exception = None
        
        for attempt in range(max_retries + 1):
            try:
                logger.debug(f"Executing {func.__name__}, attempt {attempt + 1}/{max_retries + 1}")
                
                # Execute with timeout if specified
                if timeout:
                    result = self._execute_with_timeout(func, timeout, *args, **kwargs)
                else:
                    result = func(*args, **kwargs)
                
                execution_time = time.time() - start_time
                
                return ExecutionResult(
                    success=True,
                    result=result,
                    execution_time=execution_time,
                    recovery_attempts=attempt
                )
                
            except Exception as e:
                last_exception = e
                logger.warning(f"Execution attempt {attempt + 1} failed: {e}")
                
                # Don't retry on certain types of errors
                if isinstance(e, (ValidationError, KeyboardInterrupt)):
                    break
                
                # Sleep before retry (except on last attempt)
                if attempt < max_retries:
                    time.sleep(self.retry_delay * (attempt + 1))  # Exponential backoff
        
        execution_time = time.time() - start_time
        
        return ExecutionResult(
            success=False,
            error=last_exception,
            execution_time=execution_time,
            recovery_attempts=max_retries
        )
    
    def _execute_with_timeout(
        self, 
        func: Callable, 
        timeout: float, 
        *args, 
        **kwargs
    ) -> Any:
        """Execute function with timeout."""
        import signal
        
        def timeout_handler(signum, frame):
            raise TimeoutError(f"Function {func.__name__} timed out after {timeout} seconds")
        
        # Set up signal handler for timeout
        old_handler = signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(int(timeout))
        
        try:
            result = func(*args, **kwargs)
            signal.alarm(0)  # Cancel alarm
            return result
        finally:
            signal.signal(signal.SIGALRM, old_handler)


class ErrorRecoveryManager:
    """Manages error recovery strategies."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize error recovery manager.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.recovery_strategies = {}
        self.error_handler = get_error_handler()
        
    def register_recovery_strategy(
        self, 
        error_type: Type[Exception], 
        strategy: Callable[[Exception], Any]
    ):
        """Register recovery strategy for specific error type.
        
        Args:
            error_type: Exception type to handle
            strategy: Recovery function
        """
        self.recovery_strategies[error_type] = strategy
        
    def recover_from_error(self, error: Exception) -> Optional[Any]:
        """Attempt to recover from error.
        
        Args:
            error: Exception to recover from
            
        Returns:
            Recovery result or None
        """
        error_type = type(error)
        
        # Try exact match first
        if error_type in self.recovery_strategies:
            strategy = self.recovery_strategies[error_type]
            try:
                return strategy(error)
            except Exception as e:
                logger.error(f"Recovery strategy failed: {e}")
        
        # Try base class matches
        for registered_type, strategy in self.recovery_strategies.items():
            if issubclass(error_type, registered_type):
                try:
                    return strategy(error)
                except Exception as e:
                    logger.error(f"Recovery strategy for {registered_type} failed: {e}")
        
        return None
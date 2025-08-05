"""Logging configuration for the embodied AI benchmark."""

import logging
import logging.handlers
import sys
import os
from datetime import datetime
from pathlib import Path
from typing import Optional


class BenchmarkLogger:
    """Centralized logging configuration for the benchmark."""
    
    _instance = None
    _initialized = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not self._initialized:
            self.setup_logging()
            self._initialized = True
    
    def setup_logging(self, 
                     log_level: str = "INFO",
                     log_dir: Optional[str] = None,
                     enable_file_logging: bool = True,
                     enable_console_logging: bool = True):
        """Setup comprehensive logging for the benchmark.
        
        Args:
            log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
            log_dir: Directory for log files (default: ./logs)
            enable_file_logging: Whether to log to files
            enable_console_logging: Whether to log to console
        """
        if log_dir is None:
            log_dir = Path.cwd() / "logs"
        else:
            log_dir = Path(log_dir)
        
        log_dir.mkdir(exist_ok=True)
        
        # Create formatter
        formatter = logging.Formatter(
            fmt='%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # Get root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(getattr(logging, log_level.upper()))
        
        # Clear existing handlers
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)
        
        # Console handler
        if enable_console_logging:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(getattr(logging, log_level.upper()))
            console_handler.setFormatter(formatter)
            root_logger.addHandler(console_handler)
        
        # File handlers
        if enable_file_logging:
            # Main log file with rotation
            main_log_file = log_dir / "benchmark.log"
            file_handler = logging.handlers.RotatingFileHandler(
                main_log_file,
                maxBytes=10*1024*1024,  # 10MB
                backupCount=5
            )
            file_handler.setLevel(getattr(logging, log_level.upper()))
            file_handler.setFormatter(formatter)
            root_logger.addHandler(file_handler)
            
            # Error log file
            error_log_file = log_dir / "benchmark_errors.log"
            error_handler = logging.handlers.RotatingFileHandler(
                error_log_file,
                maxBytes=5*1024*1024,  # 5MB
                backupCount=3
            )
            error_handler.setLevel(logging.ERROR)
            error_handler.setFormatter(formatter)
            root_logger.addHandler(error_handler)
            
            # Performance log file
            perf_log_file = log_dir / "benchmark_performance.log"
            perf_handler = logging.handlers.RotatingFileHandler(
                perf_log_file,
                maxBytes=5*1024*1024,  # 5MB
                backupCount=3
            )
            
            # Custom filter for performance logs
            class PerformanceFilter(logging.Filter):
                def filter(self, record):
                    return hasattr(record, 'performance') and record.performance
            
            perf_handler.addFilter(PerformanceFilter())
            perf_handler.setLevel(logging.INFO)
            perf_handler.setFormatter(formatter)
            root_logger.addHandler(perf_handler)
    
    def get_logger(self, name: str) -> logging.Logger:
        """Get a logger with the specified name."""
        return logging.getLogger(name)
    
    def log_performance(self, logger: logging.Logger, message: str, **kwargs):
        """Log performance-related information."""
        # Add performance flag to the log record
        record = logger.makeRecord(
            logger.name, logging.INFO, "", 0, message, (), None
        )
        record.performance = True
        for key, value in kwargs.items():
            setattr(record, key, value)
        logger.handle(record)
    
    def log_security_event(self, logger: logging.Logger, event_type: str, details: dict):
        """Log security-related events."""
        security_msg = f"SECURITY_EVENT: {event_type} - {details}"
        logger.warning(security_msg, extra={"security_event": True, "event_type": event_type, "details": details})
    
    def log_experiment_start(self, logger: logging.Logger, experiment_name: str, config: dict):
        """Log experiment start with configuration."""
        logger.info(f"EXPERIMENT_START: {experiment_name}", extra={
            "experiment": experiment_name,
            "config": config,
            "timestamp": datetime.now().isoformat()
        })
    
    def log_experiment_end(self, logger: logging.Logger, experiment_name: str, results: dict):
        """Log experiment completion with results."""
        logger.info(f"EXPERIMENT_END: {experiment_name}", extra={
            "experiment": experiment_name,
            "results": results,
            "timestamp": datetime.now().isoformat()
        })


# Global logger instance
benchmark_logger = BenchmarkLogger()


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance for the specified module."""
    return benchmark_logger.get_logger(name)


def setup_logging(**kwargs):
    """Setup logging with custom configuration."""
    benchmark_logger.setup_logging(**kwargs)
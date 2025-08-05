"""Utility modules for the embodied AI benchmark."""

from .logging_config import get_logger, setup_logging, benchmark_logger
from .validation import InputValidator, ValidationError, SecurityValidator
from .monitoring import performance_monitor, health_checker, BenchmarkMetrics
from .caching import LRUCache, AdaptiveCache, PersistentCache, cache_result, global_lru_cache
from .optimization import (
    BatchProcessor, ParallelEvaluator, VectorizedOperations, 
    MemoryOptimizer, JITCompiler, profile_performance
)
from .scalability import (
    LoadBalancer, DistributedBenchmark, AutoScaler, WorkerNode,
    LoadBalancingStrategy, global_load_balancer
)
from .i18n import (
    LocalizationManager, MessageCatalog, init_i18n, t, set_locale, 
    get_available_locales, default_i18n, messages
)
from .compliance import (
    ComplianceManager, ComplianceLevel, DataClassification, AuditLogEntry,
    DataRetentionPolicy, init_compliance, get_compliance_manager, audit_log
)
from .cross_platform import (
    PlatformUtils, ProcessManager, ResourceMonitor, SystemInfo,
    OperatingSystem, Architecture, get_platform_info, is_windows, 
    is_linux, is_macos, supports_multiprocessing, system_info,
    process_manager, resource_monitor
)

__all__ = [
    # Logging
    "get_logger", "setup_logging", "benchmark_logger",
    
    # Validation
    "InputValidator", "ValidationError", "SecurityValidator",
    
    # Monitoring
    "performance_monitor", "health_checker", "BenchmarkMetrics",
    
    # Caching
    "LRUCache", "AdaptiveCache", "PersistentCache", "cache_result", "global_lru_cache",
    
    # Optimization
    "BatchProcessor", "ParallelEvaluator", "VectorizedOperations", 
    "MemoryOptimizer", "JITCompiler", "profile_performance",
    
    # Scalability
    "LoadBalancer", "DistributedBenchmark", "AutoScaler", "WorkerNode",
    "LoadBalancingStrategy", "global_load_balancer",
    
    # Internationalization
    "LocalizationManager", "MessageCatalog", "init_i18n", "t", "set_locale",
    "get_available_locales", "default_i18n", "messages",
    
    # Compliance
    "ComplianceManager", "ComplianceLevel", "DataClassification", "AuditLogEntry",
    "DataRetentionPolicy", "init_compliance", "get_compliance_manager", "audit_log",
    
    # Cross-Platform
    "PlatformUtils", "ProcessManager", "ResourceMonitor", "SystemInfo",
    "OperatingSystem", "Architecture", "get_platform_info", "is_windows",
    "is_linux", "is_macos", "supports_multiprocessing", "system_info",
    "process_manager", "resource_monitor"
]
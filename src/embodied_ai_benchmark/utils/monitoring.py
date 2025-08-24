"""Monitoring and health check utilities for the embodied AI benchmark."""

import time
import psutil
import threading
import logging
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import numpy as np
from collections import deque, defaultdict
import warnings

logger = logging.getLogger(__name__)


@dataclass
class SystemMetrics:
    """System resource metrics."""
    timestamp: datetime
    cpu_percent: float
    memory_percent: float
    memory_used_mb: float
    disk_usage_percent: float
    gpu_memory_percent: Optional[float] = None
    gpu_memory_used_mb: Optional[float] = None
    network_io_mb_sent: float = 0.0
    network_io_mb_recv: float = 0.0


@dataclass
class BenchmarkMetrics:
    """Benchmark-specific performance metrics."""
    timestamp: datetime
    episode_id: int
    task_name: str
    agent_name: str
    steps_per_second: float
    reward_per_step: float
    memory_usage_mb: float
    inference_time_ms: float
    environment_step_time_ms: float
    success_rate: float


class PerformanceMonitor:
    """Real-time performance monitoring for benchmark execution."""
    
    def __init__(self, monitoring_interval: float = 1.0, history_size: int = 1000):
        """Initialize performance monitor.
        
        Args:
            monitoring_interval: Seconds between monitoring samples
            history_size: Maximum number of samples to keep in history
        """
        self.monitoring_interval = monitoring_interval
        self.history_size = history_size
        
        self.system_metrics_history = deque(maxlen=history_size)
        self.benchmark_metrics_history = deque(maxlen=history_size)
        self.alerts = []
        
        self._monitoring_thread = None
        self._stop_monitoring = threading.Event()
        self._monitoring_active = False
        
        # Performance thresholds
        self.thresholds = {
            "cpu_percent": 85.0,
            "memory_percent": 90.0,
            "gpu_memory_percent": 95.0,
            "min_steps_per_second": 1.0,
            "max_inference_time_ms": 1000.0
        }
        
        # GPU monitoring (optional)
        self.gpu_available = self._check_gpu_availability()
    
    def _check_gpu_availability(self) -> bool:
        """Check if GPU monitoring is available."""
        try:
            import pynvml
            pynvml.nvmlInit()
            return True
        except ImportError:
            logger.info("pynvml not available, GPU monitoring disabled")
            return False
        except Exception as e:
            logger.warning(f"GPU monitoring initialization failed: {e}")
            return False
    
    def start_monitoring(self):
        """Start background monitoring thread."""
        if self._monitoring_active:
            logger.warning("Monitoring already active")
            return
        
        self._stop_monitoring.clear()
        self._monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self._monitoring_thread.start()
        self._monitoring_active = True
        logger.info("Performance monitoring started")
    
    def stop_monitoring(self):
        """Stop background monitoring thread."""
        if not self._monitoring_active:
            return
        
        self._stop_monitoring.set()
        if self._monitoring_thread:
            self._monitoring_thread.join(timeout=5.0)
        self._monitoring_active = False
        logger.info("Performance monitoring stopped")
    
    def _monitoring_loop(self):
        """Main monitoring loop running in background thread."""
        while not self._stop_monitoring.wait(self.monitoring_interval):
            try:
                metrics = self._collect_system_metrics()
                self.system_metrics_history.append(metrics)
                self._check_system_alerts(metrics)
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
    
    def _collect_system_metrics(self) -> SystemMetrics:
        """Collect current system metrics."""
        # CPU and memory
        cpu_percent = psutil.cpu_percent(interval=None)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        # Network I/O
        net_io = psutil.net_io_counters()
        net_sent_mb = net_io.bytes_sent / (1024 * 1024) if net_io else 0.0
        net_recv_mb = net_io.bytes_recv / (1024 * 1024) if net_io else 0.0
        
        # GPU metrics (if available)
        gpu_memory_percent = None
        gpu_memory_used_mb = None
        
        if self.gpu_available:
            try:
                import pynvml
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                gpu_memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
                gpu_memory_percent = (gpu_memory.used / gpu_memory.total) * 100
                gpu_memory_used_mb = gpu_memory.used / (1024 * 1024)
            except Exception as e:
                logger.debug(f"GPU metrics collection failed: {e}")
        
        return SystemMetrics(
            timestamp=datetime.now(),
            cpu_percent=cpu_percent,
            memory_percent=memory.percent,
            memory_used_mb=memory.used / (1024 * 1024),
            disk_usage_percent=disk.percent,
            gpu_memory_percent=gpu_memory_percent,
            gpu_memory_used_mb=gpu_memory_used_mb,
            network_io_mb_sent=net_sent_mb,
            network_io_mb_recv=net_recv_mb
        )
    
    def _check_system_alerts(self, metrics: SystemMetrics):
        """Check system metrics against thresholds and generate alerts."""
        alerts = []
        
        if metrics.cpu_percent > self.thresholds["cpu_percent"]:
            alerts.append(f"High CPU usage: {metrics.cpu_percent:.1f}%")
        
        if metrics.memory_percent > self.thresholds["memory_percent"]:
            alerts.append(f"High memory usage: {metrics.memory_percent:.1f}%")
        
        if (metrics.gpu_memory_percent is not None and 
            metrics.gpu_memory_percent > self.thresholds["gpu_memory_percent"]):
            alerts.append(f"High GPU memory usage: {metrics.gpu_memory_percent:.1f}%")
        
        for alert in alerts:
            self.alerts.append({
                "timestamp": metrics.timestamp,
                "type": "system_alert",
                "message": alert,
                "severity": "warning"
            })
            logger.warning(f"PERFORMANCE_ALERT: {alert}")
    
    def record_benchmark_metrics(self, metrics: BenchmarkMetrics):
        """Record benchmark-specific performance metrics."""
        self.benchmark_metrics_history.append(metrics)
        self._check_benchmark_alerts(metrics)
    
    def _check_benchmark_alerts(self, metrics: BenchmarkMetrics):
        """Check benchmark metrics against thresholds."""
        alerts = []
        
        if metrics.steps_per_second < self.thresholds["min_steps_per_second"]:
            alerts.append(f"Low performance: {metrics.steps_per_second:.2f} steps/sec")
        
        if metrics.inference_time_ms > self.thresholds["max_inference_time_ms"]:
            alerts.append(f"Slow inference: {metrics.inference_time_ms:.1f}ms")
        
        for alert in alerts:
            self.alerts.append({
                "timestamp": metrics.timestamp,
                "type": "benchmark_alert",
                "message": alert,
                "severity": "warning",
                "task": metrics.task_name,
                "agent": metrics.agent_name
            })
            logger.warning(f"BENCHMARK_ALERT: {alert}")
    
    def get_system_summary(self, last_n_minutes: int = 5) -> Dict[str, Any]:
        """Get system performance summary for the last N minutes."""
        cutoff_time = datetime.now() - timedelta(minutes=last_n_minutes)
        
        recent_metrics = [
            m for m in self.system_metrics_history 
            if m.timestamp > cutoff_time
        ]
        
        if not recent_metrics:
            return {"error": "No recent metrics available"}
        
        cpu_values = [m.cpu_percent for m in recent_metrics]
        memory_values = [m.memory_percent for m in recent_metrics]
        
        summary = {
            "time_window_minutes": last_n_minutes,
            "sample_count": len(recent_metrics),
            "cpu": {
                "mean": np.mean(cpu_values),
                "max": np.max(cpu_values),
                "std": np.std(cpu_values)
            },
            "memory": {
                "mean": np.mean(memory_values),
                "max": np.max(memory_values),
                "std": np.std(memory_values)
            }
        }
        
        # Add GPU metrics if available
        gpu_values = [m.gpu_memory_percent for m in recent_metrics if m.gpu_memory_percent is not None]
        if gpu_values:
            summary["gpu_memory"] = {
                "mean": np.mean(gpu_values),
                "max": np.max(gpu_values),
                "std": np.std(gpu_values)
            }
        
        return summary
    
    def get_benchmark_summary(self, task_name: Optional[str] = None) -> Dict[str, Any]:
        """Get benchmark performance summary."""
        metrics = self.benchmark_metrics_history
        
        if task_name:
            metrics = [m for m in metrics if m.task_name == task_name]
        
        if not metrics:
            return {"error": "No benchmark metrics available"}
        
        steps_per_sec = [m.steps_per_second for m in metrics]
        inference_times = [m.inference_time_ms for m in metrics]
        success_rates = [m.success_rate for m in metrics]
        
        return {
            "sample_count": len(metrics),
            "task_name": task_name or "all_tasks",
            "performance": {
                "avg_steps_per_second": np.mean(steps_per_sec),
                "min_steps_per_second": np.min(steps_per_sec),
                "max_steps_per_second": np.max(steps_per_sec)
            },
            "inference": {
                "avg_time_ms": np.mean(inference_times),
                "min_time_ms": np.min(inference_times),
                "max_time_ms": np.max(inference_times)
            },
            "success": {
                "avg_success_rate": np.mean(success_rates),
                "min_success_rate": np.min(success_rates),
                "max_success_rate": np.max(success_rates)
            }
        }
    
    def get_recent_alerts(self, last_n_minutes: int = 10) -> List[Dict[str, Any]]:
        """Get recent alerts from the last N minutes."""
        cutoff_time = datetime.now() - timedelta(minutes=last_n_minutes)
        
        return [
            alert for alert in self.alerts
            if alert["timestamp"] > cutoff_time
        ]
    
    def clear_alerts(self):
        """Clear all stored alerts."""
        self.alerts.clear()
        logger.info("Performance alerts cleared")
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get comprehensive performance metrics summary.
        
        Returns:
            Dictionary containing system and benchmark metrics
        """
        system_summary = self.get_system_summary()
        benchmark_summary = self.get_benchmark_summary()
        recent_alerts = self.get_recent_alerts()
        
        return {
            "timestamp": datetime.now().isoformat(),
            "monitoring_active": self._monitoring_active,
            "system": system_summary,
            "benchmark": benchmark_summary,
            "alerts": recent_alerts,
            "metrics_count": {
                "system": len(self.system_metrics_history),
                "benchmark": len(self.benchmark_metrics_history)
            }
        }


class HealthChecker:
    """Health check system for benchmark components."""
    
    def __init__(self):
        self.health_checks: Dict[str, Callable] = {}
        self.last_check_results: Dict[str, Dict[str, Any]] = {}
    
    def register_health_check(self, name: str, check_function: Callable) -> None:
        """Register a health check function.
        
        Args:
            name: Unique name for the health check
            check_function: Function that returns (bool, str) for (healthy, message)
        """
        self.health_checks[name] = check_function
        logger.info(f"Registered health check: {name}")
    
    def run_health_checks(self) -> Dict[str, Dict[str, Any]]:
        """Run all registered health checks.
        
        Returns:
            Dictionary of health check results
        """
        results = {}
        
        for name, check_function in self.health_checks.items():
            try:
                start_time = time.time()
                healthy, message = check_function()
                check_time = time.time() - start_time
                
                results[name] = {
                    "healthy": healthy,
                    "message": message,
                    "check_time_ms": check_time * 1000,
                    "timestamp": datetime.now().isoformat()
                }
                
                if not healthy:
                    logger.warning(f"Health check failed: {name} - {message}")
                
            except Exception as e:
                results[name] = {
                    "healthy": False,
                    "message": f"Health check error: {str(e)}",
                    "check_time_ms": 0,
                    "timestamp": datetime.now().isoformat()
                }
                logger.error(f"Health check exception: {name} - {e}")
        
        self.last_check_results = results
        return results
    
    def get_overall_health(self) -> Dict[str, Any]:
        """Get overall system health status.
        
        Returns:
            Overall health summary
        """
        if not self.last_check_results:
            self.run_health_checks()
        
        total_checks = len(self.last_check_results)
        healthy_checks = sum(1 for result in self.last_check_results.values() if result["healthy"])
        
        overall_healthy = healthy_checks == total_checks
        health_percentage = (healthy_checks / max(1, total_checks)) * 100
        
        return {
            "overall_healthy": overall_healthy,
            "health_percentage": health_percentage,
            "total_checks": total_checks,
            "healthy_checks": healthy_checks,
            "failed_checks": total_checks - healthy_checks,
            "last_check_time": datetime.now().isoformat(),
            "details": self.last_check_results
        }


# Default health check functions
def check_memory_usage() -> tuple:
    """Check if memory usage is within acceptable limits."""
    memory = psutil.virtual_memory()
    if memory.percent > 90:
        return False, f"High memory usage: {memory.percent:.1f}%"
    return True, f"Memory usage normal: {memory.percent:.1f}%"


def check_disk_space() -> tuple:
    """Check if disk space is sufficient."""
    disk = psutil.disk_usage('/')
    if disk.percent > 90:
        return False, f"Low disk space: {disk.percent:.1f}% used"
    return True, f"Disk space normal: {disk.percent:.1f}% used"


def check_python_environment() -> tuple:
    """Check Python environment for common issues."""
    issues = []
    
    # Check for warnings
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        
        # Try importing key dependencies
        try:
            import numpy
            import torch
        except ImportError as e:
            issues.append(f"Missing dependency: {e}")
    
    if issues:
        return False, "; ".join(issues)
    
    return True, "Python environment healthy"


# Global instances
performance_monitor = PerformanceMonitor()
health_checker = HealthChecker()

# Register default health checks
health_checker.register_health_check("memory_usage", check_memory_usage)
health_checker.register_health_check("disk_space", check_disk_space)
health_checker.register_health_check("python_environment", check_python_environment)
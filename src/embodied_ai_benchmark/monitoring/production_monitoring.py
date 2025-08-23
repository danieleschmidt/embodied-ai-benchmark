"""Production monitoring system with metrics, health checks, and alerting."""

import time
import json
import logging
import threading
import psutil
import gc
from typing import Any, Dict, List, Optional, Callable, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from collections import defaultdict, deque
import concurrent.futures


@dataclass
class HealthCheckResult:
    """Result of a health check."""
    name: str
    status: str  # "healthy", "degraded", "unhealthy"
    message: str
    timestamp: datetime
    response_time_ms: float
    details: Dict[str, Any]


@dataclass
class MetricPoint:
    """Single metric data point."""
    name: str
    value: float
    timestamp: datetime
    labels: Dict[str, str]
    unit: str


class ProductionMonitor:
    """Comprehensive production monitoring system."""
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize production monitor.
        
        Args:
            config: Monitoring configuration
        """
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Health checks
        self.health_checks = {}
        self.health_check_results = {}
        self.health_check_interval = self.config.get("health_check_interval", 30)
        
        # Metrics collection
        self.metrics = defaultdict(list)
        self.metric_retention = self.config.get("metric_retention_hours", 24)
        self.max_metric_points = self.config.get("max_metric_points", 10000)
        
        # Performance tracking
        self.performance_metrics = {
            "request_count": 0,
            "error_count": 0,
            "total_response_time": 0.0,
            "avg_response_time": 0.0,
            "peak_memory_mb": 0.0,
            "cpu_usage_percent": 0.0
        }
        
        # Alerting
        self.alert_thresholds = self.config.get("alert_thresholds", {})
        self.alert_handlers = []
        
        # Threading
        self._monitoring_thread = None
        self._stop_monitoring = threading.Event()
        self._lock = threading.RLock()
        
        # Start monitoring
        self.start_monitoring()
    
    def register_health_check(self, 
                            name: str,
                            check_function: Callable[[], Tuple[bool, str, Dict[str, Any]]],
                            interval_seconds: int = None):
        """Register a health check.
        
        Args:
            name: Name of the health check
            check_function: Function that returns (is_healthy, message, details)
            interval_seconds: Override default interval for this check
        """
        with self._lock:
            self.health_checks[name] = {
                "function": check_function,
                "interval": interval_seconds or self.health_check_interval,
                "last_run": 0
            }
            self.logger.info(f"Registered health check: {name}")
    
    def record_metric(self, 
                     name: str, 
                     value: float,
                     labels: Dict[str, str] = None,
                     unit: str = ""):
        """Record a metric value.
        
        Args:
            name: Metric name
            value: Metric value
            labels: Optional labels for the metric
            unit: Unit of measurement
        """
        with self._lock:
            metric = MetricPoint(
                name=name,
                value=value,
                timestamp=datetime.now(timezone.utc),
                labels=labels or {},
                unit=unit
            )
            
            self.metrics[name].append(metric)
            
            # Limit metric history
            if len(self.metrics[name]) > self.max_metric_points:
                self.metrics[name] = self.metrics[name][-self.max_metric_points:]
    
    def record_request(self, response_time: float, success: bool = True):
        """Record request metrics."""
        with self._lock:
            self.performance_metrics["request_count"] += 1
            self.performance_metrics["total_response_time"] += response_time
            
            if not success:
                self.performance_metrics["error_count"] += 1
            
            # Update average response time
            self.performance_metrics["avg_response_time"] = (
                self.performance_metrics["total_response_time"] / 
                self.performance_metrics["request_count"]
            )
            
            # Record as time series metric
            self.record_metric("request_response_time", response_time, unit="ms")
            self.record_metric("request_count", 1, {"success": str(success)})
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get current health status."""
        with self._lock:
            overall_status = "healthy"
            unhealthy_checks = []
            
            for name, result in self.health_check_results.items():
                if result.status == "unhealthy":
                    overall_status = "unhealthy"
                    unhealthy_checks.append(name)
                elif result.status == "degraded" and overall_status == "healthy":
                    overall_status = "degraded"
            
            return {
                "overall_status": overall_status,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "checks": {name: asdict(result) for name, result in self.health_check_results.items()},
                "unhealthy_checks": unhealthy_checks,
                "total_checks": len(self.health_checks),
                "healthy_checks": len([r for r in self.health_check_results.values() if r.status == "healthy"])
            }
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get summary of collected metrics."""
        with self._lock:
            summary = {}
            
            for name, points in self.metrics.items():
                if not points:
                    continue
                
                values = [p.value for p in points]
                recent_points = [p for p in points if 
                               (datetime.now(timezone.utc) - p.timestamp).total_seconds() < 300]  # Last 5 minutes
                
                summary[name] = {
                    "current_value": values[-1] if values else 0,
                    "min_value": min(values),
                    "max_value": max(values),
                    "avg_value": sum(values) / len(values),
                    "recent_avg": sum(p.value for p in recent_points) / max(1, len(recent_points)),
                    "total_points": len(points),
                    "unit": points[-1].unit if points else ""
                }
            
            return summary
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics."""
        with self._lock:
            # Update system metrics
            self._update_system_metrics()
            
            return {
                **self.performance_metrics,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "error_rate": (
                    self.performance_metrics["error_count"] / 
                    max(1, self.performance_metrics["request_count"])
                ),
                "memory_usage_mb": psutil.Process().memory_info().rss / 1024 / 1024,
                "cpu_usage_percent": psutil.cpu_percent(interval=0.1)
            }
    
    def _update_system_metrics(self):
        """Update system-level metrics."""
        try:
            # Memory metrics
            memory_mb = psutil.Process().memory_info().rss / 1024 / 1024
            self.performance_metrics["peak_memory_mb"] = max(
                self.performance_metrics["peak_memory_mb"], 
                memory_mb
            )
            
            # CPU metrics
            cpu_percent = psutil.cpu_percent(interval=None)
            self.performance_metrics["cpu_usage_percent"] = cpu_percent
            
            # Record as time series
            self.record_metric("memory_usage", memory_mb, unit="MB")
            self.record_metric("cpu_usage", cpu_percent, unit="percent")
            
        except Exception as e:
            self.logger.warning(f"Failed to update system metrics: {e}")
    
    def start_monitoring(self):
        """Start background monitoring thread."""
        if self._monitoring_thread and self._monitoring_thread.is_alive():
            return
        
        self._stop_monitoring.clear()
        self._monitoring_thread = threading.Thread(
            target=self._monitoring_loop,
            name="ProductionMonitor",
            daemon=True
        )
        self._monitoring_thread.start()
        self.logger.info("Production monitoring started")
    
    def stop_monitoring(self):
        """Stop background monitoring."""
        if self._monitoring_thread:
            self._stop_monitoring.set()
            self._monitoring_thread.join(timeout=5)
            self.logger.info("Production monitoring stopped")
    
    def _monitoring_loop(self):
        """Main monitoring loop."""
        while not self._stop_monitoring.is_set():
            try:
                # Run health checks
                self._run_health_checks()
                
                # Update system metrics
                self._update_system_metrics()
                
                # Check alert conditions
                self._check_alerts()
                
                # Cleanup old metrics
                self._cleanup_old_metrics()
                
                # Sleep until next iteration
                self._stop_monitoring.wait(10)  # Check every 10 seconds
                
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
                self._stop_monitoring.wait(5)
    
    def _run_health_checks(self):
        """Run all registered health checks."""
        current_time = time.time()
        
        for name, check_config in list(self.health_checks.items()):
            try:
                # Check if it's time to run this check
                if current_time - check_config["last_run"] < check_config["interval"]:
                    continue
                
                check_config["last_run"] = current_time
                
                # Run health check with timeout
                start_time = time.time()
                
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                    future = executor.submit(check_config["function"])
                    is_healthy, message, details = future.result(timeout=30)
                
                response_time = (time.time() - start_time) * 1000
                
                # Determine status
                if is_healthy:
                    status = "healthy"
                elif "degraded" in message.lower():
                    status = "degraded"
                else:
                    status = "unhealthy"
                
                # Store result
                with self._lock:
                    self.health_check_results[name] = HealthCheckResult(
                        name=name,
                        status=status,
                        message=message,
                        timestamp=datetime.now(timezone.utc),
                        response_time_ms=response_time,
                        details=details or {}
                    )
                
            except Exception as e:
                # Health check failed
                with self._lock:
                    self.health_check_results[name] = HealthCheckResult(
                        name=name,
                        status="unhealthy",
                        message=f"Health check failed: {str(e)}",
                        timestamp=datetime.now(timezone.utc),
                        response_time_ms=30000,  # Timeout
                        details={"error": str(e)}
                    )
    
    def _check_alerts(self):
        """Check alert conditions and trigger notifications."""
        try:
            # Check performance thresholds
            perf_metrics = self.get_performance_metrics()
            
            alerts = []
            
            # Error rate alert
            error_rate_threshold = self.alert_thresholds.get("error_rate", 0.05)
            if perf_metrics["error_rate"] > error_rate_threshold:
                alerts.append({
                    "severity": "warning",
                    "metric": "error_rate",
                    "current_value": perf_metrics["error_rate"],
                    "threshold": error_rate_threshold,
                    "message": f"High error rate: {perf_metrics['error_rate']:.2%}"
                })
            
            # Response time alert
            response_time_threshold = self.alert_thresholds.get("avg_response_time", 1000)
            if perf_metrics["avg_response_time"] > response_time_threshold:
                alerts.append({
                    "severity": "warning",
                    "metric": "avg_response_time",
                    "current_value": perf_metrics["avg_response_time"],
                    "threshold": response_time_threshold,
                    "message": f"High response time: {perf_metrics['avg_response_time']:.2f}ms"
                })
            
            # Memory alert
            memory_threshold = self.alert_thresholds.get("memory_usage_mb", 1000)
            if perf_metrics["memory_usage_mb"] > memory_threshold:
                alerts.append({
                    "severity": "warning",
                    "metric": "memory_usage",
                    "current_value": perf_metrics["memory_usage_mb"],
                    "threshold": memory_threshold,
                    "message": f"High memory usage: {perf_metrics['memory_usage_mb']:.1f}MB"
                })
            
            # CPU alert
            cpu_threshold = self.alert_thresholds.get("cpu_usage_percent", 80)
            if perf_metrics["cpu_usage_percent"] > cpu_threshold:
                alerts.append({
                    "severity": "warning",
                    "metric": "cpu_usage",
                    "current_value": perf_metrics["cpu_usage_percent"],
                    "threshold": cpu_threshold,
                    "message": f"High CPU usage: {perf_metrics['cpu_usage_percent']:.1f}%"
                })
            
            # Health check alerts
            health_status = self.get_health_status()
            if health_status["overall_status"] != "healthy":
                alerts.append({
                    "severity": "error" if health_status["overall_status"] == "unhealthy" else "warning",
                    "metric": "health_status",
                    "current_value": health_status["overall_status"],
                    "threshold": "healthy",
                    "message": f"System health degraded: {health_status['overall_status']}",
                    "details": health_status["unhealthy_checks"]
                })
            
            # Trigger alerts
            for alert in alerts:
                self._trigger_alert(alert)
                
        except Exception as e:
            self.logger.error(f"Error checking alerts: {e}")
    
    def _trigger_alert(self, alert: Dict[str, Any]):
        """Trigger an alert notification."""
        try:
            self.logger.warning(f"ALERT: {alert['message']}")
            
            # Call registered alert handlers
            for handler in self.alert_handlers:
                try:
                    handler(alert)
                except Exception as e:
                    self.logger.error(f"Alert handler failed: {e}")
                    
        except Exception as e:
            self.logger.error(f"Failed to trigger alert: {e}")
    
    def register_alert_handler(self, handler: Callable[[Dict[str, Any]], None]):
        """Register an alert handler function."""
        self.alert_handlers.append(handler)
        self.logger.info(f"Registered alert handler: {handler.__name__}")
    
    def _cleanup_old_metrics(self):
        """Clean up old metric data."""
        try:
            with self._lock:
                cutoff_time = datetime.now(timezone.utc)
                cutoff_hours = self.metric_retention
                
                for name, points in list(self.metrics.items()):
                    # Filter out old points
                    self.metrics[name] = [
                        p for p in points 
                        if (cutoff_time - p.timestamp).total_seconds() < cutoff_hours * 3600
                    ]
                    
                    # Remove empty metric series
                    if not self.metrics[name]:
                        del self.metrics[name]
                        
        except Exception as e:
            self.logger.error(f"Error cleaning up metrics: {e}")
    
    def get_monitoring_report(self) -> Dict[str, Any]:
        """Get comprehensive monitoring report."""
        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "health_status": self.get_health_status(),
            "performance_metrics": self.get_performance_metrics(),
            "metrics_summary": self.get_metrics_summary(),
            "system_info": {
                "python_version": str(sys.version_info[:2]) if 'sys' in globals() else "unknown",
                "process_id": os.getpid() if 'os' in globals() else 0,
                "thread_count": threading.active_count(),
                "garbage_collection_stats": {
                    "collections": gc.get_count(),
                    "threshold": gc.get_threshold()
                }
            }
        }
    
    def export_metrics_prometheus(self) -> str:
        """Export metrics in Prometheus format."""
        output = []
        
        try:
            with self._lock:
                for name, points in self.metrics.items():
                    if not points:
                        continue
                    
                    latest_point = points[-1]
                    
                    # Create metric name (replace spaces and special chars)
                    metric_name = name.replace(" ", "_").replace("-", "_")
                    metric_name = "embodied_ai_" + metric_name
                    
                    # Add help text
                    output.append(f"# HELP {metric_name} {name}")
                    output.append(f"# TYPE {metric_name} gauge")
                    
                    # Add labels
                    labels = ""
                    if latest_point.labels:
                        label_pairs = [f'{k}="{v}"' for k, v in latest_point.labels.items()]
                        labels = "{" + ",".join(label_pairs) + "}"
                    
                    # Add metric value
                    output.append(f"{metric_name}{labels} {latest_point.value}")
            
            return "\n".join(output)
            
        except Exception as e:
            self.logger.error(f"Error exporting Prometheus metrics: {e}")
            return "# Error exporting metrics"


# Built-in health checks
def database_health_check() -> Tuple[bool, str, Dict[str, Any]]:
    """Check database connectivity."""
    try:
        import sqlite3
        conn = sqlite3.connect(':memory:')
        conn.execute('SELECT 1')
        conn.close()
        return True, "Database connectivity OK", {"type": "sqlite", "status": "connected"}
    except Exception as e:
        return False, f"Database check failed: {e}", {"error": str(e)}


def memory_health_check() -> Tuple[bool, str, Dict[str, Any]]:
    """Check memory usage."""
    try:
        memory_mb = psutil.Process().memory_info().rss / 1024 / 1024
        threshold = 2000  # 2GB threshold
        
        if memory_mb > threshold:
            return False, f"High memory usage: {memory_mb:.1f}MB", {"memory_mb": memory_mb}
        elif memory_mb > threshold * 0.8:
            return True, f"Memory usage elevated but acceptable: {memory_mb:.1f}MB", {"memory_mb": memory_mb}
        else:
            return True, f"Memory usage normal: {memory_mb:.1f}MB", {"memory_mb": memory_mb}
            
    except Exception as e:
        return False, f"Memory check failed: {e}", {"error": str(e)}


def disk_space_health_check() -> Tuple[bool, str, Dict[str, Any]]:
    """Check available disk space."""
    try:
        disk_usage = psutil.disk_usage('/')
        free_percent = (disk_usage.free / disk_usage.total) * 100
        
        if free_percent < 10:
            return False, f"Low disk space: {free_percent:.1f}% free", {"free_percent": free_percent}
        elif free_percent < 20:
            return True, f"Disk space getting low: {free_percent:.1f}% free", {"free_percent": free_percent}
        else:
            return True, f"Disk space OK: {free_percent:.1f}% free", {"free_percent": free_percent}
            
    except Exception as e:
        return False, f"Disk space check failed: {e}", {"error": str(e)}


# Global monitor instance
_global_monitor = None

def get_global_monitor() -> ProductionMonitor:
    """Get or create global monitor."""
    global _global_monitor
    
    if _global_monitor is None:
        _global_monitor = ProductionMonitor()
        
        # Register default health checks
        _global_monitor.register_health_check("database", database_health_check)
        _global_monitor.register_health_check("memory", memory_health_check)
        _global_monitor.register_health_check("disk_space", disk_space_health_check)
    
    return _global_monitor
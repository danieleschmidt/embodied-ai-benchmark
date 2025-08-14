"""
Advanced Performance Monitoring for Autonomous SDLC
"""

import time
import psutil
import threading
import asyncio
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import logging
import json
from collections import deque, defaultdict
import statistics

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """Container for performance metrics."""
    timestamp: datetime
    cpu_percent: float
    memory_mb: float
    memory_percent: float
    disk_io_read: int
    disk_io_write: int
    network_bytes_sent: int
    network_bytes_recv: int
    task_completion_time: Optional[float] = None
    task_throughput: Optional[float] = None
    error_rate: Optional[float] = None


class AdaptivePerformanceMonitor:
    """Advanced performance monitor with adaptive optimization."""
    
    def __init__(self, 
                 collection_interval: float = 1.0,
                 history_size: int = 3600,  # 1 hour at 1-second intervals
                 optimization_enabled: bool = True):
        """Initialize performance monitor.
        
        Args:
            collection_interval: Seconds between metric collections
            history_size: Number of historical metrics to keep
            optimization_enabled: Enable automatic optimization
        """
        self.collection_interval = collection_interval
        self.history_size = history_size
        self.optimization_enabled = optimization_enabled
        
        self.metrics_history: deque = deque(maxlen=history_size)
        self.performance_thresholds = {
            'cpu_percent': 80.0,
            'memory_percent': 85.0,
            'error_rate': 5.0,
            'throughput_degradation': 20.0
        }
        
        self.optimization_callbacks: List[Callable] = []
        self.is_running = False
        self.monitor_thread: Optional[threading.Thread] = None
        
        # Performance statistics
        self.task_times: deque = deque(maxlen=1000)
        self.task_counts = defaultdict(int)
        self.error_counts = defaultdict(int)
        
    def start_monitoring(self) -> None:
        """Start background performance monitoring."""
        if self.is_running:
            logger.warning("Performance monitor already running")
            return
            
        self.is_running = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        logger.info("Performance monitoring started")
        
    def stop_monitoring(self) -> None:
        """Stop background performance monitoring."""
        self.is_running = False
        if self.monitor_thread:
            self.monitor_thread.join()
        logger.info("Performance monitoring stopped")
        
    def _monitor_loop(self) -> None:
        """Main monitoring loop."""
        while self.is_running:
            try:
                metrics = self._collect_metrics()
                self.metrics_history.append(metrics)
                
                if self.optimization_enabled:
                    self._check_optimization_triggers(metrics)
                    
                time.sleep(self.collection_interval)
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(self.collection_interval)
                
    def _collect_metrics(self) -> PerformanceMetrics:
        """Collect current system metrics."""
        cpu_percent = psutil.cpu_percent(interval=None)
        memory = psutil.virtual_memory()
        disk_io = psutil.disk_io_counters()
        network_io = psutil.net_io_counters()
        
        # Calculate task throughput
        current_time = time.time()
        recent_tasks = [t for t in self.task_times if current_time - t < 60]  # Last minute
        throughput = len(recent_tasks) / 60.0 if recent_tasks else 0.0
        
        # Calculate error rate
        total_tasks = sum(self.task_counts.values())
        total_errors = sum(self.error_counts.values())
        error_rate = (total_errors / max(1, total_tasks)) * 100.0
        
        return PerformanceMetrics(
            timestamp=datetime.now(),
            cpu_percent=cpu_percent,
            memory_mb=memory.used / (1024 * 1024),
            memory_percent=memory.percent,
            disk_io_read=disk_io.read_bytes if disk_io else 0,
            disk_io_write=disk_io.write_bytes if disk_io else 0,
            network_bytes_sent=network_io.bytes_sent if network_io else 0,
            network_bytes_recv=network_io.bytes_recv if network_io else 0,
            task_throughput=throughput,
            error_rate=error_rate
        )
        
    def _check_optimization_triggers(self, metrics: PerformanceMetrics) -> None:
        """Check if performance optimization is needed."""
        triggers = []
        
        if metrics.cpu_percent > self.performance_thresholds['cpu_percent']:
            triggers.append(('high_cpu', metrics.cpu_percent))
            
        if metrics.memory_percent > self.performance_thresholds['memory_percent']:
            triggers.append(('high_memory', metrics.memory_percent))
            
        if metrics.error_rate and metrics.error_rate > self.performance_thresholds['error_rate']:
            triggers.append(('high_error_rate', metrics.error_rate))
            
        # Check throughput degradation
        if len(self.metrics_history) > 10:
            recent_throughput = [m.task_throughput for m in list(self.metrics_history)[-10:] if m.task_throughput]
            if recent_throughput:
                avg_recent = statistics.mean(recent_throughput)
                baseline_throughput = self._get_baseline_throughput()
                if baseline_throughput and avg_recent < baseline_throughput * 0.8:
                    triggers.append(('throughput_degradation', avg_recent))
                    
        if triggers:
            self._trigger_optimization(triggers)
            
    def _get_baseline_throughput(self) -> Optional[float]:
        """Calculate baseline throughput from historical data."""
        if len(self.metrics_history) < 100:
            return None
            
        throughputs = [m.task_throughput for m in self.metrics_history if m.task_throughput]
        return statistics.median(throughputs) if throughputs else None
        
    def _trigger_optimization(self, triggers: List[tuple]) -> None:
        """Trigger performance optimization."""
        logger.warning(f"Performance degradation detected: {triggers}")
        
        optimization_actions = []
        
        for trigger_type, value in triggers:
            if trigger_type == 'high_cpu':
                optimization_actions.append('reduce_parallelism')
                optimization_actions.append('enable_cpu_affinity')
                
            elif trigger_type == 'high_memory':
                optimization_actions.append('clear_caches')
                optimization_actions.append('reduce_batch_size')
                
            elif trigger_type == 'high_error_rate':
                optimization_actions.append('enable_circuit_breaker')
                optimization_actions.append('reduce_timeout')
                
            elif trigger_type == 'throughput_degradation':
                optimization_actions.append('scale_workers')
                optimization_actions.append('optimize_scheduling')
                
        # Execute optimization callbacks
        for callback in self.optimization_callbacks:
            try:
                callback(triggers, optimization_actions)
            except Exception as e:
                logger.error(f"Error in optimization callback: {e}")
                
    def add_optimization_callback(self, callback: Callable) -> None:
        """Add callback for performance optimization."""
        self.optimization_callbacks.append(callback)
        
    def record_task_completion(self, task_type: str, completion_time: float, success: bool = True) -> None:
        """Record task completion metrics."""
        self.task_times.append(time.time())
        self.task_counts[task_type] += 1
        
        if not success:
            self.error_counts[task_type] += 1
            
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary."""
        if not self.metrics_history:
            return {"status": "no_data"}
            
        recent_metrics = list(self.metrics_history)[-10:]  # Last 10 readings
        
        summary = {
            "current": {
                "cpu_percent": recent_metrics[-1].cpu_percent,
                "memory_percent": recent_metrics[-1].memory_percent,
                "memory_mb": recent_metrics[-1].memory_mb,
                "task_throughput": recent_metrics[-1].task_throughput,
                "error_rate": recent_metrics[-1].error_rate
            },
            "averages": {
                "cpu_percent": statistics.mean([m.cpu_percent for m in recent_metrics]),
                "memory_percent": statistics.mean([m.memory_percent for m in recent_metrics]),
                "task_throughput": statistics.mean([m.task_throughput for m in recent_metrics if m.task_throughput]) if any(m.task_throughput for m in recent_metrics) else 0.0
            },
            "trends": self._calculate_trends(),
            "recommendations": self._generate_recommendations()
        }
        
        return summary
        
    def _calculate_trends(self) -> Dict[str, str]:
        """Calculate performance trends."""
        if len(self.metrics_history) < 20:
            return {"status": "insufficient_data"}
            
        recent = list(self.metrics_history)[-10:]
        older = list(self.metrics_history)[-20:-10]
        
        trends = {}
        
        # CPU trend
        recent_cpu = statistics.mean([m.cpu_percent for m in recent])
        older_cpu = statistics.mean([m.cpu_percent for m in older])
        trends['cpu'] = 'increasing' if recent_cpu > older_cpu * 1.1 else 'decreasing' if recent_cpu < older_cpu * 0.9 else 'stable'
        
        # Memory trend
        recent_mem = statistics.mean([m.memory_percent for m in recent])
        older_mem = statistics.mean([m.memory_percent for m in older])
        trends['memory'] = 'increasing' if recent_mem > older_mem * 1.1 else 'decreasing' if recent_mem < older_mem * 0.9 else 'stable'
        
        return trends
        
    def _generate_recommendations(self) -> List[str]:
        """Generate performance optimization recommendations."""
        if not self.metrics_history:
            return []
            
        recommendations = []
        recent = self.metrics_history[-1]
        
        if recent.cpu_percent > 70:
            recommendations.append("Consider reducing CPU-intensive operations or scaling horizontally")
            
        if recent.memory_percent > 80:
            recommendations.append("Memory usage is high - consider implementing memory pooling or garbage collection tuning")
            
        if recent.error_rate and recent.error_rate > 2:
            recommendations.append("Error rate is elevated - review error handling and retry mechanisms")
            
        if recent.task_throughput and recent.task_throughput < 10:
            recommendations.append("Task throughput is low - consider optimizing task scheduling and execution")
            
        return recommendations


# Global performance monitor instance
_performance_monitor: Optional[AdaptivePerformanceMonitor] = None


def get_performance_monitor() -> AdaptivePerformanceMonitor:
    """Get global performance monitor instance."""
    global _performance_monitor
    if _performance_monitor is None:
        _performance_monitor = AdaptivePerformanceMonitor()
    return _performance_monitor


def start_performance_monitoring() -> None:
    """Start global performance monitoring."""
    monitor = get_performance_monitor()
    monitor.start_monitoring()


def stop_performance_monitoring() -> None:
    """Stop global performance monitoring."""
    global _performance_monitor
    if _performance_monitor:
        _performance_monitor.stop_monitoring()


def record_task_metrics(task_type: str, completion_time: float, success: bool = True) -> None:
    """Record task completion metrics."""
    monitor = get_performance_monitor()
    monitor.record_task_completion(task_type, completion_time, success)


# Performance optimization decorators
def monitor_performance(task_type: str = "default"):
    """Decorator to monitor function performance."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            success = True
            
            try:
                result = func(*args, **kwargs)
                return result
            except Exception as e:
                success = False
                raise
            finally:
                completion_time = time.time() - start_time
                record_task_metrics(task_type, completion_time, success)
                
        return wrapper
    return decorator
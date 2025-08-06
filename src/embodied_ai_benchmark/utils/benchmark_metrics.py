"""Comprehensive benchmark metrics collection and analysis system."""

import time
import threading
from typing import Dict, List, Any, Optional, Union, Callable
from dataclasses import dataclass, field
from collections import defaultdict, deque
from datetime import datetime, timedelta
import numpy as np
import json
from contextlib import contextmanager
from pathlib import Path
import statistics

from .logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class MetricPoint:
    """Individual metric measurement point."""
    name: str
    value: Union[float, int, str]
    timestamp: datetime = field(default_factory=datetime.now)
    tags: Dict[str, str] = field(default_factory=dict)
    unit: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "name": self.name,
            "value": self.value,
            "timestamp": self.timestamp.isoformat(),
            "tags": self.tags,
            "unit": self.unit
        }


@dataclass
class TimingContext:
    """Context for timing measurements."""
    name: str
    start_time: float
    tags: Dict[str, str] = field(default_factory=dict)


class BenchmarkMetricsCollector:
    """Thread-safe metrics collection system for benchmark performance analysis."""
    
    def __init__(self, max_history: int = 10000):
        """Initialize metrics collector.
        
        Args:
            max_history: Maximum number of metric points to keep in memory
        """
        self.max_history = max_history
        self.metrics_history = deque(maxlen=max_history)
        self.counters = defaultdict(int)
        self.gauges = defaultdict(float)
        self.timers = defaultdict(list)
        self.histograms = defaultdict(list)
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Active timing contexts
        self._active_timers = {}
        
        # Aggregation windows
        self._windowed_metrics = {
            "1min": deque(maxlen=60),   # 1 minute window
            "5min": deque(maxlen=300),  # 5 minute window  
            "15min": deque(maxlen=900), # 15 minute window
            "1hour": deque(maxlen=3600) # 1 hour window
        }
        
        # Metric metadata
        self._metric_metadata = {}
        
        # Start background processing
        self._should_stop = threading.Event()
        self._processor_thread = threading.Thread(
            target=self._process_metrics_loop, 
            daemon=True
        )
        self._processor_thread.start()
    
    def increment_counter(self, name: str, value: int = 1, tags: Optional[Dict[str, str]] = None):
        """Increment a counter metric.
        
        Args:
            name: Counter name
            value: Value to increment by
            tags: Additional tags for the metric
        """
        with self._lock:
            self.counters[name] += value
            
            metric_point = MetricPoint(
                name=name,
                value=value,
                tags=tags or {},
                unit="count"
            )
            
            self.metrics_history.append(metric_point)
            self._update_windowed_metrics(metric_point)
        
        logger.debug(f"Counter {name} incremented by {value}")
    
    def set_gauge(self, name: str, value: Union[float, int], tags: Optional[Dict[str, str]] = None, unit: str = ""):
        """Set a gauge metric value.
        
        Args:
            name: Gauge name
            value: Gauge value
            tags: Additional tags for the metric
            unit: Unit of measurement
        """
        with self._lock:
            self.gauges[name] = float(value)
            
            metric_point = MetricPoint(
                name=name,
                value=value,
                tags=tags or {},
                unit=unit
            )
            
            self.metrics_history.append(metric_point)
            self._update_windowed_metrics(metric_point)
        
        logger.debug(f"Gauge {name} set to {value} {unit}")
    
    def record_timing(self, name: str, duration: float, tags: Optional[Dict[str, str]] = None):
        """Record a timing measurement.
        
        Args:
            name: Timer name
            duration: Duration in seconds
            tags: Additional tags for the metric
        """
        with self._lock:
            self.timers[name].append(duration)
            
            # Keep only recent timings
            if len(self.timers[name]) > 1000:
                self.timers[name] = self.timers[name][-500:]
            
            metric_point = MetricPoint(
                name=name,
                value=duration,
                tags=tags or {},
                unit="seconds"
            )
            
            self.metrics_history.append(metric_point)
            self._update_windowed_metrics(metric_point)
        
        logger.debug(f"Timing {name} recorded: {duration:.3f}s")
    
    def record_histogram(self, name: str, value: Union[float, int], tags: Optional[Dict[str, str]] = None, unit: str = ""):
        """Record a value for histogram analysis.
        
        Args:
            name: Histogram name
            value: Value to record
            tags: Additional tags for the metric
            unit: Unit of measurement
        """
        with self._lock:
            self.histograms[name].append(float(value))
            
            # Keep only recent values
            if len(self.histograms[name]) > 1000:
                self.histograms[name] = self.histograms[name][-500:]
            
            metric_point = MetricPoint(
                name=name,
                value=value,
                tags=tags or {},
                unit=unit
            )
            
            self.metrics_history.append(metric_point)
            self._update_windowed_metrics(metric_point)
        
        logger.debug(f"Histogram {name} recorded value: {value}")
    
    @contextmanager
    def timer(self, name: str, tags: Optional[Dict[str, str]] = None):
        """Context manager for timing operations.
        
        Args:
            name: Timer name
            tags: Additional tags for the metric
        """
        start_time = time.time()
        try:
            yield
        finally:
            duration = time.time() - start_time
            self.record_timing(name, duration, tags)
    
    def start_timer(self, name: str, tags: Optional[Dict[str, str]] = None) -> str:
        """Start a named timer.
        
        Args:
            name: Timer name
            tags: Additional tags for the metric
            
        Returns:
            Timer ID for stopping the timer
        """
        timer_id = f"{name}_{int(time.time() * 1000000)}"
        context = TimingContext(name=name, start_time=time.time(), tags=tags or {})
        
        with self._lock:
            self._active_timers[timer_id] = context
        
        return timer_id
    
    def stop_timer(self, timer_id: str):
        """Stop a named timer and record the duration.
        
        Args:
            timer_id: Timer ID returned by start_timer
        """
        with self._lock:
            if timer_id in self._active_timers:
                context = self._active_timers.pop(timer_id)
                duration = time.time() - context.start_time
                self.record_timing(context.name, duration, context.tags)
            else:
                logger.warning(f"Timer ID {timer_id} not found")
    
    def _update_windowed_metrics(self, metric_point: MetricPoint):
        """Update windowed metric collections."""
        current_time = time.time()
        
        for window_name, window_deque in self._windowed_metrics.items():
            window_deque.append((current_time, metric_point))
    
    def _process_metrics_loop(self):
        """Background thread for processing and cleaning up metrics."""
        while not self._should_stop.wait(60):  # Process every minute
            try:
                current_time = time.time()
                
                # Clean up old windowed metrics
                for window_name, window_deque in self._windowed_metrics.items():
                    if window_name == "1min":
                        cutoff = current_time - 60
                    elif window_name == "5min":
                        cutoff = current_time - 300
                    elif window_name == "15min":
                        cutoff = current_time - 900
                    elif window_name == "1hour":
                        cutoff = current_time - 3600
                    else:
                        continue
                    
                    # Remove old entries
                    while window_deque and window_deque[0][0] < cutoff:
                        window_deque.popleft()
                
                # Clean up old active timers (over 1 hour old)
                with self._lock:
                    expired_timers = [
                        timer_id for timer_id, context in self._active_timers.items()
                        if current_time - context.start_time > 3600
                    ]
                    for timer_id in expired_timers:
                        logger.warning(f"Cleaning up expired timer: {timer_id}")
                        del self._active_timers[timer_id]
                
            except Exception as e:
                logger.error(f"Error in metrics processing loop: {e}")
    
    def get_counter(self, name: str) -> int:
        """Get current counter value.
        
        Args:
            name: Counter name
            
        Returns:
            Current counter value
        """
        with self._lock:
            return self.counters.get(name, 0)
    
    def get_gauge(self, name: str) -> float:
        """Get current gauge value.
        
        Args:
            name: Gauge name
            
        Returns:
            Current gauge value
        """
        with self._lock:
            return self.gauges.get(name, 0.0)
    
    def get_timer_stats(self, name: str) -> Dict[str, float]:
        """Get statistical summary of timer values.
        
        Args:
            name: Timer name
            
        Returns:
            Dictionary with timing statistics
        """
        with self._lock:
            timings = self.timers.get(name, [])
            
            if not timings:
                return {
                    "count": 0,
                    "mean": 0.0,
                    "min": 0.0,
                    "max": 0.0,
                    "p50": 0.0,
                    "p95": 0.0,
                    "p99": 0.0
                }
            
            return {
                "count": len(timings),
                "mean": statistics.mean(timings),
                "min": min(timings),
                "max": max(timings),
                "p50": np.percentile(timings, 50),
                "p95": np.percentile(timings, 95),
                "p99": np.percentile(timings, 99)
            }
    
    def get_histogram_stats(self, name: str) -> Dict[str, float]:
        """Get statistical summary of histogram values.
        
        Args:
            name: Histogram name
            
        Returns:
            Dictionary with histogram statistics
        """
        with self._lock:
            values = self.histograms.get(name, [])
            
            if not values:
                return {
                    "count": 0,
                    "mean": 0.0,
                    "std": 0.0,
                    "min": 0.0,
                    "max": 0.0,
                    "p25": 0.0,
                    "p50": 0.0,
                    "p75": 0.0,
                    "p95": 0.0,
                    "p99": 0.0
                }
            
            return {
                "count": len(values),
                "mean": statistics.mean(values),
                "std": statistics.stdev(values) if len(values) > 1 else 0.0,
                "min": min(values),
                "max": max(values),
                "p25": np.percentile(values, 25),
                "p50": np.percentile(values, 50),
                "p75": np.percentile(values, 75),
                "p95": np.percentile(values, 95),
                "p99": np.percentile(values, 99)
            }
    
    def get_windowed_stats(self, window: str = "5min") -> Dict[str, Any]:
        """Get statistics for a specific time window.
        
        Args:
            window: Window name (1min, 5min, 15min, 1hour)
            
        Returns:
            Statistics for the time window
        """
        if window not in self._windowed_metrics:
            raise ValueError(f"Unknown window: {window}. Available: {list(self._windowed_metrics.keys())}")
        
        with self._lock:
            window_data = self._windowed_metrics[window]
            
            if not window_data:
                return {"metrics_count": 0, "time_span": 0}
            
            # Group metrics by name
            metric_groups = defaultdict(list)
            for timestamp, metric_point in window_data:
                metric_groups[metric_point.name].append((timestamp, metric_point.value))
            
            stats = {
                "metrics_count": len(window_data),
                "time_span": window_data[-1][0] - window_data[0][0] if len(window_data) > 1 else 0,
                "metric_types": {}
            }
            
            for metric_name, values_with_time in metric_groups.items():
                values = [v for _, v in values_with_time]
                
                if all(isinstance(v, (int, float)) for v in values):
                    # Numeric metric
                    stats["metric_types"][metric_name] = {
                        "count": len(values),
                        "mean": statistics.mean(values),
                        "min": min(values),
                        "max": max(values),
                        "latest": values[-1]
                    }
                else:
                    # Non-numeric metric
                    stats["metric_types"][metric_name] = {
                        "count": len(values),
                        "latest": values[-1]
                    }
            
            return stats
    
    def get_all_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics for all metrics.
        
        Returns:
            Complete metrics statistics
        """
        with self._lock:
            stats = {
                "counters": dict(self.counters),
                "gauges": dict(self.gauges),
                "timers": {},
                "histograms": {},
                "active_timers": len(self._active_timers),
                "total_metrics_collected": len(self.metrics_history)
            }
            
            # Timer statistics
            for name in self.timers.keys():
                stats["timers"][name] = self.get_timer_stats(name)
            
            # Histogram statistics
            for name in self.histograms.keys():
                stats["histograms"][name] = self.get_histogram_stats(name)
            
            return stats
    
    def reset_metrics(self, metric_names: Optional[List[str]] = None):
        """Reset specific metrics or all metrics.
        
        Args:
            metric_names: List of metric names to reset, or None to reset all
        """
        with self._lock:
            if metric_names is None:
                # Reset everything
                self.counters.clear()
                self.gauges.clear()
                self.timers.clear()
                self.histograms.clear()
                self.metrics_history.clear()
                for window_deque in self._windowed_metrics.values():
                    window_deque.clear()
                logger.info("All metrics reset")
            else:
                # Reset specific metrics
                for name in metric_names:
                    self.counters.pop(name, None)
                    self.gauges.pop(name, None)
                    self.timers.pop(name, None)
                    self.histograms.pop(name, None)
                
                # Remove from history
                self.metrics_history = deque(
                    [mp for mp in self.metrics_history if mp.name not in metric_names],
                    maxlen=self.max_history
                )
                
                # Remove from windowed metrics
                for window_deque in self._windowed_metrics.values():
                    filtered = [(t, mp) for t, mp in window_deque if mp.name not in metric_names]
                    window_deque.clear()
                    window_deque.extend(filtered)
                
                logger.info(f"Reset metrics: {metric_names}")
    
    def export_metrics(self, filepath: str, format: str = "json"):
        """Export metrics to file.
        
        Args:
            filepath: Output file path
            format: Export format (json, csv)
        """
        path = Path(filepath)
        
        with self._lock:
            if format.lower() == "json":
                export_data = {
                    "export_timestamp": datetime.now().isoformat(),
                    "statistics": self.get_all_stats(),
                    "raw_metrics": [mp.to_dict() for mp in self.metrics_history]
                }
                
                with open(path, 'w') as f:
                    json.dump(export_data, f, indent=2, default=str)
            
            elif format.lower() == "csv":
                import csv
                
                with open(path, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(["timestamp", "name", "value", "unit", "tags"])
                    
                    for mp in self.metrics_history:
                        writer.writerow([
                            mp.timestamp.isoformat(),
                            mp.name,
                            mp.value,
                            mp.unit,
                            json.dumps(mp.tags)
                        ])
            
            else:
                raise ValueError(f"Unsupported export format: {format}")
        
        logger.info(f"Metrics exported to {filepath} in {format} format")
    
    def shutdown(self):
        """Shutdown the metrics collector."""
        self._should_stop.set()
        if self._processor_thread.is_alive():
            self._processor_thread.join(timeout=5.0)
        logger.info("Metrics collector shutdown complete")


# Global metrics collector instance
benchmark_metrics = BenchmarkMetricsCollector()


def get_metrics_collector() -> BenchmarkMetricsCollector:
    """Get the global metrics collector instance."""
    return benchmark_metrics


# Convenience functions for common metrics
def increment_counter(name: str, value: int = 1, tags: Optional[Dict[str, str]] = None):
    """Increment a counter metric."""
    benchmark_metrics.increment_counter(name, value, tags)


def set_gauge(name: str, value: Union[float, int], tags: Optional[Dict[str, str]] = None, unit: str = ""):
    """Set a gauge metric value."""
    benchmark_metrics.set_gauge(name, value, tags, unit)


def record_timing(name: str, duration: float, tags: Optional[Dict[str, str]] = None):
    """Record a timing measurement."""
    benchmark_metrics.record_timing(name, duration, tags)


def record_histogram(name: str, value: Union[float, int], tags: Optional[Dict[str, str]] = None, unit: str = ""):
    """Record a histogram value."""
    benchmark_metrics.record_histogram(name, value, tags, unit)


@contextmanager
def timer(name: str, tags: Optional[Dict[str, str]] = None):
    """Context manager for timing operations."""
    with benchmark_metrics.timer(name, tags):
        yield


def performance_monitor(metric_name: str = None, tags: Optional[Dict[str, str]] = None):
    """Decorator for monitoring function performance.
    
    Args:
        metric_name: Custom metric name (defaults to function name)
        tags: Additional tags for the metric
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            name = metric_name or f"{func.__module__}.{func.__name__}"
            
            with timer(name, tags):
                increment_counter(f"{name}.calls")
                try:
                    result = func(*args, **kwargs)
                    increment_counter(f"{name}.success")
                    return result
                except Exception as e:
                    increment_counter(f"{name}.errors")
                    raise
        
        return wrapper
    return decorator
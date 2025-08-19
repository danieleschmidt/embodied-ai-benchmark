"""Comprehensive Monitoring System for Research Components."""

import numpy as np
import torch
import time
import json
import os
import threading
import queue
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, field, asdict
from pathlib import Path
from datetime import datetime, timedelta
from collections import defaultdict, deque
import logging
import traceback
from concurrent.futures import ThreadPoolExecutor
import sqlite3
import pickle

from ..utils.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class MetricConfig:
    """Configuration for metric collection."""
    name: str
    collection_interval: float = 1.0  # seconds
    retention_period: int = 86400  # seconds (24 hours)
    aggregation_window: int = 60  # seconds
    alert_threshold: Optional[float] = None
    alert_condition: str = "greater_than"  # greater_than, less_than, equal_to
    enabled: bool = True


@dataclass
class Alert:
    """Alert configuration and state."""
    metric_name: str
    threshold: float
    condition: str
    message: str
    timestamp: float
    severity: str = "warning"  # info, warning, error, critical
    resolved: bool = False
    resolution_timestamp: Optional[float] = None


@dataclass
class MetricData:
    """Single metric data point."""
    timestamp: float
    value: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    component_name: str = ""
    experiment_id: str = ""


class MetricCollector:
    """Collects and stores metrics from research components."""
    
    def __init__(self, storage_path: str = "metrics_storage"):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(exist_ok=True)
        
        # In-memory storage with time-based retention
        self.metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=10000))
        self.metric_configs: Dict[str, MetricConfig] = {}
        
        # Database for persistent storage
        self.db_path = self.storage_path / "metrics.db"
        self._init_database()
        
        # Collection thread management
        self.collection_thread = None
        self.stop_collection = threading.Event()
        self.metric_queue = queue.Queue()
        
        # Registered collectors
        self.collectors: Dict[str, Callable] = {}
        
    def _init_database(self):
        """Initialize SQLite database for persistent metric storage."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp REAL,
                    metric_name TEXT,
                    value REAL,
                    component_name TEXT,
                    experiment_id TEXT,
                    metadata TEXT
                )
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_timestamp_metric 
                ON metrics (timestamp, metric_name)
            """)
    
    def register_metric(self, config: MetricConfig):
        """Register a metric for collection."""
        self.metric_configs[config.name] = config
        logger.info(f"Registered metric: {config.name}")
    
    def register_collector(self, metric_name: str, collector_func: Callable):
        """Register a function to collect a specific metric."""
        self.collectors[metric_name] = collector_func
        logger.info(f"Registered collector for metric: {metric_name}")
    
    def start_collection(self):
        """Start metric collection in background thread."""
        if self.collection_thread and self.collection_thread.is_alive():
            logger.warning("Metric collection already running")
            return
        
        self.stop_collection.clear()
        self.collection_thread = threading.Thread(target=self._collection_loop)
        self.collection_thread.daemon = True
        self.collection_thread.start()
        
        logger.info("Started metric collection")
    
    def stop_collection_process(self):
        """Stop metric collection."""
        self.stop_collection.set()
        if self.collection_thread:
            self.collection_thread.join(timeout=5.0)
        
        logger.info("Stopped metric collection")
    
    def _collection_loop(self):
        """Main collection loop running in background thread."""
        while not self.stop_collection.is_set():
            try:
                # Collect metrics from registered collectors
                for metric_name, collector_func in self.collectors.items():
                    if metric_name in self.metric_configs and self.metric_configs[metric_name].enabled:
                        try:
                            value = collector_func()
                            if value is not None:
                                self.record_metric(metric_name, value)
                        except Exception as e:
                            logger.error(f"Failed to collect metric {metric_name}: {e}")
                
                # Process queued metrics
                while not self.metric_queue.empty():
                    try:
                        metric_data = self.metric_queue.get_nowait()
                        self._store_metric(metric_data)
                    except queue.Empty:
                        break
                    except Exception as e:
                        logger.error(f"Failed to process queued metric: {e}")
                
                # Wait before next collection cycle
                time.sleep(1.0)
                
            except Exception as e:
                logger.error(f"Error in collection loop: {e}")
                time.sleep(5.0)  # Wait longer on error
    
    def record_metric(self, name: str, value: float, 
                     component_name: str = "", 
                     experiment_id: str = "",
                     metadata: Optional[Dict[str, Any]] = None):
        """Record a metric value."""
        if metadata is None:
            metadata = {}
        
        metric_data = MetricData(
            timestamp=time.time(),
            value=value,
            metadata=metadata,
            component_name=component_name,
            experiment_id=experiment_id
        )
        
        # Add to queue for background processing
        try:
            self.metric_queue.put_nowait(metric_data)
        except queue.Full:
            logger.warning(f"Metric queue full, dropping metric: {name}")
    
    def _store_metric(self, metric_data: MetricData):
        """Store metric data in memory and database."""
        # Store in memory
        self.metrics[metric_data.component_name].append(metric_data)
        
        # Store in database
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT INTO metrics 
                    (timestamp, metric_name, value, component_name, experiment_id, metadata)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    metric_data.timestamp,
                    metric_data.component_name,
                    metric_data.value,
                    metric_data.component_name,
                    metric_data.experiment_id,
                    json.dumps(metric_data.metadata)
                ))
        except Exception as e:
            logger.error(f"Failed to store metric in database: {e}")
    
    def get_metrics(self, metric_name: str, 
                   start_time: Optional[float] = None,
                   end_time: Optional[float] = None) -> List[MetricData]:
        """Retrieve metrics from storage."""
        if end_time is None:
            end_time = time.time()
        if start_time is None:
            start_time = end_time - 3600  # Last hour by default
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    SELECT timestamp, metric_name, value, component_name, experiment_id, metadata
                    FROM metrics
                    WHERE metric_name = ? AND timestamp BETWEEN ? AND ?
                    ORDER BY timestamp
                """, (metric_name, start_time, end_time))
                
                metrics = []
                for row in cursor:
                    timestamp, name, value, component_name, experiment_id, metadata_json = row
                    metadata = json.loads(metadata_json) if metadata_json else {}
                    
                    metrics.append(MetricData(
                        timestamp=timestamp,
                        value=value,
                        metadata=metadata,
                        component_name=component_name,
                        experiment_id=experiment_id
                    ))
                
                return metrics
                
        except Exception as e:
            logger.error(f"Failed to retrieve metrics: {e}")
            return []
    
    def get_aggregated_metrics(self, metric_name: str,
                              aggregation: str = "mean",
                              window_size: int = 60,
                              start_time: Optional[float] = None,
                              end_time: Optional[float] = None) -> List[Tuple[float, float]]:
        """Get aggregated metrics over time windows."""
        metrics = self.get_metrics(metric_name, start_time, end_time)
        
        if not metrics:
            return []
        
        # Group metrics by time windows
        aggregated = []
        current_window_start = metrics[0].timestamp
        current_window_values = []
        
        for metric in metrics:
            if metric.timestamp >= current_window_start + window_size:
                # Process current window
                if current_window_values:
                    if aggregation == "mean":
                        agg_value = np.mean(current_window_values)
                    elif aggregation == "median":
                        agg_value = np.median(current_window_values)
                    elif aggregation == "max":
                        agg_value = np.max(current_window_values)
                    elif aggregation == "min":
                        agg_value = np.min(current_window_values)
                    elif aggregation == "std":
                        agg_value = np.std(current_window_values)
                    else:
                        agg_value = np.mean(current_window_values)
                    
                    aggregated.append((current_window_start, agg_value))
                
                # Start new window
                current_window_start = metric.timestamp
                current_window_values = [metric.value]
            else:
                current_window_values.append(metric.value)
        
        # Process final window
        if current_window_values:
            if aggregation == "mean":
                agg_value = np.mean(current_window_values)
            elif aggregation == "median":
                agg_value = np.median(current_window_values)
            elif aggregation == "max":
                agg_value = np.max(current_window_values)
            elif aggregation == "min":
                agg_value = np.min(current_window_values)
            elif aggregation == "std":
                agg_value = np.std(current_window_values)
            else:
                agg_value = np.mean(current_window_values)
            
            aggregated.append((current_window_start, agg_value))
        
        return aggregated


class AlertManager:
    """Manages alerts based on metric thresholds."""
    
    def __init__(self):
        self.alerts: List[Alert] = []
        self.alert_configs: Dict[str, MetricConfig] = {}
        self.alert_handlers: List[Callable[[Alert], None]] = []
        
    def register_alert(self, metric_config: MetricConfig):
        """Register an alert for a metric."""
        if metric_config.alert_threshold is not None:
            self.alert_configs[metric_config.name] = metric_config
            logger.info(f"Registered alert for metric: {metric_config.name}")
    
    def add_alert_handler(self, handler: Callable[[Alert], None]):
        """Add a handler for alert notifications."""
        self.alert_handlers.append(handler)
    
    def check_alerts(self, metric_data: MetricData):
        """Check if metric triggers any alerts."""
        metric_name = metric_data.component_name
        
        if metric_name not in self.alert_configs:
            return
        
        config = self.alert_configs[metric_name]
        
        if config.alert_threshold is None:
            return
        
        # Check alert condition
        triggered = False
        if config.alert_condition == "greater_than":
            triggered = metric_data.value > config.alert_threshold
        elif config.alert_condition == "less_than":
            triggered = metric_data.value < config.alert_threshold
        elif config.alert_condition == "equal_to":
            triggered = abs(metric_data.value - config.alert_threshold) < 1e-6
        
        if triggered:
            # Check if this alert was already triggered recently
            recent_alerts = [
                a for a in self.alerts 
                if a.metric_name == metric_name and 
                not a.resolved and 
                time.time() - a.timestamp < 300  # 5 minutes
            ]
            
            if not recent_alerts:
                alert = Alert(
                    metric_name=metric_name,
                    threshold=config.alert_threshold,
                    condition=config.alert_condition,
                    message=f"Metric {metric_name} {config.alert_condition} {config.alert_threshold} (current: {metric_data.value:.3f})",
                    timestamp=time.time(),
                    severity="warning"
                )
                
                self.alerts.append(alert)
                
                # Notify handlers
                for handler in self.alert_handlers:
                    try:
                        handler(alert)
                    except Exception as e:
                        logger.error(f"Alert handler failed: {e}")
    
    def resolve_alert(self, alert_id: str):
        """Mark an alert as resolved."""
        for alert in self.alerts:
            if id(alert) == alert_id:
                alert.resolved = True
                alert.resolution_timestamp = time.time()
                break
    
    def get_active_alerts(self) -> List[Alert]:
        """Get all active (unresolved) alerts."""
        return [alert for alert in self.alerts if not alert.resolved]


class PerformanceMonitor:
    """Monitor performance metrics of research components."""
    
    def __init__(self, metric_collector: MetricCollector):
        self.metric_collector = metric_collector
        self.component_stats: Dict[str, Dict[str, Any]] = defaultdict(dict)
        
        # Register performance metrics
        self._register_performance_metrics()
    
    def _register_performance_metrics(self):
        """Register standard performance metrics."""
        metrics = [
            MetricConfig("cpu_usage", collection_interval=1.0),
            MetricConfig("memory_usage", collection_interval=1.0, alert_threshold=85.0),
            MetricConfig("gpu_memory_usage", collection_interval=1.0, alert_threshold=90.0),
            MetricConfig("execution_time", collection_interval=0.1),
            MetricConfig("throughput", collection_interval=1.0),
            MetricConfig("error_rate", collection_interval=5.0, alert_threshold=0.1),
            MetricConfig("queue_size", collection_interval=1.0, alert_threshold=1000.0)
        ]
        
        for metric in metrics:
            self.metric_collector.register_metric(metric)
        
        # Register collectors
        self.metric_collector.register_collector("cpu_usage", self._collect_cpu_usage)
        self.metric_collector.register_collector("memory_usage", self._collect_memory_usage)
        self.metric_collector.register_collector("gpu_memory_usage", self._collect_gpu_memory_usage)
    
    def _collect_cpu_usage(self) -> Optional[float]:
        """Collect CPU usage percentage."""
        try:
            import psutil
            return psutil.cpu_percent(interval=None)
        except ImportError:
            return None
    
    def _collect_memory_usage(self) -> Optional[float]:
        """Collect memory usage percentage."""
        try:
            import psutil
            return psutil.virtual_memory().percent
        except ImportError:
            return None
    
    def _collect_gpu_memory_usage(self) -> Optional[float]:
        """Collect GPU memory usage percentage."""
        if not torch.cuda.is_available():
            return None
        
        try:
            allocated = torch.cuda.memory_allocated()
            total = torch.cuda.get_device_properties(0).total_memory
            return (allocated / total) * 100
        except Exception:
            return None
    
    def track_component_execution(self, component_name: str):
        """Context manager to track component execution."""
        return ComponentExecutionTracker(component_name, self.metric_collector)
    
    def get_component_summary(self, component_name: str, 
                            time_window: int = 3600) -> Dict[str, Any]:
        """Get performance summary for a component."""
        end_time = time.time()
        start_time = end_time - time_window
        
        summary = {
            "component_name": component_name,
            "time_window_hours": time_window / 3600,
            "metrics": {}
        }
        
        # Get metrics for this component
        for metric_name in ["execution_time", "throughput", "error_rate"]:
            metrics = self.metric_collector.get_metrics(
                f"{component_name}_{metric_name}", start_time, end_time
            )
            
            if metrics:
                values = [m.value for m in metrics]
                summary["metrics"][metric_name] = {
                    "count": len(values),
                    "mean": np.mean(values),
                    "median": np.median(values),
                    "std": np.std(values),
                    "min": np.min(values),
                    "max": np.max(values),
                    "p95": np.percentile(values, 95),
                    "p99": np.percentile(values, 99)
                }
        
        return summary


class ComponentExecutionTracker:
    """Context manager for tracking component execution metrics."""
    
    def __init__(self, component_name: str, metric_collector: MetricCollector):
        self.component_name = component_name
        self.metric_collector = metric_collector
        self.start_time = None
        self.start_memory = None
        self.error_occurred = False
    
    def __enter__(self):
        self.start_time = time.time()
        
        # Record starting memory
        if torch.cuda.is_available():
            self.start_memory = torch.cuda.memory_allocated()
        
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        execution_time = time.time() - self.start_time
        
        # Record execution time
        self.metric_collector.record_metric(
            f"{self.component_name}_execution_time",
            execution_time,
            component_name=self.component_name
        )
        
        # Record memory usage change
        if torch.cuda.is_available() and self.start_memory is not None:
            memory_used = torch.cuda.memory_allocated() - self.start_memory
            self.metric_collector.record_metric(
                f"{self.component_name}_memory_delta",
                memory_used / 1024**2,  # MB
                component_name=self.component_name
            )
        
        # Record error if occurred
        if exc_type is not None:
            self.metric_collector.record_metric(
                f"{self.component_name}_error_count",
                1.0,
                component_name=self.component_name,
                metadata={"error_type": str(exc_type.__name__)}
            )


class ExperimentTracker:
    """Track experiments and their metrics."""
    
    def __init__(self, metric_collector: MetricCollector):
        self.metric_collector = metric_collector
        self.experiments: Dict[str, Dict[str, Any]] = {}
        self.current_experiment_id = None
    
    def start_experiment(self, experiment_name: str, 
                        config: Optional[Dict[str, Any]] = None) -> str:
        """Start tracking a new experiment."""
        experiment_id = f"{experiment_name}_{int(time.time())}"
        
        self.experiments[experiment_id] = {
            "name": experiment_name,
            "start_time": time.time(),
            "config": config or {},
            "status": "running",
            "results": {},
            "metrics": []
        }
        
        self.current_experiment_id = experiment_id
        
        logger.info(f"Started experiment: {experiment_id}")
        
        return experiment_id
    
    def log_experiment_metric(self, metric_name: str, value: float,
                             step: Optional[int] = None,
                             metadata: Optional[Dict[str, Any]] = None):
        """Log a metric for the current experiment."""
        if self.current_experiment_id is None:
            logger.warning("No active experiment to log metrics to")
            return
        
        if metadata is None:
            metadata = {}
        
        if step is not None:
            metadata["step"] = step
        
        # Record in metric collector
        self.metric_collector.record_metric(
            metric_name,
            value,
            component_name=f"experiment_{self.current_experiment_id}",
            experiment_id=self.current_experiment_id,
            metadata=metadata
        )
        
        # Store in experiment data
        self.experiments[self.current_experiment_id]["metrics"].append({
            "metric_name": metric_name,
            "value": value,
            "timestamp": time.time(),
            "step": step,
            "metadata": metadata
        })
    
    def finish_experiment(self, results: Optional[Dict[str, Any]] = None):
        """Finish the current experiment."""
        if self.current_experiment_id is None:
            logger.warning("No active experiment to finish")
            return
        
        experiment = self.experiments[self.current_experiment_id]
        experiment["end_time"] = time.time()
        experiment["duration"] = experiment["end_time"] - experiment["start_time"]
        experiment["status"] = "completed"
        
        if results:
            experiment["results"] = results
        
        logger.info(f"Finished experiment: {self.current_experiment_id}")
        
        # Save experiment data
        self._save_experiment(self.current_experiment_id)
        
        self.current_experiment_id = None
    
    def _save_experiment(self, experiment_id: str):
        """Save experiment data to file."""
        experiment_data = self.experiments[experiment_id]
        
        output_dir = Path("experiments")
        output_dir.mkdir(exist_ok=True)
        
        output_file = output_dir / f"{experiment_id}.json"
        
        try:
            with open(output_file, 'w') as f:
                json.dump(experiment_data, f, indent=2, default=str)
            
            logger.info(f"Experiment data saved: {output_file}")
            
        except Exception as e:
            logger.error(f"Failed to save experiment data: {e}")
    
    def get_experiment_summary(self, experiment_id: str) -> Optional[Dict[str, Any]]:
        """Get summary of an experiment."""
        if experiment_id not in self.experiments:
            return None
        
        experiment = self.experiments[experiment_id]
        
        # Aggregate metrics
        metric_summary = {}
        for metric_entry in experiment["metrics"]:
            metric_name = metric_entry["metric_name"]
            value = metric_entry["value"]
            
            if metric_name not in metric_summary:
                metric_summary[metric_name] = {
                    "values": [],
                    "count": 0,
                    "mean": 0,
                    "std": 0,
                    "min": float('inf'),
                    "max": float('-inf')
                }
            
            metric_summary[metric_name]["values"].append(value)
            metric_summary[metric_name]["count"] += 1
            metric_summary[metric_name]["min"] = min(metric_summary[metric_name]["min"], value)
            metric_summary[metric_name]["max"] = max(metric_summary[metric_name]["max"], value)
        
        # Compute statistics
        for metric_name, stats in metric_summary.items():
            values = stats["values"]
            stats["mean"] = np.mean(values)
            stats["std"] = np.std(values)
            stats["median"] = np.median(values)
            del stats["values"]  # Remove raw values from summary
        
        return {
            "experiment_id": experiment_id,
            "name": experiment["name"],
            "status": experiment["status"],
            "duration": experiment.get("duration", 0),
            "config": experiment["config"],
            "results": experiment["results"],
            "metric_summary": metric_summary
        }


class ComprehensiveMonitor:
    """Main monitoring system that orchestrates all monitoring components."""
    
    def __init__(self, storage_path: str = "monitoring_data"):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(exist_ok=True)
        
        # Initialize components
        self.metric_collector = MetricCollector(str(self.storage_path / "metrics"))
        self.alert_manager = AlertManager()
        self.performance_monitor = PerformanceMonitor(self.metric_collector)
        self.experiment_tracker = ExperimentTracker(self.metric_collector)
        
        # Setup alert handlers
        self.alert_manager.add_alert_handler(self._log_alert)
        
        # Dashboard data
        self.dashboard_data = {}
        
    def start_monitoring(self):
        """Start all monitoring components."""
        self.metric_collector.start_collection()
        logger.info("Comprehensive monitoring started")
    
    def stop_monitoring(self):
        """Stop all monitoring components."""
        self.metric_collector.stop_collection_process()
        logger.info("Comprehensive monitoring stopped")
    
    def _log_alert(self, alert: Alert):
        """Default alert handler that logs alerts."""
        severity_level = {
            "info": logging.INFO,
            "warning": logging.WARNING,
            "error": logging.ERROR,
            "critical": logging.CRITICAL
        }.get(alert.severity, logging.WARNING)
        
        logger.log(severity_level, f"ALERT: {alert.message}")
    
    def track_research_component(self, component_name: str):
        """Get a context manager for tracking a research component."""
        return self.performance_monitor.track_component_execution(component_name)
    
    def start_experiment(self, experiment_name: str, config: Optional[Dict[str, Any]] = None) -> str:
        """Start tracking an experiment."""
        return self.experiment_tracker.start_experiment(experiment_name, config)
    
    def log_experiment_metric(self, metric_name: str, value: float, step: Optional[int] = None):
        """Log an experiment metric."""
        self.experiment_tracker.log_experiment_metric(metric_name, value, step)
    
    def finish_experiment(self, results: Optional[Dict[str, Any]] = None):
        """Finish the current experiment."""
        self.experiment_tracker.finish_experiment(results)
    
    def get_dashboard_data(self) -> Dict[str, Any]:
        """Get data for monitoring dashboard."""
        current_time = time.time()
        one_hour_ago = current_time - 3600
        
        dashboard_data = {
            "timestamp": current_time,
            "system_metrics": {
                "cpu_usage": self._get_latest_metric("cpu_usage"),
                "memory_usage": self._get_latest_metric("memory_usage"),
                "gpu_memory_usage": self._get_latest_metric("gpu_memory_usage")
            },
            "active_alerts": len(self.alert_manager.get_active_alerts()),
            "experiments": {
                "active": 1 if self.experiment_tracker.current_experiment_id else 0,
                "total": len(self.experiment_tracker.experiments)
            },
            "component_stats": {}
        }
        
        # Get component performance data
        for component_name in ["dynamic_attention_fusion", "quantum_enhanced_planning", "emergent_swarm_coordination"]:
            summary = self.performance_monitor.get_component_summary(component_name)
            if summary["metrics"]:
                dashboard_data["component_stats"][component_name] = summary
        
        return dashboard_data
    
    def _get_latest_metric(self, metric_name: str) -> Optional[float]:
        """Get the latest value for a metric."""
        metrics = self.metric_collector.get_metrics(
            metric_name, 
            time.time() - 60,  # Last minute
            time.time()
        )
        
        if metrics:
            return metrics[-1].value
        return None
    
    def generate_monitoring_report(self, hours: int = 24) -> str:
        """Generate a comprehensive monitoring report."""
        end_time = time.time()
        start_time = end_time - (hours * 3600)
        
        # Get system metrics
        cpu_metrics = self.metric_collector.get_aggregated_metrics("cpu_usage", "mean", 300, start_time, end_time)
        memory_metrics = self.metric_collector.get_aggregated_metrics("memory_usage", "mean", 300, start_time, end_time)
        
        # Get alerts
        recent_alerts = [a for a in self.alert_manager.alerts if a.timestamp >= start_time]
        active_alerts = self.alert_manager.get_active_alerts()
        
        report = f"""
# Monitoring Report ({hours}h)

## System Performance
- **Monitoring Period**: {datetime.fromtimestamp(start_time)} to {datetime.fromtimestamp(end_time)}
- **Average CPU Usage**: {np.mean([m[1] for m in cpu_metrics]):.1f}% (last {len(cpu_metrics)} samples)
- **Average Memory Usage**: {np.mean([m[1] for m in memory_metrics]):.1f}% (last {len(memory_metrics)} samples)

## Alerts
- **Total Alerts**: {len(recent_alerts)}
- **Active Alerts**: {len(active_alerts)}
- **Resolved Alerts**: {len(recent_alerts) - len(active_alerts)}

### Active Alerts
"""
        
        for alert in active_alerts[:5]:  # Show top 5 active alerts
            report += f"- **{alert.severity.upper()}**: {alert.message} (since {datetime.fromtimestamp(alert.timestamp)})\n"
        
        # Component performance
        report += f"""
## Component Performance
"""
        
        component_names = ["dynamic_attention_fusion", "quantum_enhanced_planning", "emergent_swarm_coordination"]
        for component_name in component_names:
            summary = self.performance_monitor.get_component_summary(component_name, hours * 3600)
            
            if summary["metrics"]:
                execution_metrics = summary["metrics"].get("execution_time", {})
                if execution_metrics:
                    report += f"""
### {component_name}
- **Executions**: {execution_metrics.get('count', 0)}
- **Average Execution Time**: {execution_metrics.get('mean', 0):.3f}s
- **P95 Execution Time**: {execution_metrics.get('p95', 0):.3f}s
- **Error Rate**: {summary["metrics"].get("error_rate", {}).get("mean", 0):.2%}
"""
        
        # Experiments
        report += f"""
## Experiments
- **Total Experiments**: {len(self.experiment_tracker.experiments)}
- **Active Experiments**: {1 if self.experiment_tracker.current_experiment_id else 0}
"""
        
        # Recent experiments
        recent_experiments = [
            exp for exp in self.experiment_tracker.experiments.values()
            if exp["start_time"] >= start_time
        ]
        
        if recent_experiments:
            report += f"- **Recent Experiments**: {len(recent_experiments)}\n"
            
            for exp in recent_experiments[-3:]:  # Show last 3
                status = exp["status"]
                duration = exp.get("duration", time.time() - exp["start_time"])
                report += f"  - {exp['name']}: {status} ({duration:.1f}s)\n"
        
        return report
    
    def save_monitoring_state(self):
        """Save current monitoring state to file."""
        state = {
            "timestamp": time.time(),
            "metric_configs": {name: asdict(config) for name, config in self.metric_collector.metric_configs.items()},
            "alert_configs": {name: asdict(config) for name, config in self.alert_manager.alert_configs.items()},
            "experiments": self.experiment_tracker.experiments,
            "dashboard_data": self.get_dashboard_data()
        }
        
        state_file = self.storage_path / "monitoring_state.json"
        
        try:
            with open(state_file, 'w') as f:
                json.dump(state, f, indent=2, default=str)
            
            logger.info(f"Monitoring state saved to {state_file}")
            
        except Exception as e:
            logger.error(f"Failed to save monitoring state: {e}")


def create_comprehensive_monitor(storage_path: str = "monitoring_data") -> ComprehensiveMonitor:
    """Factory function to create comprehensive monitoring system."""
    monitor = ComprehensiveMonitor(storage_path)
    
    logger.info(f"Created Comprehensive Monitor with storage at: {storage_path}")
    
    return monitor
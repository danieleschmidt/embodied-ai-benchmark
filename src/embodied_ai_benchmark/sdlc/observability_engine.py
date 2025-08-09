"""
Terragon Observability Engine v2.0

Advanced monitoring, telemetry, and observability system for autonomous SDLC execution.
Implements comprehensive metrics collection, distributed tracing, and intelligent alerting.
"""

import asyncio
import json
import time
import logging
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum
import threading
from collections import defaultdict, deque
import statistics
from pathlib import Path


class MetricType(Enum):
    """Types of metrics collected"""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    TIMER = "timer"
    RATE = "rate"


class AlertSeverity(Enum):
    """Alert severity levels"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class Metric:
    """Individual metric data point"""
    name: str
    value: float
    metric_type: MetricType
    timestamp: datetime
    tags: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TraceSpan:
    """Distributed tracing span"""
    span_id: str
    trace_id: str
    parent_span_id: Optional[str]
    operation_name: str
    start_time: datetime
    end_time: Optional[datetime] = None
    duration_ms: Optional[float] = None
    tags: Dict[str, str] = field(default_factory=dict)
    logs: List[Dict[str, Any]] = field(default_factory=list)
    status: str = "ok"  # ok, error, timeout
    
    def finish(self):
        """Mark span as finished"""
        self.end_time = datetime.now()
        if self.start_time:
            self.duration_ms = (self.end_time - self.start_time).total_seconds() * 1000


@dataclass
class Alert:
    """System alert"""
    alert_id: str
    timestamp: datetime
    severity: AlertSeverity
    title: str
    description: str
    source: str
    tags: Dict[str, str] = field(default_factory=dict)
    resolved: bool = False
    resolution_time: Optional[datetime] = None


class MetricsCollector:
    """
    High-performance metrics collector with in-memory storage and aggregation.
    """
    
    def __init__(self):
        self.metrics: deque = deque(maxlen=10000)  # Rolling window of metrics
        self.counters: Dict[str, float] = defaultdict(float)
        self.gauges: Dict[str, float] = defaultdict(float)
        self.histograms: Dict[str, List[float]] = defaultdict(list)
        self.rates: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        self.lock = threading.Lock()
        
    def increment(self, name: str, value: float = 1.0, tags: Dict[str, str] = None):
        """Increment a counter metric"""
        with self.lock:
            self.counters[name] += value
            self._store_metric(name, value, MetricType.COUNTER, tags or {})
    
    def gauge(self, name: str, value: float, tags: Dict[str, str] = None):
        """Set a gauge metric"""
        with self.lock:
            self.gauges[name] = value
            self._store_metric(name, value, MetricType.GAUGE, tags or {})
    
    def histogram(self, name: str, value: float, tags: Dict[str, str] = None):
        """Record a histogram value"""
        with self.lock:
            self.histograms[name].append(value)
            # Keep only recent values
            if len(self.histograms[name]) > 1000:
                self.histograms[name] = self.histograms[name][-500:]
            self._store_metric(name, value, MetricType.HISTOGRAM, tags or {})
    
    def rate(self, name: str, value: float = 1.0, tags: Dict[str, str] = None):
        """Record a rate metric"""
        with self.lock:
            self.rates[name].append((time.time(), value))
            self._store_metric(name, value, MetricType.RATE, tags or {})
    
    def timer(self, name: str, tags: Dict[str, str] = None):
        """Create a timer context manager"""
        return TimerContext(self, name, tags or {})
    
    def _store_metric(self, name: str, value: float, metric_type: MetricType, tags: Dict[str, str]):
        """Store metric in rolling window"""
        metric = Metric(
            name=name,
            value=value,
            metric_type=metric_type,
            timestamp=datetime.now(),
            tags=tags
        )
        self.metrics.append(metric)
    
    def get_metric_summary(self, name: str, window_minutes: int = 5) -> Dict[str, Any]:
        """Get summary statistics for a metric"""
        cutoff = datetime.now() - timedelta(minutes=window_minutes)
        
        relevant_metrics = [
            m for m in self.metrics 
            if m.name == name and m.timestamp >= cutoff
        ]
        
        if not relevant_metrics:
            return {"count": 0}
        
        values = [m.value for m in relevant_metrics]
        
        return {
            "count": len(values),
            "min": min(values),
            "max": max(values),
            "mean": statistics.mean(values),
            "median": statistics.median(values),
            "p95": self._percentile(values, 95),
            "p99": self._percentile(values, 99)
        }
    
    def _percentile(self, values: List[float], percentile: int) -> float:
        """Calculate percentile of values"""
        if not values:
            return 0.0
        sorted_values = sorted(values)
        index = int((percentile / 100) * len(sorted_values))
        return sorted_values[min(index, len(sorted_values) - 1)]
    
    def get_all_metrics(self) -> Dict[str, Any]:
        """Get all current metric values"""
        with self.lock:
            current_rates = {}
            for name, rate_data in self.rates.items():
                if rate_data:
                    # Calculate rate per minute
                    now = time.time()
                    recent_data = [(t, v) for t, v in rate_data if now - t <= 60]
                    if recent_data:
                        total_value = sum(v for t, v in recent_data)
                        current_rates[name] = total_value
            
            histogram_summaries = {}
            for name, values in self.histograms.items():
                if values:
                    histogram_summaries[name] = {
                        "count": len(values),
                        "mean": statistics.mean(values),
                        "p95": self._percentile(values, 95)
                    }
            
            return {
                "counters": dict(self.counters),
                "gauges": dict(self.gauges),
                "rates": current_rates,
                "histograms": histogram_summaries,
                "timestamp": datetime.now().isoformat()
            }


class TimerContext:
    """Context manager for timing operations"""
    
    def __init__(self, collector: MetricsCollector, name: str, tags: Dict[str, str]):
        self.collector = collector
        self.name = name
        self.tags = tags
        self.start_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.start_time:
            duration = (time.time() - self.start_time) * 1000  # Convert to milliseconds
            self.collector.histogram(f"{self.name}.duration_ms", duration, self.tags)


class TracingEngine:
    """
    Distributed tracing engine for tracking operation flows.
    """
    
    def __init__(self):
        self.spans: Dict[str, TraceSpan] = {}
        self.traces: Dict[str, List[str]] = defaultdict(list)  # trace_id -> span_ids
        self.lock = threading.Lock()
    
    def start_span(self, operation_name: str, parent_span_id: str = None, trace_id: str = None) -> str:
        """Start a new tracing span"""
        import uuid
        
        span_id = str(uuid.uuid4())[:8]
        
        if not trace_id:
            if parent_span_id and parent_span_id in self.spans:
                trace_id = self.spans[parent_span_id].trace_id
            else:
                trace_id = str(uuid.uuid4())[:8]
        
        span = TraceSpan(
            span_id=span_id,
            trace_id=trace_id,
            parent_span_id=parent_span_id,
            operation_name=operation_name,
            start_time=datetime.now()
        )
        
        with self.lock:
            self.spans[span_id] = span
            self.traces[trace_id].append(span_id)
        
        return span_id
    
    def finish_span(self, span_id: str, status: str = "ok", tags: Dict[str, str] = None):
        """Finish a tracing span"""
        with self.lock:
            if span_id in self.spans:
                span = self.spans[span_id]
                span.finish()
                span.status = status
                if tags:
                    span.tags.update(tags)
    
    def add_span_log(self, span_id: str, level: str, message: str, metadata: Dict[str, Any] = None):
        """Add log entry to span"""
        with self.lock:
            if span_id in self.spans:
                span = self.spans[span_id]
                span.logs.append({
                    "timestamp": datetime.now().isoformat(),
                    "level": level,
                    "message": message,
                    "metadata": metadata or {}
                })
    
    def get_trace(self, trace_id: str) -> List[TraceSpan]:
        """Get all spans for a trace"""
        with self.lock:
            span_ids = self.traces.get(trace_id, [])
            return [self.spans[span_id] for span_id in span_ids if span_id in self.spans]
    
    def get_trace_summary(self, trace_id: str) -> Dict[str, Any]:
        """Get summary of a trace"""
        spans = self.get_trace(trace_id)
        
        if not spans:
            return {}
        
        total_duration = max((s.duration_ms or 0) for s in spans)
        root_span = next((s for s in spans if s.parent_span_id is None), spans[0])
        
        return {
            "trace_id": trace_id,
            "root_operation": root_span.operation_name,
            "total_spans": len(spans),
            "total_duration_ms": total_duration,
            "start_time": min(s.start_time for s in spans).isoformat(),
            "status": "error" if any(s.status == "error" for s in spans) else "ok",
            "spans": [asdict(span) for span in spans]
        }


class AlertManager:
    """
    Intelligent alert management with deduplication and escalation.
    """
    
    def __init__(self):
        self.alerts: List[Alert] = []
        self.alert_rules: List[Dict[str, Any]] = []
        self.notification_channels: List[Callable] = []
        self.lock = threading.Lock()
        
        # Initialize default alert rules
        self._initialize_default_rules()
    
    def _initialize_default_rules(self):
        """Initialize default alerting rules"""
        self.alert_rules = [
            {
                "name": "High Error Rate",
                "condition": lambda metrics: metrics.get("counters", {}).get("errors", 0) > 10,
                "severity": AlertSeverity.ERROR,
                "description": "Error rate is above threshold"
            },
            {
                "name": "High Response Time",
                "condition": lambda metrics: self._check_response_time(metrics),
                "severity": AlertSeverity.WARNING,
                "description": "Response times are elevated"
            },
            {
                "name": "System Health Critical",
                "condition": lambda metrics: metrics.get("gauges", {}).get("system_health", 1.0) < 0.3,
                "severity": AlertSeverity.CRITICAL,
                "description": "System health is critical"
            }
        ]
    
    def _check_response_time(self, metrics: Dict[str, Any]) -> bool:
        """Check if response times are elevated"""
        histograms = metrics.get("histograms", {})
        for name, stats in histograms.items():
            if "response_time" in name and stats.get("p95", 0) > 5000:  # 5 seconds
                return True
        return False
    
    def add_alert_rule(self, name: str, condition: Callable, severity: AlertSeverity, description: str):
        """Add custom alert rule"""
        rule = {
            "name": name,
            "condition": condition,
            "severity": severity,
            "description": description
        }
        self.alert_rules.append(rule)
    
    def add_notification_channel(self, channel: Callable):
        """Add notification channel (e.g., email, Slack, webhook)"""
        self.notification_channels.append(channel)
    
    def check_alerts(self, metrics: Dict[str, Any]) -> List[Alert]:
        """Check all alert rules against current metrics"""
        new_alerts = []
        
        for rule in self.alert_rules:
            try:
                if rule["condition"](metrics):
                    alert = self._create_alert(
                        title=rule["name"],
                        description=rule["description"],
                        severity=rule["severity"],
                        source="alert_manager"
                    )
                    new_alerts.append(alert)
            except Exception as e:
                # Log error but don't fail alert checking
                pass
        
        return new_alerts
    
    def _create_alert(self, title: str, description: str, severity: AlertSeverity, source: str, tags: Dict[str, str] = None) -> Alert:
        """Create new alert"""
        import uuid
        
        alert = Alert(
            alert_id=str(uuid.uuid4())[:8],
            timestamp=datetime.now(),
            severity=severity,
            title=title,
            description=description,
            source=source,
            tags=tags or {}
        )
        
        with self.lock:
            # Check for duplicate alerts (simple deduplication)
            similar_alert = next(
                (a for a in self.alerts 
                 if a.title == title and not a.resolved 
                 and (datetime.now() - a.timestamp).seconds < 300),  # 5 minutes
                None
            )
            
            if not similar_alert:
                self.alerts.append(alert)
                # Send notifications
                self._send_notifications(alert)
            
            return alert
    
    def _send_notifications(self, alert: Alert):
        """Send alert notifications through configured channels"""
        for channel in self.notification_channels:
            try:
                channel(alert)
            except Exception as e:
                # Log notification failure but continue
                pass
    
    def resolve_alert(self, alert_id: str):
        """Mark alert as resolved"""
        with self.lock:
            for alert in self.alerts:
                if alert.alert_id == alert_id:
                    alert.resolved = True
                    alert.resolution_time = datetime.now()
                    break
    
    def get_active_alerts(self) -> List[Alert]:
        """Get all unresolved alerts"""
        with self.lock:
            return [alert for alert in self.alerts if not alert.resolved]
    
    def get_alert_summary(self) -> Dict[str, Any]:
        """Get alert summary statistics"""
        with self.lock:
            active_alerts = [a for a in self.alerts if not a.resolved]
            recent_alerts = [a for a in self.alerts if 
                           (datetime.now() - a.timestamp).hours < 24]
            
            severity_counts = {}
            for alert in active_alerts:
                severity = alert.severity.value
                severity_counts[severity] = severity_counts.get(severity, 0) + 1
            
            return {
                "total_alerts": len(self.alerts),
                "active_alerts": len(active_alerts),
                "recent_alerts": len(recent_alerts),
                "severity_breakdown": severity_counts,
                "avg_resolution_time_minutes": self._calculate_avg_resolution_time()
            }
    
    def _calculate_avg_resolution_time(self) -> float:
        """Calculate average alert resolution time"""
        resolved_alerts = [a for a in self.alerts if a.resolved and a.resolution_time]
        
        if not resolved_alerts:
            return 0.0
        
        total_time = sum(
            (a.resolution_time - a.timestamp).total_seconds() 
            for a in resolved_alerts
        )
        
        return total_time / len(resolved_alerts) / 60  # Convert to minutes


class ObservabilityEngine:
    """
    Main observability engine that coordinates metrics, tracing, and alerting.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.metrics_collector = MetricsCollector()
        self.tracing_engine = TracingEngine()
        self.alert_manager = AlertManager()
        self.monitoring_active = False
        
        # Performance tracking
        self.start_time = datetime.now()
        self.system_stats = {
            "uptime_seconds": 0,
            "requests_processed": 0,
            "errors_encountered": 0,
            "avg_response_time_ms": 0.0
        }
        
        self._setup_default_monitoring()
    
    def _setup_default_monitoring(self):
        """Setup default monitoring configuration"""
        # Add default notification channel (logging)
        def log_notification(alert: Alert):
            self.logger.warning(
                f"🚨 ALERT [{alert.severity.value.upper()}] {alert.title}: {alert.description}"
            )
        
        self.alert_manager.add_notification_channel(log_notification)
    
    async def start_monitoring(self):
        """Start continuous monitoring loop"""
        self.monitoring_active = True
        self.logger.info("🔍 Starting observability monitoring")
        
        while self.monitoring_active:
            try:
                await self._monitoring_cycle()
                await asyncio.sleep(10)  # Monitor every 10 seconds
            except Exception as e:
                self.logger.error(f"Monitoring cycle error: {e}")
                await asyncio.sleep(5)  # Brief pause on error
    
    async def _monitoring_cycle(self):
        """Execute one monitoring cycle"""
        # Update system stats
        self._update_system_stats()
        
        # Collect current metrics
        current_metrics = self.metrics_collector.get_all_metrics()
        
        # Check for alerts
        new_alerts = self.alert_manager.check_alerts(current_metrics)
        
        # Log monitoring status
        if new_alerts:
            self.logger.info(f"Generated {len(new_alerts)} new alerts")
    
    def _update_system_stats(self):
        """Update system performance statistics"""
        uptime = (datetime.now() - self.start_time).total_seconds()
        
        self.system_stats["uptime_seconds"] = uptime
        self.metrics_collector.gauge("system.uptime_seconds", uptime)
        
        # Update other gauges
        for stat_name, value in self.system_stats.items():
            self.metrics_collector.gauge(f"system.{stat_name}", value)
    
    def stop_monitoring(self):
        """Stop monitoring loop"""
        self.monitoring_active = False
        self.logger.info("Observability monitoring stopped")
    
    def record_operation(self, operation_name: str, duration_ms: float, success: bool = True, tags: Dict[str, str] = None):
        """Record completion of an operation"""
        tags = tags or {}
        
        # Record metrics
        self.metrics_collector.histogram(f"{operation_name}.duration_ms", duration_ms, tags)
        self.metrics_collector.increment(f"{operation_name}.count", 1.0, tags)
        
        if success:
            self.metrics_collector.increment(f"{operation_name}.success", 1.0, tags)
        else:
            self.metrics_collector.increment(f"{operation_name}.error", 1.0, tags)
            self.system_stats["errors_encountered"] += 1
        
        self.system_stats["requests_processed"] += 1
    
    def create_span(self, operation_name: str, parent_span_id: str = None) -> str:
        """Create a new tracing span"""
        return self.tracing_engine.start_span(operation_name, parent_span_id)
    
    def finish_span(self, span_id: str, success: bool = True, tags: Dict[str, str] = None):
        """Finish a tracing span"""
        status = "ok" if success else "error"
        self.tracing_engine.finish_span(span_id, status, tags)
    
    def span_context(self, operation_name: str, parent_span_id: str = None):
        """Create a span context manager"""
        return SpanContext(self, operation_name, parent_span_id)
    
    def get_system_health(self) -> Dict[str, Any]:
        """Get overall system health metrics"""
        active_alerts = self.alert_manager.get_active_alerts()
        critical_alerts = [a for a in active_alerts if a.severity == AlertSeverity.CRITICAL]
        
        # Calculate health score
        health_score = 1.0
        if critical_alerts:
            health_score = 0.2
        elif len(active_alerts) > 5:
            health_score = 0.5
        elif len(active_alerts) > 0:
            health_score = 0.7
        
        return {
            "health_score": health_score,
            "status": self._determine_health_status(health_score),
            "uptime_seconds": self.system_stats["uptime_seconds"],
            "active_alerts": len(active_alerts),
            "critical_alerts": len(critical_alerts),
            "error_rate": self._calculate_error_rate(),
            "avg_response_time_ms": self._calculate_avg_response_time(),
            "last_updated": datetime.now().isoformat()
        }
    
    def _determine_health_status(self, health_score: float) -> str:
        """Determine health status from score"""
        if health_score >= 0.8:
            return "healthy"
        elif health_score >= 0.6:
            return "degraded"
        elif health_score >= 0.3:
            return "unhealthy"
        else:
            return "critical"
    
    def _calculate_error_rate(self) -> float:
        """Calculate current error rate"""
        total_requests = self.system_stats["requests_processed"]
        total_errors = self.system_stats["errors_encountered"]
        
        if total_requests == 0:
            return 0.0
        
        return (total_errors / total_requests) * 100
    
    def _calculate_avg_response_time(self) -> float:
        """Calculate average response time across all operations"""
        # Get all histogram metrics for duration
        current_metrics = self.metrics_collector.get_all_metrics()
        histograms = current_metrics.get("histograms", {})
        
        duration_metrics = [
            stats["mean"] for name, stats in histograms.items() 
            if "duration_ms" in name and "mean" in stats
        ]
        
        if not duration_metrics:
            return 0.0
        
        return statistics.mean(duration_metrics)
    
    def get_observability_dashboard(self) -> Dict[str, Any]:
        """Generate comprehensive observability dashboard data"""
        return {
            "system_health": self.get_system_health(),
            "metrics": self.metrics_collector.get_all_metrics(),
            "alerts": self.alert_manager.get_alert_summary(),
            "recent_traces": self._get_recent_traces(),
            "performance_trends": self._get_performance_trends(),
            "recommendations": self._get_observability_recommendations()
        }
    
    def _get_recent_traces(self) -> List[Dict[str, Any]]:
        """Get recent trace summaries"""
        # Get unique trace IDs from recent spans
        recent_trace_ids = set()
        cutoff = datetime.now() - timedelta(minutes=10)
        
        for span in self.tracing_engine.spans.values():
            if span.start_time >= cutoff:
                recent_trace_ids.add(span.trace_id)
        
        return [
            self.tracing_engine.get_trace_summary(trace_id)
            for trace_id in list(recent_trace_ids)[:10]  # Last 10 traces
        ]
    
    def _get_performance_trends(self) -> Dict[str, Any]:
        """Get performance trend analysis"""
        return {
            "request_rate_trend": "stable",  # Would calculate from historical data
            "error_rate_trend": "decreasing",
            "response_time_trend": "stable",
            "resource_usage_trend": "increasing"
        }
    
    def _get_observability_recommendations(self) -> List[str]:
        """Get observability improvement recommendations"""
        recommendations = []
        
        health = self.get_system_health()
        
        if health["error_rate"] > 5:
            recommendations.append("Investigate high error rate - check logs for patterns")
        
        if health["avg_response_time_ms"] > 1000:
            recommendations.append("Optimize response times - consider caching or performance tuning")
        
        if health["active_alerts"] > 10:
            recommendations.append("Review alert thresholds to reduce noise")
        
        if not recommendations:
            recommendations.append("System is performing well - continue monitoring")
        
        return recommendations


class SpanContext:
    """Context manager for distributed tracing spans"""
    
    def __init__(self, engine: ObservabilityEngine, operation_name: str, parent_span_id: str = None):
        self.engine = engine
        self.operation_name = operation_name
        self.parent_span_id = parent_span_id
        self.span_id = None
        self.start_time = None
        self.success = True
        self.tags = {}
    
    def __enter__(self):
        self.span_id = self.engine.create_span(self.operation_name, self.parent_span_id)
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            self.success = False
            self.tags["error"] = str(exc_val)
        
        if self.span_id:
            duration_ms = (time.time() - self.start_time) * 1000
            self.engine.record_operation(self.operation_name, duration_ms, self.success, self.tags)
            self.engine.finish_span(self.span_id, self.success, self.tags)
    
    def add_tag(self, key: str, value: str):
        """Add tag to span"""
        self.tags[key] = value
    
    def log(self, message: str, level: str = "info", metadata: Dict[str, Any] = None):
        """Add log to span"""
        if self.span_id:
            self.engine.tracing_engine.add_span_log(self.span_id, level, message, metadata)


# Integration function
def integrate_observability(orchestrator_class):
    """Integrate observability engine with autonomous orchestrator"""
    
    def __init_with_observability__(self, *args, **kwargs):
        # Call original init
        self.__class__.__bases__[0].__init__(self, *args, **kwargs)
        
        # Add observability engine
        self.observability_engine = ObservabilityEngine()
        
        # Override phase execution to add tracing
        self._wrap_phase_methods()
        
        # Start monitoring
        asyncio.create_task(self.observability_engine.start_monitoring())
    
    def _wrap_phase_methods(self):
        """Wrap phase methods with observability"""
        phase_methods = [name for name in dir(self) if name.startswith('_phase_')]
        
        for method_name in phase_methods:
            original_method = getattr(self, method_name)
            
            async def wrapped_method(*args, original=original_method, name=method_name, **kwargs):
                with self.observability_engine.span_context(name):
                    return await original(*args, **kwargs)
            
            setattr(self, method_name, wrapped_method)
    
    # Monkey patch the orchestrator class
    orchestrator_class.__init__ = __init_with_observability__
    orchestrator_class._wrap_phase_methods = _wrap_phase_methods
    
    return orchestrator_class
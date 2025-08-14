"""
Autonomous Auto-Scaling Engine for Dynamic Resource Management
"""

import time
import threading
import asyncio
from typing import Dict, List, Optional, Any, Callable, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import logging
import json
import queue
from collections import deque, defaultdict
import statistics

from .performance_monitor import get_performance_monitor, PerformanceMetrics

logger = logging.getLogger(__name__)


class ScalingAction(Enum):
    """Types of scaling actions."""
    SCALE_UP = "scale_up"
    SCALE_DOWN = "scale_down"
    SCALE_OUT = "scale_out"  # Horizontal scaling
    SCALE_IN = "scale_in"   # Horizontal scaling
    OPTIMIZE = "optimize"


@dataclass
class ScalingRule:
    """Defines a scaling rule with conditions and actions."""
    name: str
    metric: str
    threshold: float
    comparison: str  # 'gt', 'lt', 'gte', 'lte'
    action: ScalingAction
    cooldown_seconds: float = 300.0  # 5 minutes
    min_datapoints: int = 3
    last_triggered: Optional[datetime] = None
    enabled: bool = True


@dataclass
class ResourceLimits:
    """Resource limits for scaling operations."""
    min_workers: int = 1
    max_workers: int = 16
    min_memory_mb: int = 512
    max_memory_mb: int = 8192
    min_cpu_cores: float = 0.5
    max_cpu_cores: float = 8.0


class AutonomousAutoScaler:
    """Intelligent auto-scaling system with predictive capabilities."""
    
    def __init__(self, 
                 resource_limits: Optional[ResourceLimits] = None,
                 prediction_window: int = 300,  # 5 minutes
                 scaling_aggressiveness: float = 0.7):
        """Initialize the auto-scaler.
        
        Args:
            resource_limits: Limits for resource scaling
            prediction_window: Seconds to look ahead for predictions
            scaling_aggressiveness: How aggressively to scale (0.0-1.0)
        """
        self.resource_limits = resource_limits or ResourceLimits()
        self.prediction_window = prediction_window
        self.scaling_aggressiveness = scaling_aggressiveness
        
        # Current resource allocation
        self.current_workers = 2
        self.current_memory_mb = 1024
        self.current_cpu_cores = 2.0
        
        # Scaling rules
        self.scaling_rules: List[ScalingRule] = []
        self._setup_default_scaling_rules()
        
        # Prediction and learning
        self.workload_history: deque = deque(maxlen=3600)  # 1 hour
        self.scaling_history: List[Dict] = []
        self.demand_predictions: deque = deque(maxlen=60)  # Future predictions
        
        # Control
        self.is_running = False
        self.scaling_thread: Optional[threading.Thread] = None
        self.scaling_callbacks: List[Callable] = []
        
        # Statistics
        self.scaling_decisions = defaultdict(int)
        self.performance_impact = defaultdict(list)
        
    def _setup_default_scaling_rules(self) -> None:
        """Setup default scaling rules."""
        self.scaling_rules = [
            # CPU-based scaling
            ScalingRule(
                name="high_cpu_scale_out",
                metric="cpu_percent",
                threshold=80.0,
                comparison="gte",
                action=ScalingAction.SCALE_OUT,
                cooldown_seconds=180.0
            ),
            ScalingRule(
                name="low_cpu_scale_in",
                metric="cpu_percent",
                threshold=30.0,
                comparison="lte",
                action=ScalingAction.SCALE_IN,
                cooldown_seconds=300.0
            ),
            
            # Memory-based scaling
            ScalingRule(
                name="high_memory_scale_up",
                metric="memory_percent",
                threshold=85.0,
                comparison="gte",
                action=ScalingAction.SCALE_UP,
                cooldown_seconds=120.0
            ),
            ScalingRule(
                name="low_memory_scale_down",
                metric="memory_percent",
                threshold=40.0,
                comparison="lte",
                action=ScalingAction.SCALE_DOWN,
                cooldown_seconds=600.0
            ),
            
            # Throughput-based scaling
            ScalingRule(
                name="low_throughput_optimize",
                metric="task_throughput",
                threshold=5.0,
                comparison="lte",
                action=ScalingAction.OPTIMIZE,
                cooldown_seconds=240.0
            )
        ]
        
    def start_auto_scaling(self) -> None:
        """Start the auto-scaling engine."""
        if self.is_running:
            logger.warning("Auto-scaler already running")
            return
            
        self.is_running = True
        self.scaling_thread = threading.Thread(target=self._scaling_loop, daemon=True)
        self.scaling_thread.start()
        logger.info("Auto-scaling engine started")
        
    def stop_auto_scaling(self) -> None:
        """Stop the auto-scaling engine."""
        self.is_running = False
        if self.scaling_thread:
            self.scaling_thread.join()
        logger.info("Auto-scaling engine stopped")
        
    def _scaling_loop(self) -> None:
        """Main scaling decision loop."""
        while self.is_running:
            try:
                # Collect current metrics
                performance_monitor = get_performance_monitor()
                if performance_monitor.metrics_history:
                    current_metrics = performance_monitor.metrics_history[-1]
                    self.workload_history.append(current_metrics)
                    
                    # Make scaling decisions
                    self._evaluate_scaling_rules(current_metrics)
                    
                    # Update predictions
                    self._update_demand_predictions()
                    
                    # Learn from past decisions
                    self._update_learning_models()
                    
                time.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                logger.error(f"Error in scaling loop: {e}")
                time.sleep(30)
                
    def _evaluate_scaling_rules(self, metrics: PerformanceMetrics) -> None:
        """Evaluate all scaling rules against current metrics."""
        current_time = datetime.now()
        
        for rule in self.scaling_rules:
            if not rule.enabled:
                continue
                
            # Check cooldown
            if (rule.last_triggered and 
                (current_time - rule.last_triggered).total_seconds() < rule.cooldown_seconds):
                continue
                
            # Get metric value
            metric_value = getattr(metrics, rule.metric, None)
            if metric_value is None:
                continue
                
            # Check if we have enough historical data
            recent_metrics = list(self.workload_history)[-rule.min_datapoints:]
            if len(recent_metrics) < rule.min_datapoints:
                continue
                
            # Calculate average over recent datapoints
            recent_values = [getattr(m, rule.metric, 0) for m in recent_metrics]
            avg_value = statistics.mean(recent_values) if recent_values else 0
            
            # Evaluate condition
            condition_met = False
            if rule.comparison == "gt":
                condition_met = avg_value > rule.threshold
            elif rule.comparison == "gte":
                condition_met = avg_value >= rule.threshold
            elif rule.comparison == "lt":
                condition_met = avg_value < rule.threshold
            elif rule.comparison == "lte":
                condition_met = avg_value <= rule.threshold
                
            if condition_met:
                self._execute_scaling_action(rule, avg_value)
                rule.last_triggered = current_time
                
    def _execute_scaling_action(self, rule: ScalingRule, metric_value: float) -> None:
        """Execute a scaling action."""
        logger.info(f"Executing scaling action: {rule.name} (value: {metric_value:.2f})")
        
        scaling_factor = self._calculate_scaling_factor(rule, metric_value)
        action_taken = False
        
        if rule.action == ScalingAction.SCALE_OUT:
            # Increase number of workers
            new_workers = min(
                self.resource_limits.max_workers,
                int(self.current_workers * (1 + scaling_factor))
            )
            if new_workers > self.current_workers:
                self.current_workers = new_workers
                action_taken = True
                
        elif rule.action == ScalingAction.SCALE_IN:
            # Decrease number of workers
            new_workers = max(
                self.resource_limits.min_workers,
                int(self.current_workers * (1 - scaling_factor))
            )
            if new_workers < self.current_workers:
                self.current_workers = new_workers
                action_taken = True
                
        elif rule.action == ScalingAction.SCALE_UP:
            # Increase memory allocation
            new_memory = min(
                self.resource_limits.max_memory_mb,
                int(self.current_memory_mb * (1 + scaling_factor))
            )
            if new_memory > self.current_memory_mb:
                self.current_memory_mb = new_memory
                action_taken = True
                
        elif rule.action == ScalingAction.SCALE_DOWN:
            # Decrease memory allocation
            new_memory = max(
                self.resource_limits.min_memory_mb,
                int(self.current_memory_mb * (1 - scaling_factor))
            )
            if new_memory < self.current_memory_mb:
                self.current_memory_mb = new_memory
                action_taken = True
                
        elif rule.action == ScalingAction.OPTIMIZE:
            # Trigger optimization callbacks
            action_taken = True
            
        if action_taken:
            # Record scaling decision
            scaling_event = {
                "timestamp": datetime.now().isoformat(),
                "rule_name": rule.name,
                "action": rule.action.value,
                "metric_value": metric_value,
                "scaling_factor": scaling_factor,
                "new_workers": self.current_workers,
                "new_memory_mb": self.current_memory_mb
            }
            self.scaling_history.append(scaling_event)
            self.scaling_decisions[rule.action.value] += 1
            
            # Execute callbacks
            for callback in self.scaling_callbacks:
                try:
                    callback(scaling_event)
                except Exception as e:
                    logger.error(f"Error in scaling callback: {e}")
                    
    def _calculate_scaling_factor(self, rule: ScalingRule, metric_value: float) -> float:
        """Calculate how much to scale based on metric deviation."""
        threshold = rule.threshold
        
        if rule.comparison in ["gt", "gte"]:
            # Scale based on how much we exceed the threshold
            excess = max(0, metric_value - threshold)
            scaling_factor = (excess / threshold) * self.scaling_aggressiveness
        else:
            # Scale based on how much we're below the threshold
            deficit = max(0, threshold - metric_value)
            scaling_factor = (deficit / threshold) * self.scaling_aggressiveness
            
        # Cap scaling factor to reasonable bounds
        return min(0.5, max(0.1, scaling_factor))
        
    def _update_demand_predictions(self) -> None:
        """Update demand predictions using historical patterns."""
        if len(self.workload_history) < 60:  # Need at least 1 minute of data
            return
            
        # Simple trend-based prediction
        recent_throughput = [
            m.task_throughput for m in list(self.workload_history)[-10:]
            if m.task_throughput is not None
        ]
        
        if len(recent_throughput) >= 5:
            # Calculate trend
            x = list(range(len(recent_throughput)))
            slope = self._calculate_slope(x, recent_throughput)
            
            # Predict future throughput
            current_throughput = recent_throughput[-1]
            predicted_throughput = current_throughput + (slope * 10)  # 10 time steps ahead
            
            self.demand_predictions.append({
                "timestamp": datetime.now() + timedelta(seconds=self.prediction_window),
                "predicted_throughput": max(0, predicted_throughput),
                "confidence": min(1.0, 1.0 - abs(slope) * 0.1)
            })
            
    def _calculate_slope(self, x: List[float], y: List[float]) -> float:
        """Calculate linear regression slope."""
        if len(x) != len(y) or len(x) < 2:
            return 0.0
            
        n = len(x)
        sum_x = sum(x)
        sum_y = sum(y)
        sum_xy = sum(x[i] * y[i] for i in range(n))
        sum_x2 = sum(x_i ** 2 for x_i in x)
        
        denominator = n * sum_x2 - sum_x ** 2
        if denominator == 0:
            return 0.0
            
        return (n * sum_xy - sum_x * sum_y) / denominator
        
    def _update_learning_models(self) -> None:
        """Update machine learning models for better predictions."""
        # Simplified learning: analyze effectiveness of past scaling decisions
        if len(self.scaling_history) < 10:
            return
            
        # Analyze recent scaling decisions and their impact
        recent_decisions = self.scaling_history[-10:]
        
        for decision in recent_decisions:
            # Find performance metrics after the scaling decision
            decision_time = datetime.fromisoformat(decision["timestamp"])
            
            # Look for metrics 2-5 minutes after the decision
            future_metrics = [
                m for m in self.workload_history
                if (m.timestamp - decision_time).total_seconds() > 120 and
                   (m.timestamp - decision_time).total_seconds() < 300
            ]
            
            if future_metrics:
                # Calculate performance impact
                avg_cpu_after = statistics.mean([m.cpu_percent for m in future_metrics])
                avg_throughput_after = statistics.mean([
                    m.task_throughput for m in future_metrics 
                    if m.task_throughput is not None
                ])
                
                impact_score = self._calculate_impact_score(
                    decision["action"], 
                    avg_cpu_after, 
                    avg_throughput_after
                )
                
                self.performance_impact[decision["action"]].append(impact_score)
                
    def _calculate_impact_score(self, action: str, cpu_after: float, throughput_after: float) -> float:
        """Calculate the effectiveness of a scaling action."""
        score = 0.0
        
        if action in ["scale_out", "scale_up"]:
            # For scaling up, we want lower CPU and higher throughput
            score += max(0, (80 - cpu_after) / 80)  # Normalized CPU improvement
            score += min(1, throughput_after / 20)   # Normalized throughput
        elif action in ["scale_in", "scale_down"]:
            # For scaling down, we want efficient resource usage
            if cpu_after < 70:  # If CPU is still reasonable after scaling down
                score += 0.7
            if throughput_after > 5:  # If throughput is still good
                score += 0.3
                
        return min(1.0, score)
        
    def add_scaling_callback(self, callback: Callable[[Dict], None]) -> None:
        """Add callback for scaling events."""
        self.scaling_callbacks.append(callback)
        
    def add_scaling_rule(self, rule: ScalingRule) -> None:
        """Add a custom scaling rule."""
        self.scaling_rules.append(rule)
        logger.info(f"Added scaling rule: {rule.name}")
        
    def get_current_resources(self) -> Dict[str, Any]:
        """Get current resource allocation."""
        return {
            "workers": self.current_workers,
            "memory_mb": self.current_memory_mb,
            "cpu_cores": self.current_cpu_cores,
            "timestamp": datetime.now().isoformat()
        }
        
    def get_scaling_statistics(self) -> Dict[str, Any]:
        """Get comprehensive scaling statistics."""
        total_decisions = sum(self.scaling_decisions.values())
        
        stats = {
            "total_scaling_decisions": total_decisions,
            "decisions_by_action": dict(self.scaling_decisions),
            "current_resources": self.get_current_resources(),
            "active_rules": len([r for r in self.scaling_rules if r.enabled]),
            "scaling_effectiveness": {},
            "recent_predictions": list(self.demand_predictions)[-5:]
        }
        
        # Calculate effectiveness for each action
        for action, impacts in self.performance_impact.items():
            if impacts:
                stats["scaling_effectiveness"][action] = {
                    "average_impact": statistics.mean(impacts),
                    "sample_size": len(impacts)
                }
                
        return stats
        
    def optimize_scaling_rules(self) -> None:
        """Automatically optimize scaling rules based on historical performance."""
        if len(self.performance_impact) < 3:
            logger.info("Insufficient data for rule optimization")
            return
            
        logger.info("Optimizing scaling rules based on performance history")
        
        # Adjust thresholds based on effectiveness
        for rule in self.scaling_rules:
            action_impacts = self.performance_impact.get(rule.action.value, [])
            
            if len(action_impacts) >= 5:
                avg_effectiveness = statistics.mean(action_impacts)
                
                if avg_effectiveness < 0.3:  # Poor effectiveness
                    # Make rule less aggressive
                    if rule.comparison in ["gte", "gt"]:
                        rule.threshold *= 1.1  # Increase threshold
                    else:
                        rule.threshold *= 0.9  # Decrease threshold
                        
                    rule.cooldown_seconds *= 1.2  # Increase cooldown
                    logger.info(f"Made rule {rule.name} less aggressive due to poor effectiveness")
                    
                elif avg_effectiveness > 0.8:  # High effectiveness
                    # Make rule more aggressive
                    if rule.comparison in ["gte", "gt"]:
                        rule.threshold *= 0.95  # Slightly decrease threshold
                    else:
                        rule.threshold *= 1.05  # Slightly increase threshold
                        
                    rule.cooldown_seconds *= 0.9  # Decrease cooldown
                    logger.info(f"Made rule {rule.name} more aggressive due to high effectiveness")


# Global auto-scaler instance
_auto_scaler: Optional[AutonomousAutoScaler] = None


def get_auto_scaler() -> AutonomousAutoScaler:
    """Get global auto-scaler instance."""
    global _auto_scaler
    if _auto_scaler is None:
        _auto_scaler = AutonomousAutoScaler()
    return _auto_scaler


def start_auto_scaling() -> None:
    """Start global auto-scaling."""
    scaler = get_auto_scaler()
    scaler.start_auto_scaling()


def stop_auto_scaling() -> None:
    """Stop global auto-scaling."""
    global _auto_scaler
    if _auto_scaler:
        _auto_scaler.stop_auto_scaling()
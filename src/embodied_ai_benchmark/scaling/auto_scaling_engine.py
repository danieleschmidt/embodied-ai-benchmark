"""Auto-scaling and load balancing engine for distributed deployment."""

import time
import json
import threading
import logging
from typing import Any, Dict, List, Optional, Callable, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timezone, timedelta
from collections import defaultdict, deque
import concurrent.futures
import queue
import hashlib


@dataclass
class WorkerNode:
    """Worker node configuration and status."""
    node_id: str
    endpoint: str
    capacity: int
    current_load: int
    last_heartbeat: datetime
    status: str  # "healthy", "degraded", "unhealthy"
    capabilities: List[str]
    performance_score: float
    region: str


@dataclass
class WorkloadMetrics:
    """Workload metrics for scaling decisions."""
    timestamp: datetime
    total_requests: int
    active_workers: int
    avg_response_time: float
    error_rate: float
    cpu_usage: float
    memory_usage: float
    queue_length: int


@dataclass
class ScalingEvent:
    """Scaling event record."""
    timestamp: datetime
    event_type: str  # "scale_up", "scale_down", "rebalance"
    trigger_reason: str
    nodes_affected: List[str]
    target_capacity: int
    success: bool


class LoadBalancer:
    """Intelligent load balancer with multiple strategies."""
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize load balancer.
        
        Args:
            config: Load balancer configuration
        """
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Worker nodes
        self.nodes = {}
        self.healthy_nodes = []
        
        # Load balancing strategy
        self.strategy = self.config.get("strategy", "round_robin")
        self.current_node_index = 0
        
        # Health monitoring
        self.health_check_interval = self.config.get("health_check_interval", 30)
        self.unhealthy_threshold = self.config.get("unhealthy_threshold", 3)
        
        # Request tracking
        self.request_counts = defaultdict(int)
        self.response_times = defaultdict(list)
        
        # Threading
        self._lock = threading.RLock()
        self._health_check_thread = None
        self._stop_monitoring = threading.Event()
        
        self._start_health_monitoring()
    
    def register_node(self, node: WorkerNode):
        """Register a new worker node.
        
        Args:
            node: Worker node to register
        """
        with self._lock:
            self.nodes[node.node_id] = node
            self._update_healthy_nodes()
            self.logger.info(f"Registered node {node.node_id} at {node.endpoint}")
    
    def unregister_node(self, node_id: str):
        """Unregister a worker node.
        
        Args:
            node_id: Node ID to unregister
        """
        with self._lock:
            if node_id in self.nodes:
                del self.nodes[node_id]
                self._update_healthy_nodes()
                self.logger.info(f"Unregistered node {node_id}")
    
    def select_node(self, 
                   request_data: Dict[str, Any] = None,
                   required_capabilities: List[str] = None) -> Optional[WorkerNode]:
        """Select optimal node for request.
        
        Args:
            request_data: Request data for context-aware routing
            required_capabilities: Required node capabilities
            
        Returns:
            Selected worker node or None if none available
        """
        with self._lock:
            # Filter nodes by capabilities
            eligible_nodes = self.healthy_nodes
            
            if required_capabilities:
                eligible_nodes = [
                    node for node in eligible_nodes
                    if all(cap in node.capabilities for cap in required_capabilities)
                ]
            
            if not eligible_nodes:
                return None
            
            # Apply load balancing strategy
            if self.strategy == "round_robin":
                return self._round_robin_select(eligible_nodes)
            elif self.strategy == "least_connections":
                return self._least_connections_select(eligible_nodes)
            elif self.strategy == "weighted_response_time":
                return self._weighted_response_time_select(eligible_nodes)
            elif self.strategy == "performance_based":
                return self._performance_based_select(eligible_nodes)
            elif self.strategy == "geographic":
                return self._geographic_select(eligible_nodes, request_data)
            else:
                return self._round_robin_select(eligible_nodes)
    
    def _round_robin_select(self, nodes: List[WorkerNode]) -> WorkerNode:
        """Round-robin node selection."""
        if not nodes:
            return None
        
        node = nodes[self.current_node_index % len(nodes)]
        self.current_node_index += 1
        return node
    
    def _least_connections_select(self, nodes: List[WorkerNode]) -> WorkerNode:
        """Select node with least connections."""
        return min(nodes, key=lambda n: n.current_load)
    
    def _weighted_response_time_select(self, nodes: List[WorkerNode]) -> WorkerNode:
        """Select node based on weighted response time."""
        # Calculate weights based on response time history
        weights = []
        for node in nodes:
            avg_response_time = self._get_avg_response_time(node.node_id)
            # Lower response time = higher weight
            weight = 1.0 / max(avg_response_time, 0.001)
            weights.append(weight)
        
        # Weighted random selection
        import random
        total_weight = sum(weights)
        if total_weight == 0:
            return nodes[0]
        
        target = random.uniform(0, total_weight)
        current = 0
        
        for i, weight in enumerate(weights):
            current += weight
            if current >= target:
                return nodes[i]
        
        return nodes[-1]  # Fallback
    
    def _performance_based_select(self, nodes: List[WorkerNode]) -> WorkerNode:
        """Select node based on performance score."""
        # Consider both performance score and current load
        def score_node(node):
            load_factor = 1.0 - (node.current_load / max(node.capacity, 1))
            return node.performance_score * load_factor
        
        return max(nodes, key=score_node)
    
    def _geographic_select(self, nodes: List[WorkerNode], request_data: Dict[str, Any]) -> WorkerNode:
        """Select node based on geographic proximity."""
        if not request_data or "client_region" not in request_data:
            return self._least_connections_select(nodes)
        
        client_region = request_data["client_region"]
        
        # Prefer nodes in same region
        same_region_nodes = [n for n in nodes if n.region == client_region]
        if same_region_nodes:
            return self._least_connections_select(same_region_nodes)
        
        # Fall back to least connections
        return self._least_connections_select(nodes)
    
    def record_request_completion(self, node_id: str, response_time: float, success: bool):
        """Record request completion metrics.
        
        Args:
            node_id: Node that handled the request
            response_time: Request response time
            success: Whether request was successful
        """
        with self._lock:
            self.request_counts[node_id] += 1
            self.response_times[node_id].append(response_time)
            
            # Limit history
            if len(self.response_times[node_id]) > 1000:
                self.response_times[node_id] = self.response_times[node_id][-500:]
            
            # Update node load (would be more sophisticated in practice)
            if node_id in self.nodes:
                if success:
                    self.nodes[node_id].current_load = max(0, self.nodes[node_id].current_load - 1)
                else:
                    # Penalty for failed requests
                    self.nodes[node_id].performance_score *= 0.95
    
    def _get_avg_response_time(self, node_id: str) -> float:
        """Get average response time for node."""
        times = self.response_times.get(node_id, [])
        if not times:
            return 1.0  # Default
        return sum(times) / len(times)
    
    def _update_healthy_nodes(self):
        """Update list of healthy nodes."""
        self.healthy_nodes = [
            node for node in self.nodes.values()
            if node.status == "healthy"
        ]
    
    def _start_health_monitoring(self):
        """Start health monitoring thread."""
        self._health_check_thread = threading.Thread(
            target=self._health_check_loop,
            name="HealthCheck",
            daemon=True
        )
        self._health_check_thread.start()
    
    def _health_check_loop(self):
        """Health check monitoring loop."""
        while not self._stop_monitoring.is_set():
            try:
                self._perform_health_checks()
                self._stop_monitoring.wait(self.health_check_interval)
            except Exception as e:
                self.logger.error(f"Health check error: {e}")
                self._stop_monitoring.wait(5)
    
    def _perform_health_checks(self):
        """Perform health checks on all nodes."""
        current_time = datetime.now(timezone.utc)
        
        with self._lock:
            for node_id, node in list(self.nodes.items()):
                # Check heartbeat age
                if node.last_heartbeat:
                    age = (current_time - node.last_heartbeat).total_seconds()
                    
                    if age > self.health_check_interval * 2:
                        if node.status == "healthy":
                            node.status = "degraded"
                            self.logger.warning(f"Node {node_id} degraded - stale heartbeat")
                    
                    if age > self.health_check_interval * 4:
                        if node.status != "unhealthy":
                            node.status = "unhealthy"
                            self.logger.error(f"Node {node_id} unhealthy - no heartbeat")
                
                # TODO: Add actual health check HTTP requests
                
            self._update_healthy_nodes()
    
    def update_node_heartbeat(self, node_id: str, metrics: Dict[str, Any] = None):
        """Update node heartbeat and metrics.
        
        Args:
            node_id: Node ID
            metrics: Optional performance metrics
        """
        with self._lock:
            if node_id in self.nodes:
                node = self.nodes[node_id]
                node.last_heartbeat = datetime.now(timezone.utc)
                
                # Update status based on metrics
                if metrics:
                    if metrics.get("cpu_usage", 0) > 90 or metrics.get("memory_usage", 0) > 95:
                        node.status = "degraded"
                    else:
                        node.status = "healthy"
                    
                    # Update performance score
                    response_time = metrics.get("avg_response_time", 1.0)
                    error_rate = metrics.get("error_rate", 0.0)
                    
                    # Simple performance scoring
                    score = 1.0 / max(response_time, 0.1) * (1.0 - error_rate)
                    node.performance_score = 0.9 * node.performance_score + 0.1 * score
                
                self._update_healthy_nodes()
    
    def get_load_stats(self) -> Dict[str, Any]:
        """Get load balancing statistics."""
        with self._lock:
            total_requests = sum(self.request_counts.values())
            
            node_stats = {}
            for node_id, node in self.nodes.items():
                node_requests = self.request_counts.get(node_id, 0)
                avg_response_time = self._get_avg_response_time(node_id)
                
                node_stats[node_id] = {
                    "status": node.status,
                    "current_load": node.current_load,
                    "capacity": node.capacity,
                    "utilization": node.current_load / max(node.capacity, 1),
                    "requests_handled": node_requests,
                    "request_percentage": node_requests / max(total_requests, 1) * 100,
                    "avg_response_time": avg_response_time,
                    "performance_score": node.performance_score,
                    "region": node.region
                }
            
            return {
                "total_nodes": len(self.nodes),
                "healthy_nodes": len(self.healthy_nodes),
                "total_requests": total_requests,
                "load_balancing_strategy": self.strategy,
                "node_stats": node_stats
            }


class AutoScaler:
    """Auto-scaling engine with predictive scaling."""
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize auto-scaler.
        
        Args:
            config: Auto-scaling configuration
        """
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Scaling configuration
        self.min_nodes = self.config.get("min_nodes", 2)
        self.max_nodes = self.config.get("max_nodes", 20)
        self.target_cpu_utilization = self.config.get("target_cpu_utilization", 70)
        self.scale_up_threshold = self.config.get("scale_up_threshold", 80)
        self.scale_down_threshold = self.config.get("scale_down_threshold", 40)
        
        # Scaling cooldown periods
        self.scale_up_cooldown = self.config.get("scale_up_cooldown", 300)  # 5 minutes
        self.scale_down_cooldown = self.config.get("scale_down_cooldown", 600)  # 10 minutes
        
        # Metrics collection
        self.metrics_history = deque(maxlen=1000)
        self.scaling_events = deque(maxlen=100)
        
        # Load balancer integration
        self.load_balancer = None
        
        # Predictive scaling
        self.enable_predictive_scaling = self.config.get("predictive_scaling", True)
        self.prediction_window = self.config.get("prediction_window", 900)  # 15 minutes
        
        # Threading
        self._lock = threading.RLock()
        self._scaling_thread = None
        self._stop_scaling = threading.Event()
        
        # Last scaling actions
        self.last_scale_up = None
        self.last_scale_down = None
        
        self._start_scaling_monitor()
    
    def set_load_balancer(self, load_balancer: LoadBalancer):
        """Set load balancer for integration.
        
        Args:
            load_balancer: Load balancer instance
        """
        self.load_balancer = load_balancer
    
    def record_metrics(self, metrics: WorkloadMetrics):
        """Record workload metrics for scaling decisions.
        
        Args:
            metrics: Workload metrics
        """
        with self._lock:
            self.metrics_history.append(metrics)
    
    def _start_scaling_monitor(self):
        """Start scaling monitoring thread."""
        self._scaling_thread = threading.Thread(
            target=self._scaling_loop,
            name="AutoScaler",
            daemon=True
        )
        self._scaling_thread.start()
    
    def _scaling_loop(self):
        """Main auto-scaling loop."""
        while not self._stop_scaling.is_set():
            try:
                self._evaluate_scaling_decision()
                self._stop_scaling.wait(60)  # Check every minute
            except Exception as e:
                self.logger.error(f"Auto-scaling error: {e}")
                self._stop_scaling.wait(30)
    
    def _evaluate_scaling_decision(self):
        """Evaluate whether scaling action is needed."""
        if not self.metrics_history:
            return
        
        # Get recent metrics (last 5 minutes)
        current_time = datetime.now(timezone.utc)
        recent_metrics = [
            m for m in self.metrics_history
            if (current_time - m.timestamp).total_seconds() < 300
        ]
        
        if not recent_metrics:
            return
        
        # Calculate average metrics
        avg_cpu = sum(m.cpu_usage for m in recent_metrics) / len(recent_metrics)
        avg_response_time = sum(m.avg_response_time for m in recent_metrics) / len(recent_metrics)
        avg_queue_length = sum(m.queue_length for m in recent_metrics) / len(recent_metrics)
        avg_error_rate = sum(m.error_rate for m in recent_metrics) / len(recent_metrics)
        
        current_nodes = recent_metrics[-1].active_workers
        
        # Determine scaling action
        scale_decision = self._determine_scaling_action(
            avg_cpu, avg_response_time, avg_queue_length, avg_error_rate, current_nodes
        )
        
        if scale_decision["action"] != "none":
            self._execute_scaling_action(scale_decision)
    
    def _determine_scaling_action(self, 
                                avg_cpu: float,
                                avg_response_time: float,
                                avg_queue_length: float,
                                avg_error_rate: float,
                                current_nodes: int) -> Dict[str, Any]:
        """Determine scaling action based on metrics.
        
        Returns:
            Scaling decision dictionary
        """
        decision = {
            "action": "none",
            "target_nodes": current_nodes,
            "reason": "No scaling needed",
            "confidence": 0.0
        }
        
        # Check cooldown periods
        current_time = datetime.now(timezone.utc)
        
        if self.last_scale_up and (current_time - self.last_scale_up).total_seconds() < self.scale_up_cooldown:
            return decision
        
        if self.last_scale_down and (current_time - self.last_scale_down).total_seconds() < self.scale_down_cooldown:
            return decision
        
        # Scale up conditions
        scale_up_score = 0
        scale_up_reasons = []
        
        if avg_cpu > self.scale_up_threshold:
            scale_up_score += 2
            scale_up_reasons.append(f"High CPU usage: {avg_cpu:.1f}%")
        
        if avg_response_time > 2.0:  # > 2 seconds
            scale_up_score += 2
            scale_up_reasons.append(f"High response time: {avg_response_time:.2f}s")
        
        if avg_queue_length > 100:
            scale_up_score += 1
            scale_up_reasons.append(f"High queue length: {avg_queue_length}")
        
        if avg_error_rate > 0.05:  # > 5%
            scale_up_score += 1
            scale_up_reasons.append(f"High error rate: {avg_error_rate:.2%}")
        
        # Scale down conditions
        scale_down_score = 0
        scale_down_reasons = []
        
        if avg_cpu < self.scale_down_threshold:
            scale_down_score += 1
            scale_down_reasons.append(f"Low CPU usage: {avg_cpu:.1f}%")
        
        if avg_response_time < 0.5 and avg_queue_length < 10:
            scale_down_score += 1
            scale_down_reasons.append("Low load conditions")
        
        # Predictive scaling
        if self.enable_predictive_scaling:
            predicted_load = self._predict_future_load()
            if predicted_load > 1.2:  # 20% increase predicted
                scale_up_score += 1
                scale_up_reasons.append("Predicted load increase")
            elif predicted_load < 0.8:  # 20% decrease predicted
                scale_down_score += 1
                scale_down_reasons.append("Predicted load decrease")
        
        # Make decision
        if scale_up_score >= 2 and current_nodes < self.max_nodes:
            target_nodes = min(current_nodes + max(1, current_nodes // 4), self.max_nodes)
            decision = {
                "action": "scale_up",
                "target_nodes": target_nodes,
                "reason": "; ".join(scale_up_reasons),
                "confidence": min(scale_up_score / 4, 1.0)
            }
        elif scale_down_score >= 2 and current_nodes > self.min_nodes:
            target_nodes = max(current_nodes - 1, self.min_nodes)
            decision = {
                "action": "scale_down", 
                "target_nodes": target_nodes,
                "reason": "; ".join(scale_down_reasons),
                "confidence": min(scale_down_score / 3, 1.0)
            }
        
        return decision
    
    def _predict_future_load(self) -> float:
        """Predict future load based on historical patterns.
        
        Returns:
            Load multiplier (1.0 = current load, >1.0 = increase, <1.0 = decrease)
        """
        if len(self.metrics_history) < 10:
            return 1.0
        
        # Simple trend analysis - would use more sophisticated ML in practice
        recent_window = 5
        older_window = 10
        
        recent_metrics = list(self.metrics_history)[-recent_window:]
        older_metrics = list(self.metrics_history)[-(older_window + recent_window):-recent_window]
        
        if not older_metrics:
            return 1.0
        
        recent_avg_cpu = sum(m.cpu_usage for m in recent_metrics) / len(recent_metrics)
        older_avg_cpu = sum(m.cpu_usage for m in older_metrics) / len(older_metrics)
        
        recent_avg_requests = sum(m.total_requests for m in recent_metrics) / len(recent_metrics)
        older_avg_requests = sum(m.total_requests for m in older_metrics) / len(older_metrics)
        
        # Calculate trends
        cpu_trend = recent_avg_cpu / max(older_avg_cpu, 1)
        request_trend = recent_avg_requests / max(older_avg_requests, 1)
        
        # Weighted prediction
        return 0.6 * cpu_trend + 0.4 * request_trend
    
    def _execute_scaling_action(self, decision: Dict[str, Any]):
        """Execute scaling action.
        
        Args:
            decision: Scaling decision
        """
        try:
            current_time = datetime.now(timezone.utc)
            
            # Record scaling event
            event = ScalingEvent(
                timestamp=current_time,
                event_type=decision["action"],
                trigger_reason=decision["reason"],
                nodes_affected=[],  # Would be populated during actual scaling
                target_capacity=decision["target_nodes"],
                success=True  # Would be determined by actual scaling result
            )
            
            self.scaling_events.append(event)
            
            # Update last scaling time
            if decision["action"] == "scale_up":
                self.last_scale_up = current_time
            elif decision["action"] == "scale_down":
                self.last_scale_down = current_time
            
            # Log scaling decision
            self.logger.info(
                f"Auto-scaling {decision['action']}: {decision['reason']} "
                f"(confidence: {decision['confidence']:.2f})"
            )
            
            # In a real implementation, this would trigger actual node provisioning
            # For now, just simulate the scaling
            self._simulate_scaling(decision)
            
        except Exception as e:
            self.logger.error(f"Failed to execute scaling action: {e}")
            
            # Update event as failed
            if self.scaling_events:
                self.scaling_events[-1].success = False
    
    def _simulate_scaling(self, decision: Dict[str, Any]):
        """Simulate scaling action for testing.
        
        Args:
            decision: Scaling decision
        """
        # In practice, this would:
        # 1. Call cloud provider APIs to provision/terminate instances
        # 2. Update load balancer with new nodes
        # 3. Wait for health checks to pass
        # 4. Gradually shift traffic to new nodes
        
        self.logger.info(f"Simulated scaling to {decision['target_nodes']} nodes")
    
    def get_scaling_stats(self) -> Dict[str, Any]:
        """Get auto-scaling statistics."""
        with self._lock:
            if not self.metrics_history:
                return {"error": "No metrics available"}
            
            recent_events = [
                e for e in self.scaling_events
                if (datetime.now(timezone.utc) - e.timestamp).total_seconds() < 3600  # Last hour
            ]
            
            scale_up_events = [e for e in recent_events if e.event_type == "scale_up"]
            scale_down_events = [e for e in recent_events if e.event_type == "scale_down"]
            
            return {
                "min_nodes": self.min_nodes,
                "max_nodes": self.max_nodes,
                "target_cpu_utilization": self.target_cpu_utilization,
                "recent_scaling_events": len(recent_events),
                "scale_up_events_1h": len(scale_up_events),
                "scale_down_events_1h": len(scale_down_events),
                "last_scale_up": self.last_scale_up.isoformat() if self.last_scale_up else None,
                "last_scale_down": self.last_scale_down.isoformat() if self.last_scale_down else None,
                "predictive_scaling_enabled": self.enable_predictive_scaling,
                "metrics_collected": len(self.metrics_history)
            }
    
    def force_scaling_evaluation(self):
        """Force immediate scaling evaluation."""
        self._evaluate_scaling_decision()
    
    def stop(self):
        """Stop auto-scaling."""
        self._stop_scaling.set()
        if self._scaling_thread:
            self._scaling_thread.join(timeout=5)
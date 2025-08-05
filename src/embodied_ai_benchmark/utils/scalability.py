"""Scalability and load balancing utilities for distributed benchmarking."""

import time
import threading
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Callable, Union, Tuple
import logging
from dataclasses import dataclass, field
from enum import Enum
import queue
import socket
import json
import hashlib
from pathlib import Path
import uuid

logger = logging.getLogger(__name__)


class LoadBalancingStrategy(Enum):
    """Load balancing strategies."""
    ROUND_ROBIN = "round_robin"
    LEAST_LOADED = "least_loaded"
    WEIGHTED_ROUND_ROBIN = "weighted_round_robin"
    HASH_BASED = "hash_based"
    RANDOM = "random"


@dataclass
class WorkerNode:
    """Represents a worker node in the distributed system."""
    node_id: str
    host: str
    port: int
    capabilities: List[str]
    max_concurrent_tasks: int
    current_load: int = 0
    total_completed: int = 0
    total_failed: int = 0
    average_task_time: float = 0.0
    last_heartbeat: float = field(default_factory=time.time)
    is_active: bool = True
    weight: float = 1.0
    
    def get_load_percentage(self) -> float:
        """Get current load as percentage of capacity."""
        return (self.current_load / max(1, self.max_concurrent_tasks)) * 100
    
    def can_accept_task(self) -> bool:
        """Check if worker can accept another task."""
        return self.is_active and self.current_load < self.max_concurrent_tasks
    
    def update_heartbeat(self):
        """Update last heartbeat timestamp."""
        self.last_heartbeat = time.time()
    
    def is_healthy(self, heartbeat_timeout: float = 30.0) -> bool:
        """Check if worker is healthy based on heartbeat."""
        return (time.time() - self.last_heartbeat) < heartbeat_timeout


class LoadBalancer:
    """Advanced load balancer for distributing benchmark tasks."""
    
    def __init__(self, 
                 strategy: LoadBalancingStrategy = LoadBalancingStrategy.LEAST_LOADED,
                 health_check_interval: float = 10.0):
        """Initialize load balancer.
        
        Args:
            strategy: Load balancing strategy to use
            health_check_interval: Interval for health checks in seconds
        """
        self.strategy = strategy
        self.health_check_interval = health_check_interval
        
        self.workers: Dict[str, WorkerNode] = {}
        self.task_queue = queue.PriorityQueue()
        self.completed_tasks = queue.Queue()
        
        self._lock = threading.RLock()
        self._round_robin_index = 0
        self._health_check_thread = None
        self._stop_health_check = threading.Event()
        
        self._start_health_monitoring()
    
    def register_worker(self, worker: WorkerNode):
        """Register a new worker node."""
        with self._lock:
            self.workers[worker.node_id] = worker
            logger.info(f"Registered worker: {worker.node_id} at {worker.host}:{worker.port}")
    
    def unregister_worker(self, node_id: str):
        """Unregister a worker node."""
        with self._lock:
            if node_id in self.workers:
                del self.workers[node_id]
                logger.info(f"Unregistered worker: {node_id}")
    
    def get_next_worker(self, task_requirements: Optional[Dict[str, Any]] = None) -> Optional[WorkerNode]:
        """Get next worker based on load balancing strategy."""
        with self._lock:
            available_workers = [
                worker for worker in self.workers.values()
                if worker.can_accept_task() and self._worker_meets_requirements(worker, task_requirements)
            ]
            
            if not available_workers:
                return None
            
            if self.strategy == LoadBalancingStrategy.ROUND_ROBIN:
                return self._round_robin_selection(available_workers)
            elif self.strategy == LoadBalancingStrategy.LEAST_LOADED:
                return self._least_loaded_selection(available_workers)
            elif self.strategy == LoadBalancingStrategy.WEIGHTED_ROUND_ROBIN:
                return self._weighted_round_robin_selection(available_workers)
            elif self.strategy == LoadBalancingStrategy.HASH_BASED:
                return self._hash_based_selection(available_workers, task_requirements)
            elif self.strategy == LoadBalancingStrategy.RANDOM:
                return self._random_selection(available_workers)
            else:
                return available_workers[0]
    
    def _worker_meets_requirements(self, worker: WorkerNode, requirements: Optional[Dict[str, Any]]) -> bool:
        """Check if worker meets task requirements."""
        if not requirements:
            return True
        
        required_capabilities = requirements.get("capabilities", [])
        return all(cap in worker.capabilities for cap in required_capabilities)
    
    def _round_robin_selection(self, workers: List[WorkerNode]) -> WorkerNode:
        """Round-robin worker selection."""
        worker = workers[self._round_robin_index % len(workers)]
        self._round_robin_index += 1
        return worker
    
    def _least_loaded_selection(self, workers: List[WorkerNode]) -> WorkerNode:
        """Select worker with lowest load."""
        return min(workers, key=lambda w: w.get_load_percentage())
    
    def _weighted_round_robin_selection(self, workers: List[WorkerNode]) -> WorkerNode:
        """Weighted round-robin selection based on worker performance."""
        # Create weighted list based on worker weights
        weighted_workers = []
        for worker in workers:
            weight = max(1, int(worker.weight * 10))  # Scale weight
            weighted_workers.extend([worker] * weight)
        
        if weighted_workers:
            worker = weighted_workers[self._round_robin_index % len(weighted_workers)]
            self._round_robin_index += 1
            return worker
        
        return workers[0]
    
    def _hash_based_selection(self, workers: List[WorkerNode], requirements: Optional[Dict[str, Any]]) -> WorkerNode:
        """Hash-based selection for task affinity."""
        if requirements and "affinity_key" in requirements:
            key = str(requirements["affinity_key"])
            hash_value = int(hashlib.md5(key.encode()).hexdigest(), 16)
            return workers[hash_value % len(workers)]
        
        return self._least_loaded_selection(workers)
    
    def _random_selection(self, workers: List[WorkerNode]) -> WorkerNode:
        """Random worker selection."""
        import random
        return random.choice(workers)
    
    def assign_task(self, task_id: str, 
                   task_data: Dict[str, Any],
                   priority: int = 1,
                   requirements: Optional[Dict[str, Any]] = None) -> Optional[str]:
        """Assign task to a worker.
        
        Args:
            task_id: Unique task identifier
            task_data: Task data and parameters
            priority: Task priority (lower = higher priority)
            requirements: Task requirements for worker selection
            
        Returns:
            Worker node ID if assigned, None if no workers available
        """
        worker = self.get_next_worker(requirements)
        
        if worker is None:
            logger.warning(f"No available workers for task {task_id}")
            return None
        
        with self._lock:
            worker.current_load += 1
            
        task_info = {
            "task_id": task_id,
            "worker_id": worker.node_id,
            "task_data": task_data,
            "assigned_time": time.time(),
            "priority": priority
        }
        
        # Add to task queue (priority queue uses tuples: (priority, task_info))
        self.task_queue.put((priority, task_info))
        
        logger.info(f"Assigned task {task_id} to worker {worker.node_id}")
        return worker.node_id
    
    def complete_task(self, task_id: str, worker_id: str, 
                     result: Dict[str, Any], success: bool = True):
        """Mark task as completed and update worker statistics."""
        with self._lock:
            if worker_id in self.workers:
                worker = self.workers[worker_id]
                worker.current_load = max(0, worker.current_load - 1)
                
                if success:
                    worker.total_completed += 1
                else:
                    worker.total_failed += 1
                
                # Update average task time
                task_time = result.get("execution_time", 0.0)
                if task_time > 0:
                    total_tasks = worker.total_completed + worker.total_failed
                    worker.average_task_time = (
                        (worker.average_task_time * (total_tasks - 1) + task_time) / total_tasks
                    )
                
                # Update worker weight based on performance
                self._update_worker_weight(worker)
        
        # Add to completed tasks queue
        completion_info = {
            "task_id": task_id,
            "worker_id": worker_id,
            "result": result,
            "success": success,
            "completion_time": time.time()
        }
        
        self.completed_tasks.put(completion_info)
        logger.info(f"Task {task_id} completed on worker {worker_id}")
    
    def _update_worker_weight(self, worker: WorkerNode):
        """Update worker weight based on performance metrics."""
        # Calculate success rate
        total_tasks = worker.total_completed + worker.total_failed
        success_rate = worker.total_completed / max(1, total_tasks)
        
        # Calculate performance score (lower task time = better)
        time_score = 1.0 / max(0.1, worker.average_task_time)  # Avoid division by zero
        
        # Combine metrics to update weight
        performance_score = success_rate * time_score
        worker.weight = max(0.1, min(2.0, performance_score))  # Clamp between 0.1 and 2.0
    
    def get_worker_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get statistics for all workers."""
        with self._lock:
            stats = {}
            for worker_id, worker in self.workers.items():
                stats[worker_id] = {
                    "host": worker.host,
                    "port": worker.port,
                    "current_load": worker.current_load,
                    "max_concurrent_tasks": worker.max_concurrent_tasks,
                    "load_percentage": worker.get_load_percentage(),
                    "total_completed": worker.total_completed,
                    "total_failed": worker.total_failed,
                    "success_rate": worker.total_completed / max(1, worker.total_completed + worker.total_failed),
                    "average_task_time": worker.average_task_time,
                    "weight": worker.weight,
                    "is_active": worker.is_active,
                    "is_healthy": worker.is_healthy()
                }
            return stats
    
    def _start_health_monitoring(self):
        """Start health monitoring thread."""
        self._health_check_thread = threading.Thread(target=self._health_check_loop, daemon=True)
        self._health_check_thread.start()
    
    def _health_check_loop(self):
        """Health check monitoring loop."""
        while not self._stop_health_check.wait(self.health_check_interval):
            try:
                self._perform_health_checks()
            except Exception as e:
                logger.error(f"Health check error: {e}")
    
    def _perform_health_checks(self):
        """Perform health checks on all workers."""
        with self._lock:
            unhealthy_workers = []
            
            for worker_id, worker in self.workers.items():
                if not worker.is_healthy():
                    worker.is_active = False
                    unhealthy_workers.append(worker_id)
                    logger.warning(f"Worker {worker_id} marked as unhealthy")
            
            if unhealthy_workers:
                logger.info(f"Found {len(unhealthy_workers)} unhealthy workers")
    
    def shutdown(self):
        """Shutdown load balancer and cleanup resources."""
        self._stop_health_check.set()
        if self._health_check_thread:
            self._health_check_thread.join(timeout=5.0)
        
        logger.info("Load balancer shutdown complete")


class DistributedBenchmark:
    """Distributed benchmark execution system."""
    
    def __init__(self, 
                 load_balancer: LoadBalancer,
                 result_aggregation_strategy: str = "average"):
        """Initialize distributed benchmark system.
        
        Args:
            load_balancer: Load balancer instance
            result_aggregation_strategy: How to aggregate results from workers
        """
        self.load_balancer = load_balancer
        self.result_aggregation_strategy = result_aggregation_strategy
        
        self.active_tasks: Dict[str, Dict[str, Any]] = {}
        self.completed_results: Dict[str, Dict[str, Any]] = {}
        
        self._lock = threading.Lock()
    
    def distribute_evaluation(self, 
                            evaluation_tasks: List[Dict[str, Any]],
                            timeout: float = 300.0) -> Dict[str, Any]:
        """Distribute evaluation tasks across workers.
        
        Args:
            evaluation_tasks: List of evaluation task configurations
            timeout: Maximum time to wait for all tasks to complete
            
        Returns:
            Aggregated evaluation results
        """
        if not evaluation_tasks:
            return {"error": "No evaluation tasks provided"}
        
        logger.info(f"Distributing {len(evaluation_tasks)} evaluation tasks")
        
        # Assign tasks to workers
        task_assignments = {}
        for i, task in enumerate(evaluation_tasks):
            task_id = f"eval_task_{i}_{uuid.uuid4().hex[:8]}"
            
            worker_id = self.load_balancer.assign_task(
                task_id=task_id,
                task_data=task,
                priority=task.get("priority", 1),
                requirements=task.get("requirements")
            )
            
            if worker_id:
                task_assignments[task_id] = {
                    "worker_id": worker_id,
                    "task_data": task,
                    "start_time": time.time()
                }
                
                with self._lock:
                    self.active_tasks[task_id] = task_assignments[task_id]
            else:
                logger.error(f"Failed to assign task {task_id} - no available workers")
        
        if not task_assignments:
            return {"error": "No tasks could be assigned to workers"}
        
        # Wait for task completion
        start_time = time.time()
        while time.time() - start_time < timeout:
            # Check for completed tasks
            self._collect_completed_tasks()
            
            # Check if all tasks are complete
            with self._lock:
                if not self.active_tasks:
                    break
            
            time.sleep(1.0)  # Poll interval
        
        # Handle timeout
        with self._lock:
            if self.active_tasks:
                logger.warning(f"Timeout: {len(self.active_tasks)} tasks still active")
                # Mark remaining tasks as failed
                for task_id in list(self.active_tasks.keys()):
                    self.completed_results[task_id] = {
                        "error": "Task timeout",
                        "task_id": task_id
                    }
                    del self.active_tasks[task_id]
        
        # Aggregate results
        return self._aggregate_results()
    
    def _collect_completed_tasks(self):
        """Collect completed tasks from load balancer."""
        while not self.load_balancer.completed_tasks.empty():
            try:
                completion_info = self.load_balancer.completed_tasks.get_nowait()
                task_id = completion_info["task_id"]
                
                with self._lock:
                    if task_id in self.active_tasks:
                        # Move from active to completed
                        self.completed_results[task_id] = completion_info
                        del self.active_tasks[task_id]
                        
                        logger.debug(f"Collected completed task: {task_id}")
                
            except queue.Empty:
                break
            except Exception as e:
                logger.error(f"Error collecting completed task: {e}")
    
    def _aggregate_results(self) -> Dict[str, Any]:
        """Aggregate results from all completed tasks."""
        if not self.completed_results:
            return {"error": "No completed results to aggregate"}
        
        successful_results = []
        failed_results = []
        
        for task_id, result in self.completed_results.items():
            if result.get("success", False) and "error" not in result:
                successful_results.append(result)
            else:
                failed_results.append(result)
        
        total_tasks = len(self.completed_results)
        success_count = len(successful_results)
        
        aggregated = {
            "total_tasks": total_tasks,
            "successful_tasks": success_count,
            "failed_tasks": len(failed_results),
            "success_rate": success_count / max(1, total_tasks),
            "worker_stats": self.load_balancer.get_worker_stats()
        }
        
        if successful_results:
            # Aggregate metrics from successful tasks
            if self.result_aggregation_strategy == "average":
                aggregated.update(self._average_aggregation(successful_results))
            elif self.result_aggregation_strategy == "weighted_average":
                aggregated.update(self._weighted_average_aggregation(successful_results))
            else:
                aggregated["results"] = successful_results
        
        if failed_results:
            aggregated["failed_tasks_details"] = failed_results
        
        logger.info(f"Aggregation complete: {success_count}/{total_tasks} tasks successful")
        return aggregated
    
    def _average_aggregation(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate results using simple averaging."""
        if not results:
            return {}
        
        # Extract numeric metrics
        numeric_metrics = {}
        for result in results:
            task_result = result.get("result", {})
            for key, value in task_result.items():
                if isinstance(value, (int, float)):
                    if key not in numeric_metrics:
                        numeric_metrics[key] = []
                    numeric_metrics[key].append(value)
        
        # Compute averages
        averaged_metrics = {}
        for key, values in numeric_metrics.items():
            if values:
                averaged_metrics[f"avg_{key}"] = sum(values) / len(values)
                averaged_metrics[f"min_{key}"] = min(values)
                averaged_metrics[f"max_{key}"] = max(values)
                averaged_metrics[f"std_{key}"] = (sum((x - averaged_metrics[f"avg_{key}"]) ** 2 
                                                     for x in values) / len(values)) ** 0.5
        
        return {"aggregated_metrics": averaged_metrics}
    
    def _weighted_average_aggregation(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate results using weighted averaging based on task complexity."""
        # Simple implementation - can be extended with more sophisticated weighting
        return self._average_aggregation(results)


class AutoScaler:
    """Auto-scaling system for dynamic worker management."""
    
    def __init__(self, 
                 load_balancer: LoadBalancer,
                 min_workers: int = 1,
                 max_workers: int = 10,
                 scale_up_threshold: float = 80.0,
                 scale_down_threshold: float = 20.0,
                 scale_check_interval: float = 30.0):
        """Initialize auto-scaler.
        
        Args:
            load_balancer: Load balancer instance
            min_workers: Minimum number of workers
            max_workers: Maximum number of workers
            scale_up_threshold: Load percentage to trigger scale up
            scale_down_threshold: Load percentage to trigger scale down
            scale_check_interval: Interval between scaling decisions
        """
        self.load_balancer = load_balancer
        self.min_workers = min_workers
        self.max_workers = max_workers
        self.scale_up_threshold = scale_up_threshold
        self.scale_down_threshold = scale_down_threshold
        self.scale_check_interval = scale_check_interval
        
        self._scaling_thread = None
        self._stop_scaling = threading.Event()
        self._scaling_history = []
        
        self.start_auto_scaling()
    
    def start_auto_scaling(self):
        """Start auto-scaling monitoring."""
        self._scaling_thread = threading.Thread(target=self._scaling_loop, daemon=True)
        self._scaling_thread.start()
        logger.info("Auto-scaling started")
    
    def stop_auto_scaling(self):
        """Stop auto-scaling monitoring."""
        self._stop_scaling.set()
        if self._scaling_thread:
            self._scaling_thread.join(timeout=5.0)
        logger.info("Auto-scaling stopped")
    
    def _scaling_loop(self):
        """Main auto-scaling loop."""
        while not self._stop_scaling.wait(self.scale_check_interval):
            try:
                self._evaluate_scaling_decision()
            except Exception as e:
                logger.error(f"Auto-scaling error: {e}")
    
    def _evaluate_scaling_decision(self):
        """Evaluate whether to scale up or down."""
        worker_stats = self.load_balancer.get_worker_stats()
        
        if not worker_stats:
            logger.warning("No workers available for scaling decisions")
            return
        
        # Calculate average load across healthy workers
        healthy_workers = [
            stats for stats in worker_stats.values()
            if stats["is_healthy"] and stats["is_active"]
        ]
        
        if not healthy_workers:
            logger.warning("No healthy workers for scaling decisions")
            return
        
        avg_load = sum(w["load_percentage"] for w in healthy_workers) / len(healthy_workers)
        current_worker_count = len(healthy_workers)
        
        scaling_decision = None
        
        # Scale up decision
        if (avg_load > self.scale_up_threshold and 
            current_worker_count < self.max_workers):
            scaling_decision = "scale_up"
            logger.info(f"Scaling up: avg_load={avg_load:.1f}%, workers={current_worker_count}")
            self._scale_up()
        
        # Scale down decision
        elif (avg_load < self.scale_down_threshold and 
              current_worker_count > self.min_workers):
            scaling_decision = "scale_down"
            logger.info(f"Scaling down: avg_load={avg_load:.1f}%, workers={current_worker_count}")
            self._scale_down()
        
        # Record scaling history
        if scaling_decision:
            self._scaling_history.append({
                "timestamp": time.time(),
                "decision": scaling_decision,
                "avg_load": avg_load,
                "worker_count": current_worker_count
            })
            
            # Keep only recent history
            if len(self._scaling_history) > 100:
                self._scaling_history = self._scaling_history[-50:]
    
    def _scale_up(self):
        """Scale up by adding workers."""
        # This is a placeholder - in a real implementation,
        # you would integrate with container orchestration or cloud APIs
        logger.info("Scale up triggered - would add new worker nodes")
        
        # Example: Create a new worker node
        new_worker = WorkerNode(
            node_id=f"auto_worker_{uuid.uuid4().hex[:8]}",
            host="localhost",
            port=8000 + len(self.load_balancer.workers),
            capabilities=["cpu", "memory"],
            max_concurrent_tasks=4
        )
        
        self.load_balancer.register_worker(new_worker)
    
    def _scale_down(self):
        """Scale down by removing workers."""
        # This is a placeholder - in a real implementation,
        # you would gracefully remove the least utilized workers
        logger.info("Scale down triggered - would remove worker nodes")
        
        # Find worker with lowest load to remove
        worker_stats = self.load_balancer.get_worker_stats()
        if len(worker_stats) <= self.min_workers:
            return
        
        least_loaded_worker = min(
            worker_stats.items(),
            key=lambda x: x[1]["load_percentage"]
        )
        
        worker_id = least_loaded_worker[0]
        if least_loaded_worker[1]["current_load"] == 0:  # Only remove idle workers
            self.load_balancer.unregister_worker(worker_id)
            logger.info(f"Removed worker: {worker_id}")
    
    def get_scaling_history(self) -> List[Dict[str, Any]]:
        """Get scaling decision history."""
        return self._scaling_history.copy()


# Global instances for easy access
global_load_balancer = LoadBalancer()
global_distributed_benchmark = DistributedBenchmark(global_load_balancer)
global_auto_scaler = AutoScaler(global_load_balancer)
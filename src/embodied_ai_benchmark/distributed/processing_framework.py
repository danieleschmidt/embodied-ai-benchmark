"""Distributed processing framework for large-scale embodied AI benchmarking."""

import time
import json
import threading
import logging
import uuid
import hashlib
from typing import Any, Dict, List, Optional, Callable, Tuple, Union
from dataclasses import dataclass, asdict
from datetime import datetime, timezone, timedelta
from collections import defaultdict, deque
import concurrent.futures
import queue
import pickle
import sqlite3
import os


@dataclass
class Task:
    """Distributed task definition."""
    task_id: str
    task_type: str
    payload: Dict[str, Any]
    priority: int
    created_at: datetime
    timeout_seconds: int
    dependencies: List[str]
    required_capabilities: List[str]
    estimated_runtime: float
    max_retries: int
    retry_count: int = 0


@dataclass
class TaskResult:
    """Task execution result."""
    task_id: str
    worker_id: str
    status: str  # "success", "failure", "timeout", "cancelled"
    result: Any
    error: Optional[str]
    execution_time: float
    started_at: datetime
    completed_at: datetime
    metadata: Dict[str, Any]


@dataclass
class WorkerInfo:
    """Worker node information."""
    worker_id: str
    endpoint: str
    capabilities: List[str]
    max_concurrent_tasks: int
    current_tasks: int
    last_heartbeat: datetime
    performance_score: float
    status: str  # "active", "busy", "inactive", "failed"
    region: str
    resource_usage: Dict[str, float]


class TaskQueue:
    """Priority-based distributed task queue."""
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize task queue.
        
        Args:
            config: Queue configuration
        """
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Queue storage
        self.pending_tasks = queue.PriorityQueue()
        self.running_tasks = {}
        self.completed_tasks = {}
        self.failed_tasks = {}
        
        # Task persistence
        self.db_path = self.config.get("db_path", "/tmp/task_queue.db")
        self._init_database()
        
        # Task tracking
        self.task_counter = 0
        self.total_tasks_processed = 0
        
        # Threading
        self._lock = threading.RLock()
        
        # Load persisted tasks
        self._load_persisted_tasks()
    
    def _init_database(self):
        """Initialize SQLite database for task persistence."""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        
        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS tasks (
                task_id TEXT PRIMARY KEY,
                task_type TEXT,
                payload TEXT,
                priority INTEGER,
                created_at REAL,
                timeout_seconds INTEGER,
                dependencies TEXT,
                required_capabilities TEXT,
                estimated_runtime REAL,
                max_retries INTEGER,
                retry_count INTEGER,
                status TEXT,
                worker_id TEXT,
                result TEXT,
                error TEXT,
                execution_time REAL,
                started_at REAL,
                completed_at REAL
            )
        """)
        conn.commit()
        conn.close()
    
    def _load_persisted_tasks(self):
        """Load persisted tasks from database."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.execute(
                "SELECT * FROM tasks WHERE status IN ('pending', 'running')"
            )
            
            for row in cursor.fetchall():
                task = self._row_to_task(row)
                if task.status == 'pending':
                    priority_item = (-task.priority, task.created_at.timestamp(), task)
                    self.pending_tasks.put(priority_item)
                elif task.status == 'running':
                    self.running_tasks[task.task_id] = task
            
            conn.close()
            self.logger.info("Loaded persisted tasks from database")
            
        except Exception as e:
            self.logger.error(f"Failed to load persisted tasks: {e}")
    
    def _row_to_task(self, row) -> Task:
        """Convert database row to Task object."""
        return Task(
            task_id=row[0],
            task_type=row[1],
            payload=json.loads(row[2]),
            priority=row[3],
            created_at=datetime.fromtimestamp(row[4], timezone.utc),
            timeout_seconds=row[5],
            dependencies=json.loads(row[6]),
            required_capabilities=json.loads(row[7]),
            estimated_runtime=row[8],
            max_retries=row[9],
            retry_count=row[10]
        )
    
    def submit_task(self, task: Task) -> str:
        """Submit task to queue.
        
        Args:
            task: Task to submit
            
        Returns:
            Task ID
        """
        with self._lock:
            if not task.task_id:
                task.task_id = f"task_{uuid.uuid4().hex[:8]}"
            
            # Check dependencies
            if task.dependencies:
                for dep_id in task.dependencies:
                    if dep_id not in self.completed_tasks:
                        self.logger.warning(f"Task {task.task_id} depends on incomplete task {dep_id}")
            
            # Add to queue with priority (negative for max-heap behavior)
            priority_item = (-task.priority, task.created_at.timestamp(), task)
            self.pending_tasks.put(priority_item)
            
            # Persist to database
            self._persist_task(task, "pending")
            
            self.task_counter += 1
            self.logger.info(f"Submitted task {task.task_id} with priority {task.priority}")
            
            return task.task_id
    
    def get_next_task(self, worker_capabilities: List[str] = None) -> Optional[Task]:
        """Get next task for execution.
        
        Args:
            worker_capabilities: Capabilities of requesting worker
            
        Returns:
            Next task or None if no suitable tasks
        """
        with self._lock:
            # Find suitable task
            temp_queue = queue.PriorityQueue()
            selected_task = None
            
            while not self.pending_tasks.empty():
                priority_item = self.pending_tasks.get()
                _, _, task = priority_item
                
                # Check if task is suitable
                if self._is_task_suitable(task, worker_capabilities):
                    selected_task = task
                    break
                else:
                    # Put back in queue
                    temp_queue.put(priority_item)
            
            # Restore queue
            while not temp_queue.empty():
                self.pending_tasks.put(temp_queue.get())
            
            if selected_task:
                # Move to running tasks
                self.running_tasks[selected_task.task_id] = selected_task
                self._persist_task(selected_task, "running")
                
                self.logger.info(f"Assigned task {selected_task.task_id} to worker")
            
            return selected_task
    
    def _is_task_suitable(self, task: Task, worker_capabilities: List[str]) -> bool:
        """Check if task is suitable for worker.
        
        Args:
            task: Task to check
            worker_capabilities: Worker capabilities
            
        Returns:
            Whether task is suitable
        """
        # Check capabilities
        if task.required_capabilities and worker_capabilities:
            if not all(cap in worker_capabilities for cap in task.required_capabilities):
                return False
        
        # Check dependencies
        if task.dependencies:
            for dep_id in task.dependencies:
                if dep_id not in self.completed_tasks:
                    return False
        
        # Check timeout
        age = (datetime.now(timezone.utc) - task.created_at).total_seconds()
        if age > task.timeout_seconds:
            # Task has timed out
            self._move_task_to_failed(task, "Task timeout before execution")
            return False
        
        return True
    
    def complete_task(self, result: TaskResult):
        """Mark task as completed.
        
        Args:
            result: Task execution result
        """
        with self._lock:
            task_id = result.task_id
            
            if task_id in self.running_tasks:
                task = self.running_tasks.pop(task_id)
                
                if result.status == "success":
                    self.completed_tasks[task_id] = result
                    self._persist_result(result, "completed")
                    self.total_tasks_processed += 1
                    self.logger.info(f"Task {task_id} completed successfully")
                else:
                    # Handle failure
                    if task.retry_count < task.max_retries:
                        # Retry task
                        task.retry_count += 1
                        retry_delay = min(60 * (2 ** task.retry_count), 3600)  # Exponential backoff, max 1 hour
                        
                        # Add delay before retrying (simplified - would use scheduler in practice)
                        task.created_at = datetime.now(timezone.utc) + timedelta(seconds=retry_delay)
                        
                        priority_item = (-task.priority, task.created_at.timestamp(), task)
                        self.pending_tasks.put(priority_item)
                        self._persist_task(task, "pending")
                        
                        self.logger.info(f"Task {task_id} failed, retrying (attempt {task.retry_count}/{task.max_retries})")
                    else:
                        # Max retries exceeded
                        self.failed_tasks[task_id] = result
                        self._persist_result(result, "failed")
                        self.logger.error(f"Task {task_id} failed permanently: {result.error}")
    
    def _move_task_to_failed(self, task: Task, error: str):
        """Move task to failed state."""
        result = TaskResult(
            task_id=task.task_id,
            worker_id="system",
            status="failure",
            result=None,
            error=error,
            execution_time=0,
            started_at=datetime.now(timezone.utc),
            completed_at=datetime.now(timezone.utc),
            metadata={}
        )
        
        self.failed_tasks[task.task_id] = result
        self._persist_result(result, "failed")
    
    def _persist_task(self, task: Task, status: str):
        """Persist task to database."""
        try:
            conn = sqlite3.connect(self.db_path)
            conn.execute("""
                INSERT OR REPLACE INTO tasks (
                    task_id, task_type, payload, priority, created_at, timeout_seconds,
                    dependencies, required_capabilities, estimated_runtime, max_retries,
                    retry_count, status
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                task.task_id, task.task_type, json.dumps(task.payload), task.priority,
                task.created_at.timestamp(), task.timeout_seconds,
                json.dumps(task.dependencies), json.dumps(task.required_capabilities),
                task.estimated_runtime, task.max_retries, task.retry_count, status
            ))
            conn.commit()
            conn.close()
        except Exception as e:
            self.logger.error(f"Failed to persist task: {e}")
    
    def _persist_result(self, result: TaskResult, status: str):
        """Persist task result to database."""
        try:
            conn = sqlite3.connect(self.db_path)
            conn.execute("""
                UPDATE tasks SET 
                status = ?, worker_id = ?, result = ?, error = ?,
                execution_time = ?, started_at = ?, completed_at = ?
                WHERE task_id = ?
            """, (
                status, result.worker_id, json.dumps(result.result), result.error,
                result.execution_time, result.started_at.timestamp(),
                result.completed_at.timestamp(), result.task_id
            ))
            conn.commit()
            conn.close()
        except Exception as e:
            self.logger.error(f"Failed to persist result: {e}")
    
    def get_queue_stats(self) -> Dict[str, Any]:
        """Get queue statistics."""
        with self._lock:
            return {
                "pending_tasks": self.pending_tasks.qsize(),
                "running_tasks": len(self.running_tasks),
                "completed_tasks": len(self.completed_tasks),
                "failed_tasks": len(self.failed_tasks),
                "total_submitted": self.task_counter,
                "total_processed": self.total_tasks_processed
            }


class DistributedWorkerManager:
    """Manages distributed workers for task execution."""
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize worker manager.
        
        Args:
            config: Manager configuration
        """
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Worker management
        self.workers = {}
        self.task_assignments = defaultdict(list)
        
        # Health monitoring
        self.heartbeat_timeout = self.config.get("heartbeat_timeout", 300)  # 5 minutes
        self.health_check_interval = self.config.get("health_check_interval", 60)
        
        # Task execution
        self.task_queue = TaskQueue(self.config.get("task_queue", {}))
        
        # Performance tracking
        self.worker_performance = defaultdict(lambda: {"tasks_completed": 0, "avg_execution_time": 0, "error_rate": 0})
        
        # Threading
        self._lock = threading.RLock()
        self._health_check_thread = None
        self._stop_monitoring = threading.Event()
        
        self._start_health_monitoring()
    
    def register_worker(self, worker: WorkerInfo):
        """Register a new worker.
        
        Args:
            worker: Worker information
        """
        with self._lock:
            self.workers[worker.worker_id] = worker
            worker.status = "active"
            worker.last_heartbeat = datetime.now(timezone.utc)
            
            self.logger.info(f"Registered worker {worker.worker_id} with capabilities: {worker.capabilities}")
    
    def unregister_worker(self, worker_id: str):
        """Unregister a worker.
        
        Args:
            worker_id: Worker ID to unregister
        """
        with self._lock:
            if worker_id in self.workers:
                # Reassign tasks from this worker
                self._reassign_worker_tasks(worker_id)
                
                del self.workers[worker_id]
                self.logger.info(f"Unregistered worker {worker_id}")
    
    def submit_task(self, 
                   task_type: str,
                   payload: Dict[str, Any],
                   priority: int = 0,
                   timeout_seconds: int = 3600,
                   dependencies: List[str] = None,
                   required_capabilities: List[str] = None,
                   estimated_runtime: float = 60.0,
                   max_retries: int = 3) -> str:
        """Submit a new task for distributed execution.
        
        Args:
            task_type: Type of task
            payload: Task data
            priority: Task priority (higher = more important)
            timeout_seconds: Task timeout
            dependencies: Task dependencies (task IDs)
            required_capabilities: Required worker capabilities
            estimated_runtime: Estimated execution time
            max_retries: Maximum retry attempts
            
        Returns:
            Task ID
        """
        task = Task(
            task_id=f"task_{uuid.uuid4().hex[:8]}",
            task_type=task_type,
            payload=payload,
            priority=priority,
            created_at=datetime.now(timezone.utc),
            timeout_seconds=timeout_seconds,
            dependencies=dependencies or [],
            required_capabilities=required_capabilities or [],
            estimated_runtime=estimated_runtime,
            max_retries=max_retries
        )
        
        return self.task_queue.submit_task(task)
    
    def request_task(self, worker_id: str) -> Optional[Task]:
        """Worker requests next task.
        
        Args:
            worker_id: ID of requesting worker
            
        Returns:
            Next task or None
        """
        with self._lock:
            if worker_id not in self.workers:
                return None
            
            worker = self.workers[worker_id]
            
            # Check if worker can take more tasks
            if worker.current_tasks >= worker.max_concurrent_tasks:
                return None
            
            # Get next suitable task
            task = self.task_queue.get_next_task(worker.capabilities)
            
            if task:
                # Assign task to worker
                worker.current_tasks += 1
                self.task_assignments[worker_id].append(task.task_id)
                
                self.logger.info(f"Assigned task {task.task_id} to worker {worker_id}")
            
            return task
    
    def report_task_completion(self, result: TaskResult):
        """Worker reports task completion.
        
        Args:
            result: Task execution result
        """
        with self._lock:
            # Update task queue
            self.task_queue.complete_task(result)
            
            # Update worker state
            if result.worker_id in self.workers:
                worker = self.workers[result.worker_id]
                worker.current_tasks = max(0, worker.current_tasks - 1)
                
                # Remove from assignments
                if result.task_id in self.task_assignments[result.worker_id]:
                    self.task_assignments[result.worker_id].remove(result.task_id)
                
                # Update performance metrics
                self._update_worker_performance(result)
            
            self.logger.info(f"Task {result.task_id} completed by worker {result.worker_id} with status {result.status}")
    
    def worker_heartbeat(self, worker_id: str, resource_usage: Dict[str, float] = None):
        """Update worker heartbeat.
        
        Args:
            worker_id: Worker ID
            resource_usage: Current resource usage
        """
        with self._lock:
            if worker_id in self.workers:
                worker = self.workers[worker_id]
                worker.last_heartbeat = datetime.now(timezone.utc)
                
                if resource_usage:
                    worker.resource_usage = resource_usage
                    
                    # Update status based on resource usage
                    if resource_usage.get("cpu", 0) > 95 or resource_usage.get("memory", 0) > 95:
                        worker.status = "busy"
                    else:
                        worker.status = "active"
    
    def _update_worker_performance(self, result: TaskResult):
        """Update worker performance metrics."""
        perf = self.worker_performance[result.worker_id]
        
        # Update task count
        perf["tasks_completed"] += 1
        
        # Update average execution time
        current_avg = perf["avg_execution_time"]
        new_avg = (current_avg * (perf["tasks_completed"] - 1) + result.execution_time) / perf["tasks_completed"]
        perf["avg_execution_time"] = new_avg
        
        # Update error rate
        if result.status != "success":
            errors = perf.get("errors", 0) + 1
            perf["errors"] = errors
            perf["error_rate"] = errors / perf["tasks_completed"]
    
    def _start_health_monitoring(self):
        """Start health monitoring thread."""
        self._health_check_thread = threading.Thread(
            target=self._health_check_loop,
            name="WorkerHealthCheck",
            daemon=True
        )
        self._health_check_thread.start()
    
    def _health_check_loop(self):
        """Health check monitoring loop."""
        while not self._stop_monitoring.is_set():
            try:
                self._check_worker_health()
                self._stop_monitoring.wait(self.health_check_interval)
            except Exception as e:
                self.logger.error(f"Health check error: {e}")
                self._stop_monitoring.wait(30)
    
    def _check_worker_health(self):
        """Check health of all workers."""
        current_time = datetime.now(timezone.utc)
        
        with self._lock:
            failed_workers = []
            
            for worker_id, worker in list(self.workers.items()):
                # Check heartbeat age
                if worker.last_heartbeat:
                    age = (current_time - worker.last_heartbeat).total_seconds()
                    
                    if age > self.heartbeat_timeout:
                        worker.status = "failed"
                        failed_workers.append(worker_id)
                        self.logger.error(f"Worker {worker_id} failed - no heartbeat for {age}s")
            
            # Handle failed workers
            for worker_id in failed_workers:
                self._handle_worker_failure(worker_id)
    
    def _handle_worker_failure(self, worker_id: str):
        """Handle worker failure."""
        # Reassign tasks
        self._reassign_worker_tasks(worker_id)
        
        # Remove worker after grace period (in practice, might try to reconnect)
        # For now, just mark as failed
        if worker_id in self.workers:
            self.workers[worker_id].status = "failed"
    
    def _reassign_worker_tasks(self, worker_id: str):
        """Reassign tasks from failed worker."""
        if worker_id in self.task_assignments:
            task_ids = self.task_assignments[worker_id].copy()
            
            for task_id in task_ids:
                # Move task back to pending queue
                if task_id in self.task_queue.running_tasks:
                    task = self.task_queue.running_tasks.pop(task_id)
                    
                    # Reset task for retry
                    task.created_at = datetime.now(timezone.utc)
                    priority_item = (-task.priority, task.created_at.timestamp(), task)
                    self.task_queue.pending_tasks.put(priority_item)
                    
                    self.logger.info(f"Reassigned task {task_id} after worker {worker_id} failure")
            
            # Clear assignments
            self.task_assignments[worker_id] = []
    
    def get_worker_stats(self) -> Dict[str, Any]:
        """Get worker statistics."""
        with self._lock:
            active_workers = [w for w in self.workers.values() if w.status == "active"]
            busy_workers = [w for w in self.workers.values() if w.status == "busy"]
            failed_workers = [w for w in self.workers.values() if w.status == "failed"]
            
            total_capacity = sum(w.max_concurrent_tasks for w in self.workers.values())
            current_load = sum(w.current_tasks for w in self.workers.values())
            
            return {
                "total_workers": len(self.workers),
                "active_workers": len(active_workers),
                "busy_workers": len(busy_workers),
                "failed_workers": len(failed_workers),
                "total_capacity": total_capacity,
                "current_load": current_load,
                "utilization": current_load / max(total_capacity, 1),
                "queue_stats": self.task_queue.get_queue_stats(),
                "worker_performance": dict(self.worker_performance)
            }
    
    def get_comprehensive_status(self) -> Dict[str, Any]:
        """Get comprehensive system status."""
        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "worker_stats": self.get_worker_stats(),
            "queue_stats": self.task_queue.get_queue_stats(),
            "system_health": self._get_system_health()
        }
    
    def _get_system_health(self) -> Dict[str, Any]:
        """Get system health indicators."""
        with self._lock:
            total_workers = len(self.workers)
            healthy_workers = len([w for w in self.workers.values() if w.status in ["active", "busy"]])
            
            health_score = healthy_workers / max(total_workers, 1)
            
            return {
                "overall_health": "healthy" if health_score > 0.8 else "degraded" if health_score > 0.5 else "critical",
                "health_score": health_score,
                "healthy_workers": healthy_workers,
                "total_workers": total_workers
            }
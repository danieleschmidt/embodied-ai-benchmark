"""Advanced concurrent execution system for high-performance benchmarking."""

import asyncio
import threading
import multiprocessing as mp
from concurrent.futures import (
    ThreadPoolExecutor, ProcessPoolExecutor, Future, as_completed
)
import queue
import time
import sys
import weakref
from typing import (
    Any, Dict, List, Optional, Callable, Union, Tuple, 
    Coroutine, Awaitable, AsyncGenerator
)
from dataclasses import dataclass, field
from collections import defaultdict, deque
import logging
import signal
import os
from contextlib import contextmanager
import psutil

from .logging_config import get_logger
from .error_handling import handle_errors, ErrorCategory
from .monitoring import performance_monitor, benchmark_metrics

logger = get_logger(__name__)


@dataclass
class ExecutionConfig:
    """Configuration for concurrent execution."""
    max_workers: int = mp.cpu_count()
    use_processes: bool = False
    batch_size: int = 32
    queue_size: int = 1000
    timeout_seconds: float = 300.0
    enable_monitoring: bool = True
    enable_load_balancing: bool = True
    memory_limit_mb: int = 2048
    cpu_affinity: Optional[List[int]] = None


@dataclass
class TaskResult:
    """Result of a concurrent task execution."""
    task_id: str
    success: bool
    result: Any = None
    error: Optional[Exception] = None
    execution_time: float = 0.0
    memory_usage_mb: float = 0.0
    worker_id: Optional[str] = None


@dataclass 
class WorkerStats:
    """Statistics for a worker process/thread."""
    worker_id: str
    tasks_completed: int = 0
    tasks_failed: int = 0
    total_execution_time: float = 0.0
    average_memory_usage: float = 0.0
    current_load: float = 0.0
    last_activity: float = field(default_factory=time.time)


class LoadBalancer:
    """Intelligent load balancing for worker assignment."""
    
    def __init__(self, workers: List[str]):
        """Initialize load balancer.
        
        Args:
            workers: List of worker identifiers
        """
        self.workers = workers
        self.worker_stats = {worker_id: WorkerStats(worker_id) for worker_id in workers}
        self.task_queue_lengths = defaultdict(int)
        self._lock = threading.Lock()
    
    def get_best_worker(self) -> str:
        """Get the best worker for the next task based on current load."""
        with self._lock:
            # Calculate worker scores (lower is better)
            worker_scores = {}
            
            for worker_id, stats in self.worker_stats.items():
                # Base score from current load
                load_score = stats.current_load * 100
                
                # Queue length penalty
                queue_score = self.task_queue_lengths[worker_id] * 10
                
                # Recent activity bonus (prefer recently active workers)
                time_since_activity = time.time() - stats.last_activity
                activity_penalty = min(time_since_activity, 60) / 60 * 20
                
                # Memory usage penalty
                memory_penalty = stats.average_memory_usage / 1024 * 5  # Per GB
                
                worker_scores[worker_id] = (
                    load_score + queue_score + activity_penalty + memory_penalty
                )
            
            # Select worker with lowest score
            best_worker = min(worker_scores.keys(), key=lambda w: worker_scores[w])
            
            # Update queue length
            self.task_queue_lengths[best_worker] += 1
            
            return best_worker
    
    def update_worker_stats(self, worker_id: str, task_result: TaskResult):
        """Update worker statistics after task completion."""
        with self._lock:
            if worker_id not in self.worker_stats:
                return
            
            stats = self.worker_stats[worker_id]
            
            # Update completion counts
            if task_result.success:
                stats.tasks_completed += 1
            else:
                stats.tasks_failed += 1
            
            # Update timing
            stats.total_execution_time += task_result.execution_time
            stats.last_activity = time.time()
            
            # Update memory usage (exponential moving average)
            alpha = 0.1
            if stats.average_memory_usage == 0:
                stats.average_memory_usage = task_result.memory_usage_mb
            else:
                stats.average_memory_usage = (
                    alpha * task_result.memory_usage_mb + 
                    (1 - alpha) * stats.average_memory_usage
                )
            
            # Update load (based on queue length and recent activity)
            stats.current_load = (
                self.task_queue_lengths[worker_id] / 10 +  # Queue factor
                min(1.0, (time.time() - stats.last_activity) / 60)  # Activity factor
            )
            
            # Decrease queue length
            self.task_queue_lengths[worker_id] = max(
                0, self.task_queue_lengths[worker_id] - 1
            )
    
    def get_load_stats(self) -> Dict[str, Any]:
        """Get load balancing statistics."""
        with self._lock:
            total_tasks = sum(stats.tasks_completed + stats.tasks_failed 
                            for stats in self.worker_stats.values())
            
            if total_tasks == 0:
                return {"total_tasks": 0, "load_distribution": {}}
            
            load_distribution = {}
            for worker_id, stats in self.worker_stats.items():
                worker_tasks = stats.tasks_completed + stats.tasks_failed
                load_distribution[worker_id] = {
                    "task_percentage": (worker_tasks / total_tasks) * 100,
                    "success_rate": (stats.tasks_completed / max(1, worker_tasks)) * 100,
                    "avg_execution_time": (stats.total_execution_time / max(1, stats.tasks_completed)),
                    "current_load": stats.current_load,
                    "queue_length": self.task_queue_lengths[worker_id]
                }
            
            # Calculate load balance score (lower is better balanced)
            task_counts = [stats.tasks_completed + stats.tasks_failed 
                          for stats in self.worker_stats.values()]
            load_variance = sum((count - total_tasks/len(task_counts))**2 
                              for count in task_counts) / len(task_counts)
            balance_score = load_variance / max(1, total_tasks/len(task_counts))
            
            return {
                "total_tasks": total_tasks,
                "load_distribution": load_distribution,
                "balance_score": balance_score,
                "workers_count": len(self.workers)
            }


class AdvancedTaskManager:
    """Advanced task management with priority queues and dependencies."""
    
    def __init__(self, config: ExecutionConfig):
        """Initialize task manager.
        
        Args:
            config: Execution configuration
        """
        self.config = config
        
        # Task queues by priority (higher number = higher priority)
        self.priority_queues = {
            0: queue.Queue(maxsize=config.queue_size),  # Low priority
            1: queue.Queue(maxsize=config.queue_size),  # Normal priority  
            2: queue.Queue(maxsize=config.queue_size),  # High priority
            3: queue.PriorityQueue(maxsize=config.queue_size)  # Critical priority
        }
        
        # Task tracking
        self.pending_tasks = {}
        self.completed_tasks = {}
        self.failed_tasks = {}
        self.task_dependencies = {}
        
        # Results and monitoring
        self.results_queue = queue.Queue()
        self.task_counter = 0
        self._lock = threading.Lock()
        
        # Load balancer
        self.load_balancer = None
        
        # Worker management
        self.workers = []
        self.worker_futures = {}
        
    def initialize_workers(self, executor):
        """Initialize worker processes/threads."""
        worker_ids = [f"worker_{i}" for i in range(self.config.max_workers)]
        self.workers = worker_ids
        
        if self.config.enable_load_balancing:
            self.load_balancer = LoadBalancer(worker_ids)
        
        logger.info(f"Initialized {len(worker_ids)} workers with load balancing: {self.config.enable_load_balancing}")
    
    def submit_task(self, 
                   task_func: Callable,
                   task_args: Tuple = (),
                   task_kwargs: Optional[Dict] = None,
                   priority: int = 1,
                   dependencies: Optional[List[str]] = None,
                   timeout: Optional[float] = None) -> str:
        """Submit task for execution.
        
        Args:
            task_func: Function to execute
            task_args: Function arguments
            task_kwargs: Function keyword arguments
            priority: Task priority (0=low, 1=normal, 2=high, 3=critical)
            dependencies: List of task IDs this task depends on
            timeout: Task timeout in seconds
            
        Returns:
            Task ID
        """
        if task_kwargs is None:
            task_kwargs = {}
        
        with self._lock:
            task_id = f"task_{self.task_counter}"
            self.task_counter += 1
        
        task_info = {
            "task_id": task_id,
            "func": task_func,
            "args": task_args,
            "kwargs": task_kwargs,
            "priority": priority,
            "dependencies": dependencies or [],
            "timeout": timeout or self.config.timeout_seconds,
            "submitted_time": time.time(),
            "status": "pending"
        }
        
        # Check if dependencies are satisfied
        if self._dependencies_satisfied(dependencies):
            self._queue_task(task_info)
        else:
            # Store in pending until dependencies are met
            self.pending_tasks[task_id] = task_info
            self.task_dependencies[task_id] = set(dependencies or [])
        
        logger.debug(f"Submitted task {task_id} with priority {priority}")
        return task_id
    
    def _dependencies_satisfied(self, dependencies: Optional[List[str]]) -> bool:
        """Check if task dependencies are satisfied."""
        if not dependencies:
            return True
        
        return all(dep_id in self.completed_tasks for dep_id in dependencies)
    
    def _queue_task(self, task_info: Dict[str, Any]):
        """Queue task for execution."""
        priority = task_info["priority"]
        
        try:
            if priority == 3:  # Critical priority - use priority queue
                # Use negative time for FIFO within same priority
                self.priority_queues[3].put((0, -task_info["submitted_time"], task_info), timeout=1.0)
            else:
                self.priority_queues[priority].put(task_info, timeout=1.0)
            
            task_info["status"] = "queued"
            logger.debug(f"Queued task {task_info['task_id']} with priority {priority}")
            
        except queue.Full:
            logger.error(f"Queue full for priority {priority}, dropping task {task_info['task_id']}")
            self._mark_task_failed(task_info["task_id"], Exception("Queue overflow"))
    
    def get_next_task(self) -> Optional[Dict[str, Any]]:
        """Get next task to execute (highest priority first)."""
        # Try queues in priority order
        for priority in [3, 2, 1, 0]:
            try:
                if priority == 3:
                    _, _, task_info = self.priority_queues[3].get_nowait()
                else:
                    task_info = self.priority_queues[priority].get_nowait()
                
                task_info["status"] = "executing"
                logger.debug(f"Retrieved task {task_info['task_id']} from priority {priority} queue")
                return task_info
                
            except queue.Empty:
                continue
        
        return None
    
    def complete_task(self, task_id: str, result: TaskResult):
        """Mark task as completed and process dependencies."""
        if result.success:
            self.completed_tasks[task_id] = result
            logger.debug(f"Task {task_id} completed successfully")
        else:
            self.failed_tasks[task_id] = result
            logger.debug(f"Task {task_id} failed: {result.error}")
        
        # Update load balancer stats
        if self.load_balancer and result.worker_id:
            self.load_balancer.update_worker_stats(result.worker_id, result)
        
        # Check if any pending tasks can now be queued
        self._check_pending_dependencies()
        
        # Store result for retrieval
        self.results_queue.put(result)
    
    def _check_pending_dependencies(self):
        """Check pending tasks for satisfied dependencies."""
        satisfied_tasks = []
        
        for task_id, remaining_deps in list(self.task_dependencies.items()):
            # Remove completed dependencies
            remaining_deps.difference_update(self.completed_tasks.keys())
            
            if not remaining_deps:  # All dependencies satisfied
                satisfied_tasks.append(task_id)
        
        # Queue satisfied tasks
        for task_id in satisfied_tasks:
            if task_id in self.pending_tasks:
                task_info = self.pending_tasks.pop(task_id)
                del self.task_dependencies[task_id]
                self._queue_task(task_info)
    
    def _mark_task_failed(self, task_id: str, error: Exception):
        """Mark task as failed."""
        result = TaskResult(
            task_id=task_id,
            success=False,
            error=error,
            execution_time=0.0
        )
        self.failed_tasks[task_id] = result
        self.results_queue.put(result)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get task manager statistics."""
        stats = {
            "completed_tasks": len(self.completed_tasks),
            "failed_tasks": len(self.failed_tasks),
            "pending_tasks": len(self.pending_tasks),
            "queue_sizes": {
                priority: q.qsize() for priority, q in self.priority_queues.items()
            },
            "total_submitted": self.task_counter
        }
        
        if self.load_balancer:
            stats["load_balancing"] = self.load_balancer.get_load_stats()
        
        return stats


class ConcurrentBenchmarkExecutor:
    """High-performance concurrent executor for benchmark tasks."""
    
    def __init__(self, config: Optional[ExecutionConfig] = None):
        """Initialize concurrent executor.
        
        Args:
            config: Execution configuration
        """
        self.config = config or ExecutionConfig()
        self.task_manager = AdvancedTaskManager(self.config)
        
        self._executor = None
        self._running = False
        self._worker_threads = []
        self._monitor_thread = None
        self._stop_event = threading.Event()
        
        # Performance tracking
        self._start_time = None
        self._execution_stats = {
            "tasks_per_second": deque(maxlen=100),
            "memory_usage": deque(maxlen=100),
            "cpu_usage": deque(maxlen=100)
        }
        
        # Signal handling for graceful shutdown
        self._setup_signal_handlers()
    
    def _setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown."""
        def signal_handler(signum, frame):
            logger.info(f"Received signal {signum}, initiating graceful shutdown")
            self.shutdown(wait=True)
            sys.exit(0)
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
    
    @handle_errors(category=ErrorCategory.COMPUTATION, auto_recover=True)
    def start(self):
        """Start the concurrent execution system."""
        if self._running:
            logger.warning("Executor already running")
            return
        
        logger.info(f"Starting concurrent executor with {self.config.max_workers} workers")
        
        # Create executor
        if self.config.use_processes:
            # Configure process pool with optimal settings
            ctx = mp.get_context('spawn')  # Use spawn for better isolation
            self._executor = ProcessPoolExecutor(
                max_workers=self.config.max_workers,
                mp_context=ctx
            )
        else:
            self._executor = ThreadPoolExecutor(
                max_workers=self.config.max_workers,
                thread_name_prefix="benchmark_worker"
            )
        
        # Initialize workers
        self.task_manager.initialize_workers(self._executor)
        
        # Start worker coordination threads
        self._running = True
        self._start_time = time.time()
        self._stop_event.clear()
        
        # Start task dispatcher threads
        for i in range(min(4, self.config.max_workers)):  # Max 4 dispatchers
            dispatcher_thread = threading.Thread(
                target=self._task_dispatcher,
                name=f"dispatcher_{i}",
                daemon=True
            )
            dispatcher_thread.start()
            self._worker_threads.append(dispatcher_thread)
        
        # Start monitoring thread if enabled
        if self.config.enable_monitoring:
            self._monitor_thread = threading.Thread(
                target=self._monitoring_loop,
                name="performance_monitor",
                daemon=True
            )
            self._monitor_thread.start()
        
        logger.info("Concurrent executor started successfully")
    
    def _task_dispatcher(self):
        """Dispatcher thread that processes tasks from queues."""
        dispatcher_id = threading.current_thread().name
        logger.debug(f"Task dispatcher {dispatcher_id} started")
        
        while not self._stop_event.is_set():
            try:
                # Get next task
                task_info = self.task_manager.get_next_task()
                if task_info is None:
                    time.sleep(0.01)  # Brief pause if no tasks
                    continue
                
                # Select worker (load balancing)
                worker_id = None
                if self.task_manager.load_balancer:
                    worker_id = self.task_manager.load_balancer.get_best_worker()
                
                # Submit to executor
                future = self._executor.submit(
                    self._execute_task_safely,
                    task_info,
                    worker_id
                )
                
                # Track future
                task_id = task_info["task_id"]
                self.task_manager.worker_futures[task_id] = future
                
                # Process completion asynchronously
                future.add_done_callback(
                    lambda f, tid=task_id: self._handle_task_completion(tid, f)
                )
                
            except Exception as e:
                logger.error(f"Error in task dispatcher {dispatcher_id}: {e}")
                time.sleep(0.1)  # Pause on error
        
        logger.debug(f"Task dispatcher {dispatcher_id} stopped")
    
    def _execute_task_safely(self, task_info: Dict[str, Any], worker_id: Optional[str]) -> TaskResult:
        """Execute task with comprehensive error handling and monitoring."""
        task_id = task_info["task_id"]
        start_time = time.time()
        start_memory = 0
        
        try:
            # Get initial memory usage
            try:
                process = psutil.Process()
                start_memory = process.memory_info().rss
            except:
                pass
            
            # Set CPU affinity if configured
            if self.config.cpu_affinity and self.config.use_processes:
                try:
                    os.sched_setaffinity(0, self.config.cpu_affinity)
                except:
                    pass
            
            # Execute the task function
            func = task_info["func"]
            args = task_info["args"]
            kwargs = task_info["kwargs"]
            timeout = task_info["timeout"]
            
            # Apply timeout
            if timeout > 0:
                result = self._execute_with_timeout(func, args, kwargs, timeout)
            else:
                result = func(*args, **kwargs)
            
            # Calculate execution metrics
            execution_time = time.time() - start_time
            memory_usage = 0
            
            try:
                end_memory = psutil.Process().memory_info().rss
                memory_usage = max(0, (end_memory - start_memory) / (1024 * 1024))
            except:
                pass
            
            # Record successful execution
            benchmark_metrics.increment_counter("successful_tasks")
            
            return TaskResult(
                task_id=task_id,
                success=True,
                result=result,
                execution_time=execution_time,
                memory_usage_mb=memory_usage,
                worker_id=worker_id
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            
            # Record failed execution
            benchmark_metrics.increment_counter("failed_tasks")
            
            logger.error(f"Task {task_id} failed after {execution_time:.2f}s: {e}")
            
            return TaskResult(
                task_id=task_id,
                success=False,
                error=e,
                execution_time=execution_time,
                memory_usage_mb=0,
                worker_id=worker_id
            )
    
    def _execute_with_timeout(self, func: Callable, args: Tuple, kwargs: Dict, timeout: float) -> Any:
        """Execute function with timeout using threading."""
        result_queue = queue.Queue()
        exception_queue = queue.Queue()
        
        def target():
            try:
                result = func(*args, **kwargs)
                result_queue.put(result)
            except Exception as e:
                exception_queue.put(e)
        
        thread = threading.Thread(target=target)
        thread.daemon = True
        thread.start()
        thread.join(timeout)
        
        if thread.is_alive():
            # Timeout occurred
            logger.warning(f"Task timed out after {timeout}s")
            raise TimeoutError(f"Task execution exceeded {timeout}s timeout")
        
        # Check for exceptions
        try:
            exception = exception_queue.get_nowait()
            raise exception
        except queue.Empty:
            pass
        
        # Get result
        try:
            return result_queue.get_nowait()
        except queue.Empty:
            raise RuntimeError("Task completed without result or exception")
    
    def _handle_task_completion(self, task_id: str, future: Future):
        """Handle task completion callback."""
        try:
            result = future.result()
            self.task_manager.complete_task(task_id, result)
            
            # Clean up future reference
            if task_id in self.task_manager.worker_futures:
                del self.task_manager.worker_futures[task_id]
                
        except Exception as e:
            logger.error(f"Error handling completion for task {task_id}: {e}")
            
            # Create error result
            error_result = TaskResult(
                task_id=task_id,
                success=False,
                error=e,
                execution_time=0.0
            )
            self.task_manager.complete_task(task_id, error_result)
    
    def _monitoring_loop(self):
        """Performance monitoring loop."""
        logger.debug("Performance monitoring started")
        
        while not self._stop_event.is_set():
            try:
                # Collect performance metrics
                if hasattr(psutil, 'Process'):
                    process = psutil.Process()
                    memory_mb = process.memory_info().rss / (1024 * 1024)
                    cpu_percent = process.cpu_percent()
                    
                    self._execution_stats["memory_usage"].append(memory_mb)
                    self._execution_stats["cpu_usage"].append(cpu_percent)
                
                # Calculate tasks per second
                if self._start_time:
                    elapsed = time.time() - self._start_time
                    total_tasks = (
                        len(self.task_manager.completed_tasks) + 
                        len(self.task_manager.failed_tasks)
                    )
                    tps = total_tasks / max(elapsed, 1.0)
                    self._execution_stats["tasks_per_second"].append(tps)
                
                # Log performance summary periodically
                if len(self._execution_stats["tasks_per_second"]) % 60 == 0:  # Every 60 samples
                    self._log_performance_summary()
                
                time.sleep(1.0)  # Monitor every second
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(5.0)
        
        logger.debug("Performance monitoring stopped")
    
    def _log_performance_summary(self):
        """Log performance summary."""
        stats = self.get_performance_stats()
        
        logger.info(
            f"PERFORMANCE SUMMARY - "
            f"TPS: {stats['avg_tasks_per_second']:.2f}, "
            f"Memory: {stats['avg_memory_mb']:.1f}MB, "
            f"CPU: {stats['avg_cpu_percent']:.1f}%, "
            f"Completed: {stats['completed_tasks']}, "
            f"Failed: {stats['failed_tasks']}"
        )
    
    def submit_task(self, *args, **kwargs) -> str:
        """Submit task for execution."""
        if not self._running:
            raise RuntimeError("Executor not started")
        
        return self.task_manager.submit_task(*args, **kwargs)
    
    def submit_batch(self, 
                    tasks: List[Tuple[Callable, Tuple, Dict]],
                    priority: int = 1) -> List[str]:
        """Submit multiple tasks as a batch.
        
        Args:
            tasks: List of (func, args, kwargs) tuples
            priority: Priority for all tasks in batch
            
        Returns:
            List of task IDs
        """
        task_ids = []
        
        for func, args, kwargs in tasks:
            task_id = self.submit_task(
                task_func=func,
                task_args=args,
                task_kwargs=kwargs,
                priority=priority
            )
            task_ids.append(task_id)
        
        logger.info(f"Submitted batch of {len(tasks)} tasks with priority {priority}")
        return task_ids
    
    def get_result(self, timeout: Optional[float] = None) -> Optional[TaskResult]:
        """Get next completed task result.
        
        Args:
            timeout: Timeout in seconds (None for blocking)
            
        Returns:
            Task result or None if timeout
        """
        try:
            return self.task_manager.results_queue.get(timeout=timeout)
        except queue.Empty:
            return None
    
    def get_results(self, count: int = 1, timeout: Optional[float] = None) -> List[TaskResult]:
        """Get multiple task results.
        
        Args:
            count: Number of results to get
            timeout: Total timeout in seconds
            
        Returns:
            List of task results (may be shorter than count if timeout)
        """
        results = []
        start_time = time.time()
        
        for _ in range(count):
            remaining_timeout = None
            if timeout is not None:
                elapsed = time.time() - start_time
                remaining_timeout = max(0, timeout - elapsed)
                if remaining_timeout <= 0:
                    break
            
            result = self.get_result(timeout=remaining_timeout)
            if result is None:
                break
            
            results.append(result)
        
        return results
    
    def wait_for_completion(self, timeout: Optional[float] = None) -> bool:
        """Wait for all submitted tasks to complete.
        
        Args:
            timeout: Maximum time to wait in seconds
            
        Returns:
            True if all tasks completed, False if timeout
        """
        start_time = time.time()
        
        while True:
            stats = self.task_manager.get_stats()
            
            # Check if all tasks are completed
            pending = stats["pending_tasks"]
            queued = sum(stats["queue_sizes"].values())
            
            if pending == 0 and queued == 0:
                return True
            
            # Check timeout
            if timeout is not None:
                elapsed = time.time() - start_time
                if elapsed >= timeout:
                    return False
            
            time.sleep(0.1)
    
    def shutdown(self, wait: bool = True, timeout: float = 30.0):
        """Shutdown the executor gracefully.
        
        Args:
            wait: Whether to wait for running tasks to complete
            timeout: Maximum time to wait for shutdown
        """
        if not self._running:
            return
        
        logger.info("Shutting down concurrent executor")
        
        # Stop accepting new tasks
        self._running = False
        self._stop_event.set()
        
        # Wait for worker threads to stop
        for thread in self._worker_threads:
            thread.join(timeout=5.0)
        
        if self._monitor_thread:
            self._monitor_thread.join(timeout=2.0)
        
        # Shutdown executor
        if self._executor:
            self._executor.shutdown(wait=wait)
        
        logger.info("Concurrent executor shutdown complete")
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        stats = self.task_manager.get_stats()
        
        # Add execution statistics
        if self._execution_stats["tasks_per_second"]:
            stats["avg_tasks_per_second"] = sum(self._execution_stats["tasks_per_second"]) / len(self._execution_stats["tasks_per_second"])
        else:
            stats["avg_tasks_per_second"] = 0.0
        
        if self._execution_stats["memory_usage"]:
            stats["avg_memory_mb"] = sum(self._execution_stats["memory_usage"]) / len(self._execution_stats["memory_usage"])
            stats["peak_memory_mb"] = max(self._execution_stats["memory_usage"])
        else:
            stats["avg_memory_mb"] = 0.0
            stats["peak_memory_mb"] = 0.0
        
        if self._execution_stats["cpu_usage"]:
            stats["avg_cpu_percent"] = sum(self._execution_stats["cpu_usage"]) / len(self._execution_stats["cpu_usage"])
            stats["peak_cpu_percent"] = max(self._execution_stats["cpu_usage"])
        else:
            stats["avg_cpu_percent"] = 0.0
            stats["peak_cpu_percent"] = 0.0
        
        # Add runtime info
        if self._start_time:
            stats["uptime_seconds"] = time.time() - self._start_time
        else:
            stats["uptime_seconds"] = 0
        
        stats["is_running"] = self._running
        stats["worker_count"] = self.config.max_workers
        stats["use_processes"] = self.config.use_processes
        
        return stats
    
    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.shutdown(wait=True)


# Async execution support
class AsyncBenchmarkExecutor:
    """Asynchronous benchmark executor for I/O-bound tasks."""
    
    def __init__(self, max_concurrent: int = 100):
        """Initialize async executor.
        
        Args:
            max_concurrent: Maximum concurrent coroutines
        """
        self.max_concurrent = max_concurrent
        self._semaphore = None
        self._running_tasks = set()
        self._completed_tasks = []
        self._failed_tasks = []
    
    async def execute_async_batch(self, 
                                 coroutines: List[Coroutine],
                                 return_exceptions: bool = True) -> List[Any]:
        """Execute batch of coroutines concurrently.
        
        Args:
            coroutines: List of coroutines to execute
            return_exceptions: Whether to return exceptions as results
            
        Returns:
            List of results
        """
        self._semaphore = asyncio.Semaphore(self.max_concurrent)
        
        async def run_with_semaphore(coro):
            async with self._semaphore:
                return await coro
        
        # Execute all coroutines with semaphore limiting
        limited_coroutines = [run_with_semaphore(coro) for coro in coroutines]
        
        results = await asyncio.gather(*limited_coroutines, return_exceptions=return_exceptions)
        
        # Track results
        for result in results:
            if isinstance(result, Exception):
                self._failed_tasks.append(result)
            else:
                self._completed_tasks.append(result)
        
        return results
    
    def get_stats(self) -> Dict[str, Any]:
        """Get async executor statistics."""
        return {
            "completed_tasks": len(self._completed_tasks),
            "failed_tasks": len(self._failed_tasks),
            "max_concurrent": self.max_concurrent,
            "running_tasks": len(self._running_tasks)
        }


# Global concurrent executor instance (lazily initialized)
_global_executor = None

def get_global_executor(config: Optional[ExecutionConfig] = None) -> ConcurrentBenchmarkExecutor:
    """Get global concurrent executor instance."""
    global _global_executor
    
    if _global_executor is None:
        _global_executor = ConcurrentBenchmarkExecutor(config)
    
    return _global_executor
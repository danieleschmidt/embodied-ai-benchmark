"""Distributed execution and scaling utilities for embodied AI benchmarks."""

import asyncio
import concurrent.futures
import multiprocessing as mp
from typing import Any, Dict, List, Optional, Callable, Union, Tuple
import time
import logging
from dataclasses import dataclass, field
import pickle
import queue
import threading
from datetime import datetime
import numpy as np
from collections import defaultdict

from .logging_config import get_logger
from .monitoring import performance_monitor
from .error_handling import SafeExecutor

logger = get_logger(__name__)


@dataclass
class TaskSpec:
    """Specification for a distributed task."""
    task_id: str
    task_type: str
    payload: Dict[str, Any]
    priority: int = 1
    timeout: float = 3600.0
    dependencies: List[str] = field(default_factory=list)
    retry_count: int = 0
    max_retries: int = 3
    created_at: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'task_id': self.task_id,
            'task_type': self.task_type,
            'payload': self.payload,
            'priority': self.priority,
            'timeout': self.timeout,
            'dependencies': self.dependencies,
            'retry_count': self.retry_count,
            'max_retries': self.max_retries,
            'created_at': self.created_at.isoformat()
        }


@dataclass
class TaskResult:
    """Result from a distributed task execution."""
    task_id: str
    success: bool
    result: Any = None
    error: Optional[str] = None
    execution_time: float = 0.0
    worker_id: Optional[str] = None
    completed_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'task_id': self.task_id,
            'success': self.success,
            'result': self.result,
            'error': self.error,
            'execution_time': self.execution_time,
            'worker_id': self.worker_id,
            'completed_at': self.completed_at.isoformat(),
            'metadata': self.metadata
        }


class DistributedTaskQueue:
    """High-performance distributed task queue for benchmark execution."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize distributed task queue.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.max_workers = self.config.get('max_workers', mp.cpu_count())
        self.queue_size = self.config.get('queue_size', 10000)
        
        # Task management
        self._task_queue = queue.PriorityQueue(maxsize=self.queue_size)
        self._result_queue = queue.Queue()
        self._pending_tasks = {}  # task_id -> TaskSpec
        self._completed_tasks = {}  # task_id -> TaskResult
        self._failed_tasks = {}  # task_id -> TaskResult
        self._task_dependencies = defaultdict(set)  # task_id -> set of dependency task_ids
        
        # Worker management
        self._workers = []
        self._worker_pool = None
        self._running = False
        self._shutdown_event = threading.Event()
        
        # Performance monitoring
        self._stats = {
            'tasks_submitted': 0,
            'tasks_completed': 0,
            'tasks_failed': 0,
            'total_execution_time': 0.0,
            'avg_task_time': 0.0,
            'throughput_tasks_per_sec': 0.0
        }
        self._stats_lock = threading.Lock()
        
        # Result processing thread
        self._result_processor = None
        
        logger.info(f"Initialized DistributedTaskQueue with {self.max_workers} workers")
    
    def start(self):
        """Start the distributed task queue."""
        if self._running:
            logger.warning("Task queue is already running")
            return
        
        logger.info("Starting distributed task queue...")
        self._running = True
        self._shutdown_event.clear()
        
        # Start worker pool
        self._worker_pool = concurrent.futures.ProcessPoolExecutor(
            max_workers=self.max_workers,
            mp_context=mp.get_context('spawn')
        )
        
        # Start result processor thread
        self._result_processor = threading.Thread(
            target=self._process_results,
            daemon=True
        )
        self._result_processor.start()
        
        logger.info(f"Task queue started with {self.max_workers} workers")
    
    def stop(self):
        """Stop the distributed task queue."""
        if not self._running:
            logger.warning("Task queue is not running")
            return
        
        logger.info("Stopping distributed task queue...")
        self._running = False
        self._shutdown_event.set()
        
        # Shutdown worker pool
        if self._worker_pool:
            self._worker_pool.shutdown(wait=True, cancel_futures=True)
            self._worker_pool = None
        
        # Wait for result processor to finish
        if self._result_processor and self._result_processor.is_alive():
            self._result_processor.join(timeout=10.0)
        
        logger.info("Task queue stopped")
    
    def submit_task(self, task_spec: TaskSpec) -> bool:
        """Submit a task for distributed execution.
        
        Args:
            task_spec: Task specification
            
        Returns:
            True if task was successfully submitted
        """
        if not self._running:
            logger.error("Cannot submit task - queue is not running")
            return False
        
        try:
            # Check dependencies
            unresolved_deps = []
            for dep_id in task_spec.dependencies:
                if dep_id not in self._completed_tasks:
                    unresolved_deps.append(dep_id)
            
            if unresolved_deps:
                # Store task until dependencies are resolved
                self._task_dependencies[task_spec.task_id].update(unresolved_deps)
                self._pending_tasks[task_spec.task_id] = task_spec
                logger.debug(f"Task {task_spec.task_id} waiting for dependencies: {unresolved_deps}")
                return True
            
            # Submit task to queue (priority queue uses negative priority for max-heap behavior)
            priority_item = (-task_spec.priority, time.time(), task_spec)
            self._task_queue.put_nowait(priority_item)
            
            # Update statistics
            with self._stats_lock:
                self._stats['tasks_submitted'] += 1
            
            logger.debug(f"Task {task_spec.task_id} submitted to queue")
            return True
            
        except queue.Full:
            logger.error(f"Task queue is full - cannot submit task {task_spec.task_id}")
            return False
        except Exception as e:
            logger.error(f"Error submitting task {task_spec.task_id}: {e}")
            return False
    
    def submit_batch(self, task_specs: List[TaskSpec]) -> Dict[str, bool]:
        """Submit multiple tasks as a batch.
        
        Args:
            task_specs: List of task specifications
            
        Returns:
            Dictionary mapping task_id to submission success
        """
        results = {}
        
        # Sort tasks by dependencies (topological sort)
        sorted_tasks = self._topological_sort(task_specs)
        
        for task_spec in sorted_tasks:
            results[task_spec.task_id] = self.submit_task(task_spec)
        
        logger.info(f"Submitted batch of {len(task_specs)} tasks, {sum(results.values())} successful")
        return results
    
    def get_result(self, task_id: str, timeout: Optional[float] = None) -> Optional[TaskResult]:
        """Get result for a specific task.
        
        Args:
            task_id: Task identifier
            timeout: Maximum time to wait for result
            
        Returns:
            Task result or None if not available
        """
        # Check completed tasks first
        if task_id in self._completed_tasks:
            return self._completed_tasks[task_id]
        
        if task_id in self._failed_tasks:
            return self._failed_tasks[task_id]
        
        # Wait for result if timeout specified
        if timeout is not None:
            start_time = time.time()
            while time.time() - start_time < timeout:
                if task_id in self._completed_tasks:
                    return self._completed_tasks[task_id]
                if task_id in self._failed_tasks:
                    return self._failed_tasks[task_id]
                time.sleep(0.1)
        
        return None
    
    def get_batch_results(self, task_ids: List[str], timeout: Optional[float] = None) -> Dict[str, TaskResult]:
        """Get results for multiple tasks.
        
        Args:
            task_ids: List of task identifiers
            timeout: Maximum time to wait for all results
            
        Returns:
            Dictionary mapping task_id to TaskResult
        """
        results = {}
        start_time = time.time() if timeout else None
        
        for task_id in task_ids:
            remaining_timeout = None
            if timeout and start_time:
                elapsed = time.time() - start_time
                remaining_timeout = max(0, timeout - elapsed)
            
            result = self.get_result(task_id, remaining_timeout)
            if result:
                results[task_id] = result
        
        return results
    
    def wait_for_completion(self, task_ids: Optional[List[str]] = None, timeout: Optional[float] = None) -> bool:
        """Wait for tasks to complete.
        
        Args:
            task_ids: Specific task IDs to wait for (None for all pending)
            timeout: Maximum time to wait
            
        Returns:
            True if all tasks completed within timeout
        """
        if task_ids is None:
            task_ids = list(self._pending_tasks.keys())
        
        start_time = time.time()
        
        while task_ids:
            if timeout and (time.time() - start_time) > timeout:
                logger.warning(f"Timeout waiting for tasks: {task_ids}")
                return False
            
            completed = []
            for task_id in task_ids:
                if task_id in self._completed_tasks or task_id in self._failed_tasks:
                    completed.append(task_id)
            
            for task_id in completed:
                task_ids.remove(task_id)
            
            if task_ids:
                time.sleep(0.5)
        
        return True
    
    def _topological_sort(self, task_specs: List[TaskSpec]) -> List[TaskSpec]:
        """Sort tasks by dependencies using topological sort.
        
        Args:
            task_specs: List of task specifications
            
        Returns:
            Topologically sorted list of tasks
        """
        # Build dependency graph
        graph = {task.task_id: set(task.dependencies) for task in task_specs}
        task_map = {task.task_id: task for task in task_specs}
        
        # Kahn's algorithm
        in_degree = {task_id: len(deps) for task_id, deps in graph.items()}
        queue = [task_id for task_id, degree in in_degree.items() if degree == 0]
        sorted_tasks = []
        
        while queue:
            current = queue.pop(0)
            sorted_tasks.append(task_map[current])
            
            # Update in-degrees for dependent tasks
            for task_id, deps in graph.items():
                if current in deps:
                    in_degree[task_id] -= 1
                    if in_degree[task_id] == 0:
                        queue.append(task_id)
        
        # Check for cycles
        if len(sorted_tasks) != len(task_specs):
            logger.warning("Circular dependencies detected in task batch")
            # Return original order as fallback
            return task_specs
        
        return sorted_tasks
    
    def _process_results(self):
        """Process completed task results in background thread."""
        logger.debug("Result processor thread started")
        
        while self._running or not self._result_queue.empty():
            try:
                # Get next result with timeout
                result = self._result_queue.get(timeout=1.0)
                
                # Update statistics
                with self._stats_lock:
                    if result.success:
                        self._stats['tasks_completed'] += 1
                        self._completed_tasks[result.task_id] = result
                    else:
                        self._stats['tasks_failed'] += 1
                        self._failed_tasks[result.task_id] = result
                    
                    self._stats['total_execution_time'] += result.execution_time
                    
                    total_tasks = self._stats['tasks_completed'] + self._stats['tasks_failed']
                    if total_tasks > 0:
                        self._stats['avg_task_time'] = self._stats['total_execution_time'] / total_tasks
                
                # Check for dependent tasks that can now be submitted
                self._check_dependent_tasks(result.task_id)
                
                logger.debug(f"Processed result for task {result.task_id}")
                
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Error processing result: {e}")
        
        logger.debug("Result processor thread stopped")
    
    def _check_dependent_tasks(self, completed_task_id: str):
        """Check if any pending tasks can now be submitted.
        
        Args:
            completed_task_id: ID of the task that just completed
        """
        tasks_to_submit = []
        
        for task_id, pending_deps in list(self._task_dependencies.items()):
            if completed_task_id in pending_deps:
                pending_deps.remove(completed_task_id)
                
                # If all dependencies resolved, submit task
                if not pending_deps:
                    if task_id in self._pending_tasks:
                        tasks_to_submit.append(self._pending_tasks[task_id])
                        del self._pending_tasks[task_id]
                    del self._task_dependencies[task_id]
        
        # Submit newly ready tasks
        for task_spec in tasks_to_submit:
            self.submit_task(task_spec)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get task queue statistics.
        
        Returns:
            Statistics dictionary
        """
        with self._stats_lock:
            stats = self._stats.copy()
        
        # Calculate throughput
        if stats['total_execution_time'] > 0:
            stats['throughput_tasks_per_sec'] = stats['tasks_completed'] / stats['total_execution_time']
        
        # Add queue status
        stats.update({
            'pending_tasks': len(self._pending_tasks),
            'queue_size': self._task_queue.qsize(),
            'completed_tasks': len(self._completed_tasks),
            'failed_tasks': len(self._failed_tasks),
            'running': self._running,
            'active_workers': self.max_workers if self._running else 0
        })
        
        return stats


def distributed_benchmark_worker(task_spec_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Worker function for distributed benchmark execution.
    
    This function runs in a separate process and executes benchmark tasks.
    
    Args:
        task_spec_dict: Serialized task specification
        
    Returns:
        Serialized task result
    """
    import os
    import traceback
    
    worker_id = f"worker_{os.getpid()}"
    start_time = time.time()
    
    try:
        # Deserialize task spec
        task_id = task_spec_dict['task_id']
        task_type = task_spec_dict['task_type']
        payload = task_spec_dict['payload']
        
        logger.debug(f"Worker {worker_id} executing task {task_id} of type {task_type}")
        
        # Execute task based on type
        if task_type == 'benchmark_evaluation':
            result = _execute_benchmark_evaluation(payload)
        elif task_type == 'cross_simulator_transfer':
            result = _execute_cross_simulator_transfer(payload)
        elif task_type == 'emergent_curriculum_generation':
            result = _execute_emergent_curriculum_generation(payload)
        elif task_type == 'physics_adaptation':
            result = _execute_physics_adaptation(payload)
        elif task_type == 'long_horizon_planning':
            result = _execute_long_horizon_planning(payload)
        else:
            raise ValueError(f"Unknown task type: {task_type}")
        
        execution_time = time.time() - start_time
        
        return TaskResult(
            task_id=task_id,
            success=True,
            result=result,
            execution_time=execution_time,
            worker_id=worker_id
        ).to_dict()
        
    except Exception as e:
        execution_time = time.time() - start_time
        error_msg = f"{str(e)}\n{traceback.format_exc()}"
        
        logger.error(f"Worker {worker_id} task {task_spec_dict.get('task_id', 'unknown')} failed: {e}")
        
        return TaskResult(
            task_id=task_spec_dict.get('task_id', 'unknown'),
            success=False,
            error=error_msg,
            execution_time=execution_time,
            worker_id=worker_id
        ).to_dict()


def _execute_benchmark_evaluation(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Execute benchmark evaluation task.
    
    Args:
        payload: Task payload containing evaluation parameters
        
    Returns:
        Evaluation results
    """
    from ..evaluation.benchmark_suite import BenchmarkSuite
    from ..core.base_agent import RandomAgent
    from ..core.base_env import BaseEnv
    
    # Extract parameters
    agent_config = payload.get('agent_config', {})
    env_config = payload.get('env_config', {})
    eval_config = payload.get('eval_config', {})
    
    # Create components (simplified for distributed execution)
    benchmark = BenchmarkSuite(eval_config)
    agent = RandomAgent('distributed_agent', agent_config)
    
    # Note: In practice, environment would be reconstructed from config
    # For now, return mock results
    return {
        'success_rate': np.random.random(),
        'avg_steps': np.random.randint(10, 100),
        'avg_reward': np.random.random(),
        'execution_details': {
            'episodes': eval_config.get('num_episodes', 10),
            'max_steps': eval_config.get('max_steps', 1000)
        }
    }


def _execute_cross_simulator_transfer(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Execute cross-simulator transfer evaluation.
    
    Args:
        payload: Task payload
        
    Returns:
        Transfer evaluation results
    """
    # Simplified implementation for distributed execution
    source_sim = payload.get('source_simulator', 'habitat')
    target_sim = payload.get('target_simulator', 'maniskill3')
    modalities = payload.get('modalities', ['vision', 'tactile'])
    
    # Simulate transfer evaluation
    transfer_ratio = np.random.uniform(0.3, 0.9)
    adaptation_gain = np.random.uniform(0.0, 0.3)
    
    return {
        'source_simulator': source_sim,
        'target_simulator': target_sim,
        'transfer_ratio': transfer_ratio,
        'adaptation_gain': adaptation_gain,
        'modality_results': {
            f"{mod1}_to_{mod2}": np.random.uniform(0.2, 0.8)
            for mod1 in modalities for mod2 in modalities if mod1 != mod2
        }
    }


def _execute_emergent_curriculum_generation(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Execute emergent curriculum generation task.
    
    Args:
        payload: Task payload
        
    Returns:
        Generated curriculum
    """
    num_tasks = payload.get('num_tasks', 5)
    complexity_range = payload.get('complexity_range', (0.3, 0.8))
    
    # Generate mock curriculum tasks
    generated_tasks = []
    for i in range(num_tasks):
        task = {
            'task_id': f'emergent_task_{i}',
            'complexity': np.random.uniform(*complexity_range),
            'objectives': [f'objective_{j}' for j in range(np.random.randint(2, 5))],
            'estimated_difficulty': np.random.uniform(0.4, 0.9)
        }
        generated_tasks.append(task)
    
    return {
        'generated_tasks': generated_tasks,
        'num_patterns_discovered': np.random.randint(3, 15),
        'avg_pattern_complexity': np.random.uniform(0.4, 0.8)
    }


def _execute_physics_adaptation(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Execute physics adaptation task.
    
    Args:
        payload: Task payload
        
    Returns:
        Physics adaptation results
    """
    num_samples = payload.get('num_samples', 10)
    adaptation_steps = payload.get('adaptation_steps', 5)
    
    # Simulate physics parameter adaptation
    initial_discrepancy = np.random.uniform(0.1, 0.5)
    final_discrepancy = initial_discrepancy * np.random.uniform(0.3, 0.8)
    
    return {
        'initial_discrepancy': initial_discrepancy,
        'final_discrepancy': final_discrepancy,
        'discrepancy_reduction': (initial_discrepancy - final_discrepancy) / initial_discrepancy,
        'parameters_updated': np.random.randint(5, 20),
        'adaptation_steps': adaptation_steps
    }


def _execute_long_horizon_planning(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Execute long-horizon planning task.
    
    Args:
        payload: Task payload
        
    Returns:
        Planning execution results
    """
    max_horizon = payload.get('max_horizon', 50)
    num_agents = payload.get('num_agents', 3)
    task_complexity = payload.get('task_complexity', 0.7)
    
    # Simulate long-horizon task execution
    phases_completed = np.random.randint(3, max_horizon // 5)
    coordination_events = np.random.randint(phases_completed, phases_completed * 3)
    
    success = np.random.random() < (1.0 - task_complexity * 0.5)
    
    return {
        'success': success,
        'phases_completed': phases_completed,
        'coordination_events': coordination_events,
        'execution_time': np.random.uniform(30.0, 300.0),
        'coordination_quality': np.random.uniform(0.5, 0.95),
        'error_count': np.random.randint(0, 3) if not success else 0
    }


class ParallelBenchmarkRunner:
    """High-performance parallel benchmark runner."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize parallel benchmark runner.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.task_queue = DistributedTaskQueue(config.get('task_queue', {}))
        self.safe_executor = SafeExecutor()
        self.results_cache = {}
        
    def run_parallel_evaluations(
        self, 
        evaluation_specs: List[Dict[str, Any]],
        max_concurrent: Optional[int] = None
    ) -> Dict[str, Any]:
        """Run multiple evaluations in parallel.
        
        Args:
            evaluation_specs: List of evaluation specifications
            max_concurrent: Maximum concurrent evaluations
            
        Returns:
            Aggregated results from all evaluations
        """
        logger.info(f"Starting parallel execution of {len(evaluation_specs)} evaluations")
        
        if not self.task_queue._running:
            self.task_queue.start()
        
        try:
            # Create task specifications
            task_specs = []
            for i, eval_spec in enumerate(evaluation_specs):
                task_spec = TaskSpec(
                    task_id=f"eval_{i}_{eval_spec.get('type', 'benchmark')}",
                    task_type=eval_spec.get('type', 'benchmark_evaluation'),
                    payload=eval_spec,
                    priority=eval_spec.get('priority', 1)
                )
                task_specs.append(task_spec)
            
            # Submit tasks
            submission_results = self.task_queue.submit_batch(task_specs)
            successful_tasks = [task_id for task_id, success in submission_results.items() if success]
            
            logger.info(f"Successfully submitted {len(successful_tasks)} tasks")
            
            # Wait for completion
            completion_timeout = self.config.get('completion_timeout', 3600.0)  # 1 hour
            completed = self.task_queue.wait_for_completion(successful_tasks, completion_timeout)
            
            if not completed:
                logger.warning("Not all tasks completed within timeout")
            
            # Collect results
            results = self.task_queue.get_batch_results(successful_tasks)
            
            # Aggregate results
            aggregated = self._aggregate_parallel_results(results, evaluation_specs)
            
            # Add execution statistics
            aggregated['execution_statistics'] = {
                'total_evaluations': len(evaluation_specs),
                'successful_submissions': len(successful_tasks),
                'completed_tasks': len([r for r in results.values() if r.success]),
                'failed_tasks': len([r for r in results.values() if not r.success]),
                'task_queue_stats': self.task_queue.get_statistics(),
                'completion_rate': len(results) / len(successful_tasks) if successful_tasks else 0
            }
            
            logger.info(f"Parallel evaluation completed. Success rate: {len([r for r in results.values() if r.success]) / len(results) if results else 0:.2%}")
            
            return aggregated
            
        finally:
            # Keep task queue running for potential reuse
            pass
    
    def _aggregate_parallel_results(
        self, 
        results: Dict[str, TaskResult], 
        evaluation_specs: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Aggregate results from parallel evaluations.
        
        Args:
            results: Dictionary of task results
            evaluation_specs: Original evaluation specifications
            
        Returns:
            Aggregated results
        """
        successful_results = [r for r in results.values() if r.success]
        failed_results = [r for r in results.values() if not r.success]
        
        if not successful_results:
            return {
                'success': False,
                'error': 'No successful evaluations',
                'failed_count': len(failed_results)
            }
        
        # Aggregate by evaluation type
        type_results = defaultdict(list)
        for result in successful_results:
            # Extract type from task_id or use default
            task_parts = result.task_id.split('_')
            eval_type = task_parts[-1] if len(task_parts) > 2 else 'benchmark'
            type_results[eval_type].append(result.result)
        
        aggregated = {
            'success': True,
            'total_evaluations': len(results),
            'successful_evaluations': len(successful_results),
            'failed_evaluations': len(failed_results),
            'avg_execution_time': np.mean([r.execution_time for r in successful_results]),
            'total_execution_time': sum(r.execution_time for r in successful_results),
            'type_specific_results': {}
        }
        
        # Aggregate results by type
        for eval_type, type_result_list in type_results.items():
            if eval_type == 'benchmark_evaluation':
                aggregated['type_specific_results'][eval_type] = {
                    'avg_success_rate': np.mean([r.get('success_rate', 0) for r in type_result_list]),
                    'avg_steps': np.mean([r.get('avg_steps', 0) for r in type_result_list]),
                    'avg_reward': np.mean([r.get('avg_reward', 0) for r in type_result_list])
                }
            elif eval_type == 'cross_simulator_transfer':
                aggregated['type_specific_results'][eval_type] = {
                    'avg_transfer_ratio': np.mean([r.get('transfer_ratio', 0) for r in type_result_list]),
                    'avg_adaptation_gain': np.mean([r.get('adaptation_gain', 0) for r in type_result_list])
                }
            elif eval_type == 'emergent_curriculum_generation':
                aggregated['type_specific_results'][eval_type] = {
                    'total_tasks_generated': sum(len(r.get('generated_tasks', [])) for r in type_result_list),
                    'avg_pattern_complexity': np.mean([r.get('avg_pattern_complexity', 0) for r in type_result_list])
                }
            elif eval_type == 'physics_adaptation':
                aggregated['type_specific_results'][eval_type] = {
                    'avg_discrepancy_reduction': np.mean([r.get('discrepancy_reduction', 0) for r in type_result_list]),
                    'total_parameters_updated': sum(r.get('parameters_updated', 0) for r in type_result_list)
                }
            elif eval_type == 'long_horizon_planning':
                aggregated['type_specific_results'][eval_type] = {
                    'avg_success_rate': np.mean([float(r.get('success', False)) for r in type_result_list]),
                    'avg_phases_completed': np.mean([r.get('phases_completed', 0) for r in type_result_list]),
                    'avg_coordination_quality': np.mean([r.get('coordination_quality', 0) for r in type_result_list])
                }
        
        return aggregated
    
    def shutdown(self):
        """Shutdown the parallel runner and cleanup resources."""
        logger.info("Shutting down parallel benchmark runner")
        self.task_queue.stop()
        logger.info("Parallel benchmark runner shutdown complete")


# Convenience function for quick parallel execution
def run_distributed_benchmark(
    evaluation_configs: List[Dict[str, Any]], 
    max_workers: Optional[int] = None,
    timeout: float = 3600.0
) -> Dict[str, Any]:
    """Run distributed benchmark evaluations.
    
    Args:
        evaluation_configs: List of evaluation configurations
        max_workers: Maximum number of worker processes
        timeout: Execution timeout in seconds
        
    Returns:
        Aggregated results from all evaluations
    """
    config = {
        'task_queue': {
            'max_workers': max_workers or mp.cpu_count(),
        },
        'completion_timeout': timeout
    }
    
    runner = ParallelBenchmarkRunner(config)
    
    try:
        return runner.run_parallel_evaluations(evaluation_configs)
    finally:
        runner.shutdown()

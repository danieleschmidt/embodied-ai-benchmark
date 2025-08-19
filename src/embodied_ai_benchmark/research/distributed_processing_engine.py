"""Distributed Processing Engine for Scalable Research Execution."""

import asyncio
import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
import time
import json
import queue
import threading
from typing import Dict, List, Optional, Tuple, Any, Callable, Union
from dataclasses import dataclass, field, asdict
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import multiprocessing
from collections import defaultdict, deque
import redis
import pickle
import hashlib
import socket
import psutil

from ..utils.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class ClusterConfig:
    """Configuration for distributed cluster."""
    master_addr: str = "localhost"
    master_port: int = 12355
    world_size: int = 1
    rank: int = 0
    backend: str = "nccl"  # nccl for GPU, gloo for CPU
    max_workers_per_node: int = 8
    enable_mixed_precision: bool = True
    enable_gradient_compression: bool = True
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 0


@dataclass
class TaskConfig:
    """Configuration for distributed task."""
    task_id: str
    task_type: str
    priority: int = 1
    timeout_seconds: float = 3600.0
    memory_limit_mb: int = 4096
    gpu_required: bool = False
    min_workers: int = 1
    max_workers: int = 8
    retry_count: int = 3
    checkpoint_interval: int = 100


@dataclass
class WorkerStatus:
    """Status of a distributed worker."""
    worker_id: str
    node_id: str
    status: str  # idle, busy, error, offline
    current_task: Optional[str] = None
    cpu_usage: float = 0.0
    memory_usage_mb: float = 0.0
    gpu_usage: float = 0.0
    last_heartbeat: float = field(default_factory=time.time)
    capabilities: List[str] = field(default_factory=list)


@dataclass
class TaskResult:
    """Result from distributed task execution."""
    task_id: str
    worker_id: str
    success: bool
    execution_time: float
    result_data: Any = None
    error_message: Optional[str] = None
    metrics: Dict[str, float] = field(default_factory=dict)
    checkpoint_path: Optional[str] = None


class RedisTaskQueue:
    """Redis-based distributed task queue."""
    
    def __init__(self, config: ClusterConfig):
        self.config = config
        self.redis_client = redis.Redis(
            host=config.redis_host,
            port=config.redis_port,
            db=config.redis_db,
            decode_responses=False
        )
        
        # Queue names
        self.pending_queue = "tasks:pending"
        self.processing_queue = "tasks:processing"
        self.completed_queue = "tasks:completed"
        self.failed_queue = "tasks:failed"
        
        # Heartbeat tracking
        self.worker_heartbeats = "workers:heartbeats"
        
    def submit_task(self, task_config: TaskConfig, task_data: Any) -> bool:
        """Submit task to distributed queue."""
        try:
            task_package = {
                'config': asdict(task_config),
                'data': task_data,
                'submitted_at': time.time()
            }
            
            serialized_task = pickle.dumps(task_package)
            
            # Add to pending queue with priority
            self.redis_client.zadd(
                self.pending_queue,
                {serialized_task: task_config.priority}
            )
            
            logger.info(f"Task submitted: {task_config.task_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to submit task {task_config.task_id}: {e}")
            return False
    
    def get_next_task(self, worker_capabilities: List[str]) -> Optional[Tuple[TaskConfig, Any]]:
        """Get next task from queue matching worker capabilities."""
        try:
            # Get highest priority task
            result = self.redis_client.zrevrange(
                self.pending_queue, 0, 0, withscores=True
            )
            
            if not result:
                return None
            
            task_data, priority = result[0]
            task_package = pickle.loads(task_data)
            
            task_config = TaskConfig(**task_package['config'])
            
            # Check if worker can handle this task
            if task_config.gpu_required and 'gpu' not in worker_capabilities:
                return None
            
            # Move task to processing queue
            pipe = self.redis_client.pipeline()
            pipe.zrem(self.pending_queue, task_data)
            pipe.hset(self.processing_queue, task_config.task_id, task_data)
            pipe.execute()
            
            return task_config, task_package['data']
            
        except Exception as e:
            logger.error(f"Failed to get next task: {e}")
            return None
    
    def complete_task(self, task_id: str, result: TaskResult) -> bool:
        """Mark task as completed."""
        try:
            pipe = self.redis_client.pipeline()
            pipe.hdel(self.processing_queue, task_id)
            pipe.hset(self.completed_queue, task_id, pickle.dumps(result))
            pipe.execute()
            
            logger.info(f"Task completed: {task_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to complete task {task_id}: {e}")
            return False
    
    def fail_task(self, task_id: str, result: TaskResult) -> bool:
        """Mark task as failed."""
        try:
            pipe = self.redis_client.pipeline()
            pipe.hdel(self.processing_queue, task_id)
            pipe.hset(self.failed_queue, task_id, pickle.dumps(result))
            pipe.execute()
            
            logger.error(f"Task failed: {task_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to mark task as failed {task_id}: {e}")
            return False
    
    def get_queue_stats(self) -> Dict[str, int]:
        """Get queue statistics."""
        try:
            return {
                'pending': self.redis_client.zcard(self.pending_queue),
                'processing': self.redis_client.hlen(self.processing_queue),
                'completed': self.redis_client.hlen(self.completed_queue),
                'failed': self.redis_client.hlen(self.failed_queue)
            }
        except Exception as e:
            logger.error(f"Failed to get queue stats: {e}")
            return {'pending': 0, 'processing': 0, 'completed': 0, 'failed': 0}
    
    def update_worker_heartbeat(self, worker_status: WorkerStatus):
        """Update worker heartbeat."""
        try:
            self.redis_client.hset(
                self.worker_heartbeats,
                worker_status.worker_id,
                pickle.dumps(worker_status)
            )
        except Exception as e:
            logger.error(f"Failed to update worker heartbeat: {e}")
    
    def get_worker_statuses(self) -> List[WorkerStatus]:
        """Get all worker statuses."""
        try:
            worker_data = self.redis_client.hgetall(self.worker_heartbeats)
            workers = []
            
            current_time = time.time()
            
            for worker_id, data in worker_data.items():
                worker_status = pickle.loads(data)
                
                # Check if worker is still alive (heartbeat within last 30 seconds)
                if current_time - worker_status.last_heartbeat > 30:
                    worker_status.status = "offline"
                
                workers.append(worker_status)
            
            return workers
            
        except Exception as e:
            logger.error(f"Failed to get worker statuses: {e}")
            return []


class DistributedWorker:
    """Distributed worker for processing tasks."""
    
    def __init__(self, config: ClusterConfig, worker_id: str):
        self.config = config
        self.worker_id = worker_id
        self.node_id = socket.gethostname()
        self.task_queue = RedisTaskQueue(config)
        self.running = False
        self.current_task = None
        
        # Worker capabilities
        self.capabilities = ['cpu']
        if torch.cuda.is_available():
            self.capabilities.append('gpu')
        
        # Performance monitoring
        self.task_history = deque(maxlen=100)
        self.performance_metrics = defaultdict(list)
        
        # Resource monitoring
        self.resource_monitor_thread = None
        self.status = WorkerStatus(
            worker_id=worker_id,
            node_id=self.node_id,
            status="idle",
            capabilities=self.capabilities
        )
    
    def start(self):
        """Start worker."""
        self.running = True
        
        # Start resource monitoring
        self.resource_monitor_thread = threading.Thread(
            target=self._monitor_resources, daemon=True
        )
        self.resource_monitor_thread.start()
        
        logger.info(f"Worker {self.worker_id} started on node {self.node_id}")
        
        # Main work loop
        while self.running:
            try:
                self._work_cycle()
                time.sleep(1)  # Small delay between checks
                
            except KeyboardInterrupt:
                logger.info(f"Worker {self.worker_id} received shutdown signal")
                break
            except Exception as e:
                logger.error(f"Worker {self.worker_id} error: {e}")
                time.sleep(5)  # Wait before retrying
        
        self.stop()
    
    def stop(self):
        """Stop worker."""
        self.running = False
        self.status.status = "offline"
        self.task_queue.update_worker_heartbeat(self.status)
        
        logger.info(f"Worker {self.worker_id} stopped")
    
    def _work_cycle(self):
        """Single work cycle."""
        # Update status
        self.status.last_heartbeat = time.time()
        self.task_queue.update_worker_heartbeat(self.status)
        
        # Get next task
        task_info = self.task_queue.get_next_task(self.capabilities)
        
        if task_info is None:
            self.status.status = "idle"
            return
        
        task_config, task_data = task_info
        
        logger.info(f"Worker {self.worker_id} processing task {task_config.task_id}")
        
        # Execute task
        self.current_task = task_config.task_id
        self.status.status = "busy"
        self.status.current_task = task_config.task_id
        
        result = self._execute_task(task_config, task_data)
        
        # Update task queue
        if result.success:
            self.task_queue.complete_task(task_config.task_id, result)
        else:
            self.task_queue.fail_task(task_config.task_id, result)
        
        # Update worker status
        self.current_task = None
        self.status.status = "idle"
        self.status.current_task = None
        
        # Store task history
        self.task_history.append({
            'task_id': task_config.task_id,
            'execution_time': result.execution_time,
            'success': result.success,
            'timestamp': time.time()
        })
    
    def _execute_task(self, task_config: TaskConfig, task_data: Any) -> TaskResult:
        """Execute a single task."""
        start_time = time.time()
        
        try:
            # Set resource limits
            self._set_resource_limits(task_config)
            
            # Execute task based on type
            if task_config.task_type == "attention_fusion":
                result_data = self._execute_attention_fusion(task_data)
            elif task_config.task_type == "quantum_planning":
                result_data = self._execute_quantum_planning(task_data)
            elif task_config.task_type == "swarm_coordination":
                result_data = self._execute_swarm_coordination(task_data)
            elif task_config.task_type == "validation_test":
                result_data = self._execute_validation_test(task_data)
            else:
                raise ValueError(f"Unknown task type: {task_config.task_type}")
            
            execution_time = time.time() - start_time
            
            return TaskResult(
                task_id=task_config.task_id,
                worker_id=self.worker_id,
                success=True,
                execution_time=execution_time,
                result_data=result_data,
                metrics=self._get_performance_metrics()
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            
            logger.error(f"Task {task_config.task_id} failed: {e}")
            
            return TaskResult(
                task_id=task_config.task_id,
                worker_id=self.worker_id,
                success=False,
                execution_time=execution_time,
                error_message=str(e),
                metrics=self._get_performance_metrics()
            )
    
    def _execute_attention_fusion(self, task_data: Dict[str, Any]) -> Any:
        """Execute dynamic attention fusion task."""
        from .dynamic_attention_fusion import create_dynamic_attention_fusion, benchmark_attention_fusion
        
        config = task_data.get('config')
        test_inputs = task_data.get('test_inputs')
        
        # Create model
        model = create_dynamic_attention_fusion(config)
        
        # Run benchmark if test inputs provided
        if test_inputs:
            results = benchmark_attention_fusion(model, num_trials=test_inputs.get('num_trials', 10))
            return results
        else:
            # Run inference
            modality_inputs = task_data.get('modality_inputs')
            return model(modality_inputs)
    
    def _execute_quantum_planning(self, task_data: Dict[str, Any]) -> Any:
        """Execute quantum planning task."""
        from .quantum_enhanced_planning import create_quantum_planner, benchmark_quantum_planning
        
        config = task_data.get('config')
        test_inputs = task_data.get('test_inputs')
        
        # Create planner
        planner = create_quantum_planner(config)
        
        # Run benchmark if test inputs provided
        if test_inputs:
            results = benchmark_quantum_planning(planner, num_trials=test_inputs.get('num_trials', 10))
            return results
        else:
            # Run planning
            state = task_data.get('state')
            goal = task_data.get('goal')
            return planner(state, goal)
    
    def _execute_swarm_coordination(self, task_data: Dict[str, Any]) -> Any:
        """Execute swarm coordination task."""
        from .emergent_swarm_coordination import create_swarm_coordination_engine, benchmark_swarm_coordination
        
        config = task_data.get('config')
        test_inputs = task_data.get('test_inputs')
        
        # Create engine
        engine = create_swarm_coordination_engine(config)
        
        # Run benchmark if test inputs provided
        if test_inputs:
            results = benchmark_swarm_coordination(
                engine,
                num_agents=test_inputs.get('num_agents', 10),
                num_timesteps=test_inputs.get('num_timesteps', 100)
            )
            return results
        else:
            # Run coordination
            agent_states = task_data.get('agent_states')
            task_context = task_data.get('task_context')
            return engine.coordinate_swarm(agent_states, task_context)
    
    def _execute_validation_test(self, task_data: Dict[str, Any]) -> Any:
        """Execute validation test task."""
        from .robust_validation_framework import create_robust_validation_framework
        
        config = task_data.get('config')
        component_func = task_data.get('component_func')
        test_suite = task_data.get('test_suite')
        
        # Create validation framework
        framework = create_robust_validation_framework(config)
        
        # Run validation
        results = framework.validate_component(
            "distributed_component",
            component_func,
            test_suite
        )
        
        return results
    
    def _set_resource_limits(self, task_config: TaskConfig):
        """Set resource limits for task execution."""
        try:
            # Set memory limit (Linux only)
            if hasattr(psutil.Process(), 'memory_limit'):
                process = psutil.Process()
                process.memory_limit(task_config.memory_limit_mb * 1024 * 1024)
        except Exception as e:
            logger.warning(f"Failed to set memory limit: {e}")
    
    def _get_performance_metrics(self) -> Dict[str, float]:
        """Get current performance metrics."""
        try:
            process = psutil.Process()
            
            return {
                'cpu_percent': process.cpu_percent(),
                'memory_mb': process.memory_info().rss / 1024**2,
                'gpu_memory_mb': torch.cuda.memory_allocated() / 1024**2 if torch.cuda.is_available() else 0
            }
        except Exception:
            return {}
    
    def _monitor_resources(self):
        """Monitor resource usage in background thread."""
        while self.running:
            try:
                process = psutil.Process()
                
                self.status.cpu_usage = process.cpu_percent()
                self.status.memory_usage_mb = process.memory_info().rss / 1024**2
                
                if torch.cuda.is_available():
                    self.status.gpu_usage = torch.cuda.utilization()
                
                time.sleep(5)  # Update every 5 seconds
                
            except Exception as e:
                logger.error(f"Resource monitoring error: {e}")
                time.sleep(10)


class DistributedCoordinator:
    """Coordinates distributed execution across multiple workers."""
    
    def __init__(self, config: ClusterConfig):
        self.config = config
        self.task_queue = RedisTaskQueue(config)
        self.workers = []
        self.running = False
        
        # Auto-scaling parameters
        self.target_queue_length = 10
        self.scale_up_threshold = 20
        self.scale_down_threshold = 5
        self.min_workers = 2
        self.max_workers = config.max_workers_per_node * 4  # Assuming 4 nodes max
        
        # Performance tracking
        self.throughput_history = deque(maxlen=100)
        self.load_balancing_stats = defaultdict(int)
        
    def start_workers(self, num_workers: int):
        """Start distributed workers."""
        for i in range(num_workers):
            worker_id = f"worker_{i}_{int(time.time())}"
            
            # Start worker in separate process
            worker_process = mp.Process(
                target=self._run_worker,
                args=(worker_id,)
            )
            worker_process.start()
            
            self.workers.append({
                'worker_id': worker_id,
                'process': worker_process,
                'start_time': time.time()
            })
        
        logger.info(f"Started {num_workers} distributed workers")
    
    def _run_worker(self, worker_id: str):
        """Run worker in separate process."""
        worker = DistributedWorker(self.config, worker_id)
        worker.start()
    
    def submit_batch_tasks(self, tasks: List[Tuple[TaskConfig, Any]]) -> List[str]:
        """Submit batch of tasks for distributed execution."""
        task_ids = []
        
        for task_config, task_data in tasks:
            success = self.task_queue.submit_task(task_config, task_data)
            if success:
                task_ids.append(task_config.task_id)
        
        logger.info(f"Submitted {len(task_ids)} tasks for distributed execution")
        
        return task_ids
    
    def wait_for_completion(self, task_ids: List[str], 
                           timeout: Optional[float] = None) -> List[TaskResult]:
        """Wait for tasks to complete and return results."""
        start_time = time.time()
        completed_results = []
        pending_task_ids = set(task_ids)
        
        while pending_task_ids:
            # Check for completed tasks
            for task_id in list(pending_task_ids):
                try:
                    result_data = self.task_queue.redis_client.hget(
                        self.task_queue.completed_queue, task_id
                    )
                    
                    if result_data:
                        result = pickle.loads(result_data)
                        completed_results.append(result)
                        pending_task_ids.remove(task_id)
                        continue
                    
                    # Check failed queue
                    result_data = self.task_queue.redis_client.hget(
                        self.task_queue.failed_queue, task_id
                    )
                    
                    if result_data:
                        result = pickle.loads(result_data)
                        completed_results.append(result)
                        pending_task_ids.remove(task_id)
                        
                except Exception as e:
                    logger.error(f"Error checking task {task_id}: {e}")
            
            # Check timeout
            if timeout and (time.time() - start_time) > timeout:
                logger.warning(f"Timeout waiting for tasks: {pending_task_ids}")
                break
            
            time.sleep(1)
        
        return completed_results
    
    def auto_scale_workers(self):
        """Automatically scale workers based on queue length."""
        stats = self.task_queue.get_queue_stats()
        pending_tasks = stats['pending']
        
        worker_statuses = self.task_queue.get_worker_statuses()
        active_workers = sum(1 for w in worker_statuses if w.status in ['idle', 'busy'])
        
        # Scale up if queue is too long
        if pending_tasks > self.scale_up_threshold and active_workers < self.max_workers:
            workers_to_add = min(
                pending_tasks // self.target_queue_length,
                self.max_workers - active_workers
            )
            
            if workers_to_add > 0:
                self.start_workers(workers_to_add)
                logger.info(f"Scaled up: added {workers_to_add} workers")
        
        # Scale down if queue is too short
        elif pending_tasks < self.scale_down_threshold and active_workers > self.min_workers:
            # Signal workers to shut down gracefully
            excess_workers = active_workers - max(self.min_workers, pending_tasks)
            
            if excess_workers > 0:
                # Find idle workers to shut down
                idle_workers = [w for w in worker_statuses if w.status == 'idle']
                
                for i, worker in enumerate(idle_workers[:excess_workers]):
                    # Send shutdown signal via Redis
                    self.task_queue.redis_client.hset(
                        "worker_commands",
                        worker.worker_id,
                        "shutdown"
                    )
                
                logger.info(f"Scaled down: signaled {min(excess_workers, len(idle_workers))} workers to shut down")
    
    def get_cluster_status(self) -> Dict[str, Any]:
        """Get comprehensive cluster status."""
        stats = self.task_queue.get_queue_stats()
        worker_statuses = self.task_queue.get_worker_statuses()
        
        # Aggregate worker stats
        worker_stats = {
            'total': len(worker_statuses),
            'idle': sum(1 for w in worker_statuses if w.status == 'idle'),
            'busy': sum(1 for w in worker_statuses if w.status == 'busy'),
            'offline': sum(1 for w in worker_statuses if w.status == 'offline'),
            'error': sum(1 for w in worker_statuses if w.status == 'error')
        }
        
        # Calculate throughput
        completed_in_last_minute = self._calculate_recent_throughput()
        
        # Resource utilization
        avg_cpu = np.mean([w.cpu_usage for w in worker_statuses if w.status != 'offline'])
        avg_memory = np.mean([w.memory_usage_mb for w in worker_statuses if w.status != 'offline'])
        avg_gpu = np.mean([w.gpu_usage for w in worker_statuses if w.status != 'offline' and w.gpu_usage > 0])
        
        return {
            'queue_stats': stats,
            'worker_stats': worker_stats,
            'throughput_per_minute': completed_in_last_minute,
            'resource_utilization': {
                'avg_cpu_percent': avg_cpu if not np.isnan(avg_cpu) else 0,
                'avg_memory_mb': avg_memory if not np.isnan(avg_memory) else 0,
                'avg_gpu_percent': avg_gpu if not np.isnan(avg_gpu) else 0
            },
            'auto_scaling': {
                'target_queue_length': self.target_queue_length,
                'scale_up_threshold': self.scale_up_threshold,
                'scale_down_threshold': self.scale_down_threshold
            }
        }
    
    def _calculate_recent_throughput(self) -> int:
        """Calculate throughput in the last minute."""
        try:
            current_time = time.time()
            one_minute_ago = current_time - 60
            
            # Count completed tasks in the last minute
            completed_keys = self.task_queue.redis_client.hkeys(self.task_queue.completed_queue)
            
            recent_completions = 0
            for key in completed_keys[-100:]:  # Check last 100 to avoid performance issues
                try:
                    result_data = self.task_queue.redis_client.hget(self.task_queue.completed_queue, key)
                    if result_data:
                        result = pickle.loads(result_data)
                        # Approximate completion time (task doesn't store this directly)
                        if hasattr(result, 'completion_time'):
                            if result.completion_time > one_minute_ago:
                                recent_completions += 1
                except Exception:
                    continue
            
            return recent_completions
            
        except Exception as e:
            logger.error(f"Failed to calculate throughput: {e}")
            return 0
    
    def shutdown(self):
        """Shutdown coordinator and all workers."""
        logger.info("Shutting down distributed coordinator")
        
        # Signal all workers to stop
        worker_statuses = self.task_queue.get_worker_statuses()
        for worker in worker_statuses:
            self.task_queue.redis_client.hset(
                "worker_commands",
                worker.worker_id,
                "shutdown"
            )
        
        # Wait for workers to shut down
        time.sleep(5)
        
        # Force kill any remaining processes
        for worker_info in self.workers:
            try:
                worker_info['process'].terminate()
                worker_info['process'].join(timeout=5)
                
                if worker_info['process'].is_alive():
                    worker_info['process'].kill()
            except Exception as e:
                logger.error(f"Error shutting down worker: {e}")
        
        self.workers.clear()


class DistributedProcessingEngine:
    """Main engine for distributed processing of research workloads."""
    
    def __init__(self, config: ClusterConfig):
        self.config = config
        self.coordinator = DistributedCoordinator(config)
        self.task_cache = {}
        self.result_cache = {}
        
        # Performance optimization
        self.batch_size = 32
        self.enable_caching = True
        self.cache_ttl = 3600  # 1 hour
        
    def initialize_cluster(self, num_workers: int = None):
        """Initialize distributed cluster."""
        if num_workers is None:
            num_workers = min(multiprocessing.cpu_count(), self.config.max_workers_per_node)
        
        logger.info(f"Initializing cluster with {num_workers} workers")
        
        # Start workers
        self.coordinator.start_workers(num_workers)
        
        # Wait for workers to become available
        time.sleep(5)
        
        # Enable auto-scaling
        self._start_auto_scaling()
        
        logger.info("Distributed cluster initialized")
    
    def _start_auto_scaling(self):
        """Start auto-scaling monitor."""
        def auto_scale_monitor():
            while True:
                try:
                    self.coordinator.auto_scale_workers()
                    time.sleep(30)  # Check every 30 seconds
                except Exception as e:
                    logger.error(f"Auto-scaling error: {e}")
                    time.sleep(60)
        
        auto_scale_thread = threading.Thread(target=auto_scale_monitor, daemon=True)
        auto_scale_thread.start()
    
    def execute_distributed_benchmark(self, 
                                    component_configs: List[Dict[str, Any]], 
                                    num_trials_per_config: int = 10) -> Dict[str, Any]:
        """Execute distributed benchmark across multiple component configurations."""
        
        logger.info(f"Starting distributed benchmark with {len(component_configs)} configurations")
        
        # Create tasks for each configuration
        tasks = []
        task_id_to_config = {}
        
        for i, config in enumerate(component_configs):
            for trial in range(num_trials_per_config):
                task_id = f"benchmark_{i}_{trial}_{int(time.time())}"
                
                task_config = TaskConfig(
                    task_id=task_id,
                    task_type=config['component_type'],
                    priority=1,
                    timeout_seconds=1800.0,  # 30 minutes
                    gpu_required=config.get('gpu_required', False)
                )
                
                task_data = {
                    'config': config.get('component_config'),
                    'test_inputs': {
                        'num_trials': 1,  # Single trial per task for parallelization
                        'batch_size': config.get('batch_size', 16)
                    }
                }
                
                tasks.append((task_config, task_data))
                task_id_to_config[task_id] = (i, config)
        
        # Submit tasks
        task_ids = self.coordinator.submit_batch_tasks(tasks)
        
        # Wait for completion
        results = self.coordinator.wait_for_completion(task_ids, timeout=3600)
        
        # Aggregate results by configuration
        config_results = defaultdict(list)
        
        for result in results:
            if result.task_id in task_id_to_config:
                config_idx, config = task_id_to_config[result.task_id]
                config_results[config_idx].append(result)
        
        # Compute statistics for each configuration
        benchmark_summary = {}
        
        for config_idx, config in enumerate(component_configs):
            config_name = config.get('name', f'config_{config_idx}')
            trial_results = config_results[config_idx]
            
            if trial_results:
                # Extract metrics
                execution_times = [r.execution_time for r in trial_results if r.success]
                success_rate = sum(1 for r in trial_results if r.success) / len(trial_results)
                
                # Component-specific metrics
                component_metrics = defaultdict(list)
                for result in trial_results:
                    if result.success and result.result_data:
                        for metric_name, value in result.result_data.items():
                            if isinstance(value, (int, float)):
                                component_metrics[metric_name].append(value)
                
                # Compute aggregated metrics
                aggregated_metrics = {}
                for metric_name, values in component_metrics.items():
                    if values:
                        aggregated_metrics[metric_name] = {
                            'mean': np.mean(values),
                            'std': np.std(values),
                            'min': np.min(values),
                            'max': np.max(values),
                            'median': np.median(values)
                        }
                
                benchmark_summary[config_name] = {
                    'config': config,
                    'num_trials': len(trial_results),
                    'success_rate': success_rate,
                    'execution_time': {
                        'mean': np.mean(execution_times) if execution_times else 0,
                        'std': np.std(execution_times) if execution_times else 0,
                        'min': np.min(execution_times) if execution_times else 0,
                        'max': np.max(execution_times) if execution_times else 0
                    },
                    'component_metrics': aggregated_metrics
                }
            else:
                benchmark_summary[config_name] = {
                    'config': config,
                    'num_trials': 0,
                    'success_rate': 0.0,
                    'execution_time': {'mean': 0, 'std': 0, 'min': 0, 'max': 0},
                    'component_metrics': {}
                }
        
        # Overall statistics
        total_tasks = len(tasks)
        successful_tasks = sum(1 for r in results if r.success)
        total_execution_time = sum(r.execution_time for r in results)
        
        return {
            'benchmark_summary': benchmark_summary,
            'overall_stats': {
                'total_tasks': total_tasks,
                'successful_tasks': successful_tasks,
                'success_rate': successful_tasks / total_tasks if total_tasks > 0 else 0,
                'total_execution_time': total_execution_time,
                'average_task_time': total_execution_time / total_tasks if total_tasks > 0 else 0
            },
            'cluster_performance': self.coordinator.get_cluster_status()
        }
    
    def execute_hyperparameter_sweep(self, 
                                   component_type: str,
                                   parameter_grid: Dict[str, List[Any]],
                                   evaluation_metric: str = "accuracy") -> Dict[str, Any]:
        """Execute distributed hyperparameter sweep."""
        
        # Generate all parameter combinations
        import itertools
        
        param_names = list(parameter_grid.keys())
        param_values = list(parameter_grid.values())
        param_combinations = list(itertools.product(*param_values))
        
        logger.info(f"Starting hyperparameter sweep with {len(param_combinations)} combinations")
        
        # Create tasks for each parameter combination
        tasks = []
        task_id_to_params = {}
        
        for i, param_combo in enumerate(param_combinations):
            task_id = f"hyperparam_{i}_{int(time.time())}"
            
            # Create config from parameter combination
            config = dict(zip(param_names, param_combo))
            
            task_config = TaskConfig(
                task_id=task_id,
                task_type=component_type,
                priority=1,
                timeout_seconds=1800.0,
                gpu_required=config.get('gpu_required', False)
            )
            
            task_data = {
                'config': config,
                'test_inputs': {
                    'num_trials': 5,  # Multiple trials for statistical significance
                    'evaluation_metric': evaluation_metric
                }
            }
            
            tasks.append((task_config, task_data))
            task_id_to_params[task_id] = config
        
        # Submit and execute tasks
        task_ids = self.coordinator.submit_batch_tasks(tasks)
        results = self.coordinator.wait_for_completion(task_ids, timeout=7200)  # 2 hours
        
        # Process results
        sweep_results = []
        
        for result in results:
            if result.task_id in task_id_to_params and result.success:
                params = task_id_to_params[result.task_id]
                
                # Extract evaluation metric
                metric_value = 0.0
                if result.result_data and evaluation_metric in result.result_data:
                    metric_value = result.result_data[evaluation_metric]
                elif result.result_data and isinstance(result.result_data, dict):
                    # Try to find metric in nested structure
                    for key, value in result.result_data.items():
                        if evaluation_metric in str(key).lower() and isinstance(value, (int, float)):
                            metric_value = value
                            break
                
                sweep_results.append({
                    'parameters': params,
                    'metric_value': metric_value,
                    'execution_time': result.execution_time,
                    'all_metrics': result.result_data
                })
        
        # Find best parameters
        if sweep_results:
            best_result = max(sweep_results, key=lambda x: x['metric_value'])
            
            # Sort results by metric value
            sweep_results.sort(key=lambda x: x['metric_value'], reverse=True)
            
            return {
                'best_parameters': best_result['parameters'],
                'best_metric_value': best_result['metric_value'],
                'all_results': sweep_results,
                'num_combinations_tested': len(sweep_results),
                'num_combinations_total': len(param_combinations)
            }
        else:
            return {
                'best_parameters': None,
                'best_metric_value': 0.0,
                'all_results': [],
                'num_combinations_tested': 0,
                'num_combinations_total': len(param_combinations)
            }
    
    def shutdown(self):
        """Shutdown distributed processing engine."""
        logger.info("Shutting down distributed processing engine")
        self.coordinator.shutdown()


def create_distributed_processing_engine(config: Optional[ClusterConfig] = None) -> DistributedProcessingEngine:
    """Factory function to create distributed processing engine."""
    if config is None:
        config = ClusterConfig()
    
    engine = DistributedProcessingEngine(config)
    
    logger.info(f"Created Distributed Processing Engine")
    logger.info(f"Master: {config.master_addr}:{config.master_port}")
    logger.info(f"Backend: {config.backend}, World size: {config.world_size}")
    
    return engine


def run_distributed_benchmark_suite(component_configs: List[Dict[str, Any]],
                                   cluster_config: Optional[ClusterConfig] = None) -> Dict[str, Any]:
    """Run comprehensive distributed benchmark suite."""
    
    # Create engine
    engine = create_distributed_processing_engine(cluster_config)
    
    try:
        # Initialize cluster
        engine.initialize_cluster()
        
        # Run benchmark
        results = engine.execute_distributed_benchmark(component_configs)
        
        logger.info("Distributed benchmark suite completed successfully")
        
        return results
        
    finally:
        # Always shutdown
        engine.shutdown()


if __name__ == "__main__":
    # Example usage
    example_configs = [
        {
            'name': 'attention_fusion_small',
            'component_type': 'attention_fusion',
            'component_config': {'hidden_dim': 256, 'num_heads': 4},
            'batch_size': 16,
            'gpu_required': True
        },
        {
            'name': 'attention_fusion_large',
            'component_type': 'attention_fusion',
            'component_config': {'hidden_dim': 512, 'num_heads': 8},
            'batch_size': 32,
            'gpu_required': True
        },
        {
            'name': 'quantum_planning_basic',
            'component_type': 'quantum_planning',
            'component_config': {'num_qubits': 6, 'planning_horizon': 10},
            'batch_size': 8,
            'gpu_required': False
        }
    ]
    
    results = run_distributed_benchmark_suite(example_configs)
    
    print("Distributed Benchmark Results:")
    print(json.dumps(results, indent=2))
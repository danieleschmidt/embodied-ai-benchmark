"""Performance optimization utilities for the embodied AI benchmark."""

import time
import threading
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Callable, Union, Tuple
import numpy as np
import logging
from functools import wraps, lru_cache
from dataclasses import dataclass
import queue
import weakref
import gc

logger = logging.getLogger(__name__)


@dataclass
class OptimizationConfig:
    """Configuration for optimization settings."""
    enable_multiprocessing: bool = True
    enable_multithreading: bool = True
    max_workers: int = min(mp.cpu_count(), 8)  # Reasonable limit for resource usage
    batch_size: int = 32
    prefetch_size: int = 2
    memory_limit_mb: int = 2048  # Increased for production workloads
    enable_jit_compilation: bool = False  # Disabled by default for compatibility
    enable_vectorization: bool = True
    enable_caching: bool = True


class BatchProcessor:
    """Efficient batch processing for benchmark operations."""
    
    def __init__(self, 
                 batch_size: int = 32,
                 max_workers: int = None,
                 use_processes: bool = False):
        """Initialize batch processor.
        
        Args:
            batch_size: Size of processing batches
            max_workers: Maximum number of worker threads/processes
            use_processes: Whether to use processes instead of threads
        """
        self.batch_size = batch_size
        self.max_workers = max_workers or mp.cpu_count()
        self.use_processes = use_processes
        
        self._executor = None
        self._setup_executor()
    
    def _setup_executor(self):
        """Setup thread/process pool executor."""
        if self.use_processes:
            self._executor = ProcessPoolExecutor(max_workers=self.max_workers)
        else:
            self._executor = ThreadPoolExecutor(max_workers=self.max_workers)
    
    def process_batch(self, 
                     items: List[Any], 
                     process_func: Callable,
                     *args, **kwargs) -> List[Any]:
        """Process items in batches.
        
        Args:
            items: Items to process
            process_func: Function to apply to each item
            *args, **kwargs: Additional arguments for process_func
            
        Returns:
            List of processed results
        """
        if not items:
            return []
        
        results = []
        batches = [items[i:i + self.batch_size] 
                  for i in range(0, len(items), self.batch_size)]
        
        if len(batches) == 1 or self.max_workers == 1:
            # Process sequentially
            for batch in batches:
                batch_results = [process_func(item, *args, **kwargs) for item in batch]
                results.extend(batch_results)
        else:
            # Process in parallel
            futures = []
            for batch in batches:
                future = self._executor.submit(self._process_batch_items, 
                                             batch, process_func, args, kwargs)
                futures.append(future)
            
            for future in as_completed(futures):
                try:
                    batch_results = future.result()
                    results.extend(batch_results)
                except Exception as e:
                    logger.error(f"Batch processing error: {e}")
                    raise
        
        return results
    
    @staticmethod
    def _process_batch_items(batch: List[Any], 
                            process_func: Callable,
                            args: tuple, 
                            kwargs: dict) -> List[Any]:
        """Process a batch of items."""
        return [process_func(item, *args, **kwargs) for item in batch]
    
    def close(self):
        """Close the executor."""
        if self._executor:
            self._executor.shutdown(wait=True)
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


class ParallelEvaluator:
    """Parallel evaluation system for benchmark tasks."""
    
    def __init__(self, 
                 max_workers: int = None,
                 use_processes: bool = False,
                 batch_size: int = 8):
        """Initialize parallel evaluator.
        
        Args:
            max_workers: Maximum number of workers
            use_processes: Whether to use processes (for CPU-bound tasks)
            batch_size: Batch size for parallel execution
        """
        self.max_workers = max_workers or mp.cpu_count()
        self.use_processes = use_processes
        self.batch_size = batch_size
        
        self._executor = None
        self._active_futures = []
        self._results_queue = queue.Queue()
        
    def evaluate_parallel(self, 
                         eval_func: Callable,
                         eval_args_list: List[Tuple],
                         callback: Optional[Callable] = None) -> List[Any]:
        """Evaluate multiple tasks in parallel.
        
        Args:
            eval_func: Evaluation function
            eval_args_list: List of argument tuples for eval_func
            callback: Optional callback for each completed evaluation
            
        Returns:
            List of evaluation results
        """
        if not eval_args_list:
            return []
        
        # Setup executor
        if self.use_processes:
            self._executor = ProcessPoolExecutor(max_workers=self.max_workers)
        else:
            self._executor = ThreadPoolExecutor(max_workers=self.max_workers)
        
        try:
            # Submit all tasks
            futures = []
            for i, args in enumerate(eval_args_list):
                future = self._executor.submit(self._safe_eval, eval_func, args, i)
                futures.append((future, i))
            
            # Collect results as they complete
            results = [None] * len(eval_args_list)
            completed_count = 0
            
            for future, index in futures:
                try:
                    result = future.result()
                    results[index] = result
                    completed_count += 1
                    
                    if callback:
                        callback(result, index, completed_count, len(eval_args_list))
                    
                except Exception as e:
                    logger.error(f"Parallel evaluation error at index {index}: {e}")
                    results[index] = {"error": str(e), "index": index}
            
            return results
            
        finally:
            if self._executor:
                self._executor.shutdown(wait=True)
                self._executor = None
    
    @staticmethod
    def _safe_eval(eval_func: Callable, args: Tuple, index: int) -> Any:
        """Safely evaluate function with error handling."""
        try:
            return eval_func(*args)
        except Exception as e:
            logger.error(f"Evaluation failed at index {index}: {e}")
            return {"error": str(e), "index": index}


class VectorizedOperations:
    """Vectorized operations for performance-critical computations."""
    
    @staticmethod
    def batch_normalize(data: np.ndarray, 
                       epsilon: float = 1e-8) -> np.ndarray:
        """Vectorized batch normalization."""
        mean = np.mean(data, axis=0, keepdims=True)
        std = np.std(data, axis=0, keepdims=True)
        return (data - mean) / (std + epsilon)
    
    @staticmethod
    def batch_distance_matrix(points1: np.ndarray, 
                             points2: np.ndarray) -> np.ndarray:
        """Compute distance matrix between two sets of points."""
        # points1: (N1, D), points2: (N2, D)
        # Returns: (N1, N2) distance matrix
        diff = points1[:, np.newaxis, :] - points2[np.newaxis, :, :]
        return np.sqrt(np.sum(diff ** 2, axis=2))
    
    @staticmethod
    def batch_softmax(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
        """Vectorized softmax with temperature scaling."""
        scaled_logits = logits / temperature
        exp_logits = np.exp(scaled_logits - np.max(scaled_logits, axis=-1, keepdims=True))
        return exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)
    
    @staticmethod
    def batch_reward_computation(states: np.ndarray,
                                actions: np.ndarray,
                                next_states: np.ndarray,
                                reward_weights: np.ndarray) -> np.ndarray:
        """Vectorized reward computation for multiple state transitions."""
        # Compute state differences
        state_diff = next_states - states
        
        # Progress reward based on movement toward goal
        progress = np.sum(state_diff * reward_weights, axis=-1)
        
        # Action penalty
        action_penalty = -0.01 * np.sum(np.abs(actions), axis=-1)
        
        return progress + action_penalty


class MemoryOptimizer:
    """Memory optimization utilities for large-scale benchmarks."""
    
    def __init__(self, memory_limit_mb: int = 1024):
        """Initialize memory optimizer.
        
        Args:
            memory_limit_mb: Memory limit in megabytes
        """
        self.memory_limit_bytes = memory_limit_mb * 1024 * 1024
        self._object_pool = {}
        self._weak_refs = weakref.WeakSet()
        
    def get_memory_usage(self) -> Dict[str, float]:
        """Get current memory usage statistics."""
        import psutil
        process = psutil.Process()
        memory_info = process.memory_info()
        
        return {
            "rss_mb": memory_info.rss / (1024 * 1024),
            "vms_mb": memory_info.vms / (1024 * 1024),
            "percent": process.memory_percent(),
            "available_mb": psutil.virtual_memory().available / (1024 * 1024)
        }
    
    def check_memory_pressure(self) -> bool:
        """Check if memory usage is high."""
        memory_info = self.get_memory_usage()
        return memory_info["rss_mb"] * 1024 * 1024 > self.memory_limit_bytes
    
    def optimize_memory(self):
        """Perform memory optimization."""
        if self.check_memory_pressure():
            logger.info("Memory pressure detected, running optimization")
            
            # Clear object pool
            self._object_pool.clear()
            
            # Force garbage collection
            gc.collect()
            
            logger.info(f"Memory optimization completed. Current usage: {self.get_memory_usage()}")
    
    def create_object_pool(self, 
                          object_type: type, 
                          pool_size: int = 100,
                          init_args: tuple = (),
                          init_kwargs: dict = None) -> List[Any]:
        """Create a pool of reusable objects."""
        if init_kwargs is None:
            init_kwargs = {}
        
        pool_key = f"{object_type.__name__}_{hash(init_args)}_{hash(tuple(init_kwargs.items()))}"
        
        if pool_key not in self._object_pool:
            pool = []
            for _ in range(pool_size):
                obj = object_type(*init_args, **init_kwargs)
                pool.append(obj)
                self._weak_refs.add(obj)
            
            self._object_pool[pool_key] = pool
            logger.info(f"Created object pool: {pool_key} with {pool_size} objects")
        
        return self._object_pool[pool_key]
    
    def get_pooled_object(self, object_type: type, **kwargs) -> Any:
        """Get object from pool or create new one."""
        pool_key = f"{object_type.__name__}_{hash(tuple(kwargs.items()))}"
        
        if pool_key in self._object_pool and self._object_pool[pool_key]:
            return self._object_pool[pool_key].pop()
        else:
            return object_type(**kwargs)
    
    def return_to_pool(self, obj: Any, object_type: type, **kwargs):
        """Return object to pool for reuse."""
        pool_key = f"{object_type.__name__}_{hash(tuple(kwargs.items()))}"
        
        if pool_key not in self._object_pool:
            self._object_pool[pool_key] = []
        
        # Reset object state if it has a reset method
        if hasattr(obj, 'reset'):
            obj.reset()
        
        self._object_pool[pool_key].append(obj)


class JITCompiler:
    """Just-in-time compilation for performance-critical functions."""
    
    def __init__(self):
        self._numba_available = self._check_numba()
        self._compiled_functions = {}
    
    def _check_numba(self) -> bool:
        """Check if Numba is available for JIT compilation."""
        try:
            import numba
            return True
        except ImportError:
            logger.info("Numba not available, JIT compilation disabled")
            return False
    
    def jit_compile(self, func: Callable, 
                   nopython: bool = True,
                   cache: bool = True) -> Callable:
        """JIT compile function if Numba is available."""
        if not self._numba_available:
            return func
        
        func_key = f"{func.__name__}_{id(func)}"
        
        if func_key not in self._compiled_functions:
            try:
                import numba
                compiled_func = numba.jit(nopython=nopython, cache=cache)(func)
                self._compiled_functions[func_key] = compiled_func
                logger.info(f"JIT compiled function: {func.__name__}")
            except Exception as e:
                logger.warning(f"JIT compilation failed for {func.__name__}: {e}")
                self._compiled_functions[func_key] = func
        
        return self._compiled_functions[func_key]
    
    def vectorize(self, func: Callable, 
                 signature: Optional[str] = None) -> Callable:
        """Vectorize function for NumPy arrays."""
        if not self._numba_available:
            return np.vectorize(func)
        
        try:
            import numba
            if signature:
                return numba.vectorize([signature], cache=True)(func)
            else:
                return numba.vectorize(cache=True)(func)
        except Exception as e:
            logger.warning(f"Vectorization failed for {func.__name__}: {e}")
            return np.vectorize(func)


def profile_performance(func: Callable) -> Callable:
    """Decorator to profile function performance."""
    
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        start_memory = 0
        
        # Try to get memory usage
        try:
            import psutil
            process = psutil.Process()
            start_memory = process.memory_info().rss
        except ImportError:
            pass
        
        try:
            result = func(*args, **kwargs)
            
            end_time = time.perf_counter()
            execution_time = end_time - start_time
            
            # Calculate memory change
            memory_change = 0
            if start_memory > 0:
                try:
                    end_memory = process.memory_info().rss
                    memory_change = end_memory - start_memory
                except:
                    pass
            
            # Log performance metrics
            logger.info(f"PERFORMANCE: {func.__name__} - "
                       f"Time: {execution_time:.4f}s, "
                       f"Memory: {memory_change / (1024*1024):.2f}MB")
            
            return result
            
        except Exception as e:
            logger.error(f"Performance profiling error in {func.__name__}: {e}")
            raise
    
    return wrapper


def optimize_numpy_operations():
    """Configure NumPy for optimal performance."""
    # Set optimal thread count for NumPy operations
    optimal_threads = min(mp.cpu_count(), 8)  # Don't use too many threads
    
    try:
        import os
        os.environ['OMP_NUM_THREADS'] = str(optimal_threads)
        os.environ['MKL_NUM_THREADS'] = str(optimal_threads)
        os.environ['NUMEXPR_NUM_THREADS'] = str(optimal_threads)
        
        # Configure NumPy
        np.seterr(all='warn')  # Warn on numerical errors
        
        logger.info(f"Optimized NumPy operations with {optimal_threads} threads")
        
    except Exception as e:
        logger.warning(f"Failed to optimize NumPy operations: {e}")


# Global optimization instances
batch_processor = BatchProcessor()
parallel_evaluator = ParallelEvaluator()
vectorized_ops = VectorizedOperations()
memory_optimizer = MemoryOptimizer()
jit_compiler = JITCompiler()

# Initialize NumPy optimizations
optimize_numpy_operations()
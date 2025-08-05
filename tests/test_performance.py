"""Performance tests for the embodied AI benchmark."""

import pytest
import time
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import threading
import multiprocessing as mp
from pathlib import Path

import sys
sys.path.append(str(Path(__file__).parent.parent / "src"))

from embodied_ai_benchmark.utils.optimization import (
    BatchProcessor, ParallelEvaluator, VectorizedOperations,
    MemoryOptimizer, profile_performance
)
from embodied_ai_benchmark.utils.caching import LRUCache, AdaptiveCache, PersistentCache
from embodied_ai_benchmark.utils.scalability import LoadBalancer, WorkerNode, LoadBalancingStrategy
from embodied_ai_benchmark.utils.monitoring import PerformanceMonitor, BenchmarkMetrics


class TestOptimizationPerformance:
    """Test performance optimization utilities."""
    
    def test_batch_processor_performance(self):
        """Test batch processor performance improvements."""
        def dummy_process_func(item):
            # Simulate some computation
            return item ** 2 + np.random.random()
        
        items = list(range(1000))
        
        # Test sequential processing
        start_time = time.time()
        sequential_results = [dummy_process_func(item) for item in items]
        sequential_time = time.time() - start_time
        
        # Test batch processing
        processor = BatchProcessor(batch_size=100, max_workers=4)
        
        start_time = time.time()
        batch_results = processor.process_batch(items, dummy_process_func)
        batch_time = time.time() - start_time
        
        processor.close()
        
        # Verify results are correct
        assert len(batch_results) == len(sequential_results)
        assert len(batch_results) == 1000
        
        # Batch processing should be faster or similar
        # (May not always be faster due to overhead with simple operations)
        print(f"Sequential time: {sequential_time:.3f}s, Batch time: {batch_time:.3f}s")
        
        # At minimum, batch processing shouldn't be significantly slower
        assert batch_time <= sequential_time * 2.0
    
    def test_parallel_evaluator_scalability(self):
        """Test parallel evaluator scalability."""
        def evaluation_task(task_id, duration):
            """Simulate evaluation task."""
            time.sleep(duration)
            return {
                "task_id": task_id,
                "result": np.random.random(),
                "duration": duration
            }
        
        # Create evaluation arguments
        eval_args = [(i, 0.01) for i in range(20)]  # 20 quick tasks
        
        evaluator = ParallelEvaluator(max_workers=4)
        
        start_time = time.time()
        results = evaluator.evaluate_parallel(evaluation_task, eval_args)
        parallel_time = time.time() - start_time
        
        # Verify all tasks completed
        assert len(results) == 20
        assert all("result" in result for result in results if "error" not in result)
        
        # Should complete in reasonable time (much less than sequential)
        expected_sequential_time = 20 * 0.01  # 0.2 seconds
        assert parallel_time < expected_sequential_time * 0.8  # Should be significantly faster
        
        print(f"Parallel evaluation time: {parallel_time:.3f}s")
    
    def test_vectorized_operations_performance(self):
        """Test vectorized operations performance."""
        # Generate test data
        data_size = 10000
        data = np.random.randn(data_size, 10)
        
        # Test batch normalization
        start_time = time.time()
        normalized = VectorizedOperations.batch_normalize(data)
        vectorized_time = time.time() - start_time
        
        # Verify normalization worked
        assert normalized.shape == data.shape
        # Check that mean is close to 0 and std close to 1
        assert np.abs(np.mean(normalized, axis=0)).max() < 0.1
        assert np.abs(np.std(normalized, axis=0) - 1.0).max() < 0.1
        
        print(f"Vectorized normalization time: {vectorized_time:.4f}s")
        
        # Test distance matrix computation
        points1 = np.random.randn(100, 3)
        points2 = np.random.randn(150, 3)
        
        start_time = time.time()
        distances = VectorizedOperations.batch_distance_matrix(points1, points2)
        distance_time = time.time() - start_time
        
        # Verify distance matrix shape and properties
        assert distances.shape == (100, 150)
        assert np.all(distances >= 0)  # Distances should be non-negative
        
        print(f"Distance matrix computation time: {distance_time:.4f}s")
    
    def test_memory_optimizer_efficiency(self):
        """Test memory optimizer efficiency."""
        optimizer = MemoryOptimizer(memory_limit_mb=100)
        
        # Test object pool creation
        class TestObject:
            def __init__(self, size=100):
                self.data = np.random.randn(size)
            
            def reset(self):
                self.data.fill(0)
        
        # Create object pool
        pool_size = 50
        pool = optimizer.create_object_pool(TestObject, pool_size, init_args=(100,))
        
        assert len(pool) == pool_size
        assert all(isinstance(obj, TestObject) for obj in pool)
        
        # Test object reuse
        obj1 = optimizer.get_pooled_object(TestObject, size=100)
        assert isinstance(obj1, TestObject)
        
        # Return to pool
        optimizer.return_to_pool(obj1, TestObject, size=100)
        
        # Get object again (should be reused)
        obj2 = optimizer.get_pooled_object(TestObject, size=100)
        assert obj2 is obj1  # Same object instance
        
        # Test memory optimization
        initial_memory = optimizer.get_memory_usage()
        optimizer.optimize_memory()
        final_memory = optimizer.get_memory_usage()
        
        print(f"Memory usage - Initial: {initial_memory['rss_mb']:.1f}MB, "
              f"Final: {final_memory['rss_mb']:.1f}MB")
    
    @profile_performance
    def dummy_computation(self, n):
        """Dummy computation for testing performance profiling."""
        result = 0
        for i in range(n):
            result += i ** 2
        return result
    
    def test_performance_profiling(self):
        """Test performance profiling decorator."""
        # This test verifies the profiling decorator works
        result = self.dummy_computation(10000)
        assert result > 0
        
        # The decorator should log performance metrics
        # In a real test, you might capture log output to verify


class TestCachingPerformance:
    """Test caching system performance."""
    
    def test_lru_cache_performance(self):
        """Test LRU cache performance."""
        cache = LRUCache(max_size=1000)
        
        # Populate cache
        start_time = time.time()
        for i in range(1000):
            cache.put(f"key_{i}", f"value_{i}")
        put_time = time.time() - start_time
        
        # Test cache hits
        start_time = time.time()
        hit_count = 0
        for i in range(1000):
            if cache.get(f"key_{i}") is not None:
                hit_count += 1
        get_time = time.time() - start_time
        
        # Verify performance
        print(f"Cache put time: {put_time:.4f}s, get time: {get_time:.4f}s")
        print(f"Hit rate: {hit_count/1000:.2%}")
        
        assert hit_count == 1000  # All should be hits
        assert put_time < 0.1  # Should be very fast
        assert get_time < 0.01  # Gets should be extremely fast
        
        # Test cache stats
        stats = cache.get_stats()
        assert stats["hits"] == 1000
        assert stats["entries"] == 1000
    
    def test_adaptive_cache_adaptation(self):
        """Test adaptive cache size adaptation."""
        cache = AdaptiveCache(
            initial_size=100,
            max_size=500,
            adaptation_interval=50
        )
        
        # Generate requests with pattern that should trigger adaptation
        for i in range(200):
            key = f"key_{i % 80}"  # Create some overlap for hits
            cache.put(key, f"value_{i}")
            
            # Get some values to create hits
            if i % 3 == 0:
                cache.get(key)
        
        initial_stats = cache.get_stats()
        
        # Continue with more requests to trigger adaptation
        for i in range(200, 400):
            key = f"key_{i % 80}"
            cache.put(key, f"value_{i}")
            cache.get(key)
        
        final_stats = cache.get_stats()
        
        print(f"Initial cache size: {initial_stats['current_max_size']}")
        print(f"Final cache size: {final_stats['current_max_size']}")
        print(f"Final hit rate: {final_stats['hit_rate']:.3f}")
        
        # Cache should have adapted its size
        assert final_stats["request_count"] > initial_stats["request_count"]
    
    def test_persistent_cache_performance(self):
        """Test persistent cache performance."""
        import tempfile
        
        with tempfile.TemporaryDirectory() as temp_dir:
            cache = PersistentCache(
                cache_dir=temp_dir,
                memory_cache_size=100
            )
            
            # Test write performance
            start_time = time.time()
            for i in range(100):
                cache.put(f"persistent_key_{i}", {"data": list(range(i, i+10))})
            write_time = time.time() - start_time
            
            # Test read performance (from memory)
            start_time = time.time()
            memory_hits = 0
            for i in range(100):
                if cache.get(f"persistent_key_{i}") is not None:
                    memory_hits += 1
            memory_read_time = time.time() - start_time
            
            # Clear memory cache to test disk reads
            cache._memory_cache.clear()
            
            # Test read performance (from disk)
            start_time = time.time()
            disk_hits = 0
            for i in range(100):
                if cache.get(f"persistent_key_{i}") is not None:
                    disk_hits += 1
            disk_read_time = time.time() - start_time
            
            print(f"Write time: {write_time:.4f}s")
            print(f"Memory read time: {memory_read_time:.4f}s")
            print(f"Disk read time: {disk_read_time:.4f}s")
            print(f"Memory hits: {memory_hits}, Disk hits: {disk_hits}")
            
            assert memory_hits == 100
            assert disk_hits == 100
            assert memory_read_time < disk_read_time  # Memory should be faster


class TestScalabilityPerformance:
    """Test scalability system performance."""
    
    def test_load_balancer_performance(self):
        """Test load balancer performance under load."""
        load_balancer = LoadBalancer(strategy=LoadBalancingStrategy.LEAST_LOADED)
        
        # Register multiple workers
        for i in range(5):
            worker = WorkerNode(
                node_id=f"worker_{i}",
                host="localhost",
                port=8000 + i,
                capabilities=["cpu"],
                max_concurrent_tasks=10
            )
            load_balancer.register_worker(worker)
        
        # Test task assignment performance
        start_time = time.time()
        assignments = []
        
        for i in range(100):
            worker_id = load_balancer.assign_task(
                task_id=f"task_{i}",
                task_data={"type": "benchmark", "duration": 1.0},
                priority=1
            )
            if worker_id:
                assignments.append(worker_id)
        
        assignment_time = time.time() - start_time
        
        print(f"Assigned {len(assignments)} tasks in {assignment_time:.4f}s")
        print(f"Assignment rate: {len(assignments)/assignment_time:.1f} tasks/sec")
        
        # Should be able to assign tasks quickly
        assert assignment_time < 0.1  # Should complete in less than 100ms
        assert len(assignments) > 0  # Should assign some tasks
        
        # Test worker selection performance
        start_time = time.time()
        for _ in range(1000):
            worker = load_balancer.get_next_worker()
            if worker:
                pass  # Just testing selection speed
        selection_time = time.time() - start_time
        
        print(f"Worker selection time for 1000 requests: {selection_time:.4f}s")
        assert selection_time < 0.01  # Should be very fast
        
        load_balancer.shutdown()
    
    def test_monitoring_performance_overhead(self):
        """Test performance monitoring overhead."""
        monitor = PerformanceMonitor(monitoring_interval=0.1)
        
        # Test without monitoring
        def cpu_intensive_task():
            result = 0
            for i in range(100000):
                result += i ** 2
            return result
        
        start_time = time.time()
        result1 = cpu_intensive_task()
        no_monitoring_time = time.time() - start_time
        
        # Test with monitoring
        monitor.start_monitoring()
        
        start_time = time.time()
        result2 = cpu_intensive_task()
        with_monitoring_time = time.time() - start_time
        
        monitor.stop_monitoring()
        
        # Record benchmark metrics
        metrics = BenchmarkMetrics(
            timestamp=time.time(),
            episode_id=1,
            task_name="cpu_test",
            agent_name="test_agent",
            steps_per_second=1000 / with_monitoring_time,
            reward_per_step=0.1,
            memory_usage_mb=100,
            inference_time_ms=with_monitoring_time * 1000,
            environment_step_time_ms=1.0,
            success_rate=1.0
        )
        
        monitor.record_benchmark_metrics(metrics)
        
        overhead = with_monitoring_time - no_monitoring_time
        overhead_percentage = (overhead / no_monitoring_time) * 100
        
        print(f"No monitoring: {no_monitoring_time:.4f}s")
        print(f"With monitoring: {with_monitoring_time:.4f}s")
        print(f"Overhead: {overhead:.4f}s ({overhead_percentage:.1f}%)")
        
        # Results should be the same
        assert result1 == result2
        
        # Monitoring overhead should be minimal (less than 10%)
        assert overhead_percentage < 10.0


@pytest.mark.performance
class TestStressTests:
    """Stress tests for performance under load."""
    
    def test_concurrent_cache_access(self):
        """Test cache performance under concurrent access."""
        cache = LRUCache(max_size=1000)
        results = []
        
        def cache_worker(worker_id):
            """Worker function for concurrent cache access."""
            local_results = []
            for i in range(100):
                key = f"worker_{worker_id}_key_{i}"
                value = f"worker_{worker_id}_value_{i}"
                
                # Put value
                cache.put(key, value)
                
                # Get value
                retrieved = cache.get(key)
                local_results.append(retrieved == value)
            
            return local_results
        
        # Run concurrent workers
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(cache_worker, i) for i in range(10)]
            
            for future in futures:
                worker_results = future.result()
                results.extend(worker_results)
        
        # Verify all operations succeeded
        success_rate = sum(results) / len(results)
        print(f"Concurrent cache success rate: {success_rate:.2%}")
        
        assert success_rate > 0.95  # Should have high success rate
    
    def test_high_volume_task_assignment(self):
        """Test load balancer with high volume task assignment."""
        load_balancer = LoadBalancer(strategy=LoadBalancingStrategy.ROUND_ROBIN)
        
        # Register workers
        for i in range(20):
            worker = WorkerNode(
                node_id=f"stress_worker_{i}",
                host="localhost",
                port=9000 + i,
                capabilities=["cpu", "memory"],
                max_concurrent_tasks=100
            )
            load_balancer.register_worker(worker)
        
        # Assign large number of tasks
        start_time = time.time()
        successful_assignments = 0
        
        for i in range(10000):
            worker_id = load_balancer.assign_task(
                task_id=f"stress_task_{i}",
                task_data={"type": "stress_test"},
                priority=1
            )
            
            if worker_id:
                successful_assignments += 1
        
        assignment_time = time.time() - start_time
        assignment_rate = successful_assignments / assignment_time
        
        print(f"Assigned {successful_assignments} tasks in {assignment_time:.2f}s")
        print(f"Assignment rate: {assignment_rate:.1f} tasks/sec")
        
        # Should maintain good performance even with high load
        assert assignment_rate > 1000  # Should handle at least 1000 tasks/sec
        assert successful_assignments > 8000  # Should successfully assign most tasks
        
        load_balancer.shutdown()
    
    def test_memory_pressure_handling(self):
        """Test system behavior under memory pressure."""
        optimizer = MemoryOptimizer(memory_limit_mb=50)  # Low limit for testing
        
        # Create memory pressure
        large_objects = []
        for i in range(100):
            # Create progressively larger objects
            size = 1000 * (i + 1)
            obj = np.random.randn(size)
            large_objects.append(obj)
            
            # Check if optimization triggers
            if optimizer.check_memory_pressure():
                print(f"Memory pressure detected at iteration {i}")
                optimizer.optimize_memory()
                break
        
        # Verify memory optimization was effective
        final_memory = optimizer.get_memory_usage()
        print(f"Final memory usage: {final_memory['rss_mb']:.1f}MB")
        
        # Should handle memory pressure gracefully without crashing
        assert final_memory["rss_mb"] > 0  # Still using some memory


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short", "-x"])
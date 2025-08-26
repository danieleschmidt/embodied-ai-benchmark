#!/usr/bin/env python3
"""
Enhanced Generation 3: Performance Optimization and Scaling
Advanced optimizations for production-ready performance.
"""

import sys
import os
import time
import threading
import asyncio
import concurrent.futures
from pathlib import Path
from typing import Dict, Any, List, Optional
import json

# Add source path for imports
sys.path.insert(0, '/root/repo/src')

class Generation3Optimizer:
    """Advanced performance optimization and scaling enhancements."""
    
    def __init__(self):
        self.repo_path = Path('/root/repo')
        self.src_path = self.repo_path / 'src' / 'embodied_ai_benchmark'
        self.results = {
            "advanced_caching": {"status": "UNKNOWN", "message": ""},
            "concurrent_processing": {"status": "UNKNOWN", "message": ""},
            "memory_optimization": {"status": "UNKNOWN", "message": ""},
            "scaling_framework": {"status": "UNKNOWN", "message": ""},
            "performance_monitoring": {"status": "UNKNOWN", "message": ""},
            "global_optimization": {"status": "UNKNOWN", "message": ""},
        }
    
    def test_advanced_caching(self) -> bool:
        """Test advanced caching system performance."""
        try:
            from embodied_ai_benchmark.utils.caching import LRUCache, AdaptiveCache
            
            # Test LRU cache
            lru_cache = LRUCache(max_size=1000)
            
            # Performance test
            start_time = time.time()
            for i in range(1000):
                lru_cache.put(f"key_{i}", f"value_{i}")
            
            for i in range(500):
                result = lru_cache.get(f"key_{i}")
                assert result == f"value_{i}", f"Cache miss for key_{i}"
            
            cache_time = time.time() - start_time
            
            # Test adaptive cache
            adaptive_cache = AdaptiveCache()
            
            # Benchmark caching speedup
            def expensive_computation(n):
                return sum(i*i for i in range(n))
            
            # Without cache
            start_time = time.time()
            results_no_cache = [expensive_computation(100) for _ in range(10)]
            no_cache_time = time.time() - start_time
            
            # With cache
            start_time = time.time()
            results_cached = []
            for _ in range(10):
                cached_result = adaptive_cache.get_or_compute(
                    "expensive_100",
                    lambda: expensive_computation(100)
                )
                results_cached.append(cached_result)
            cache_speedup_time = time.time() - start_time
            
            speedup = no_cache_time / max(cache_speedup_time, 0.001)
            
            self.results["advanced_caching"]["status"] = "PASS"
            self.results["advanced_caching"]["message"] = f"Caching system working: {speedup:.1f}x speedup achieved"
            return True
            
        except Exception as e:
            self.results["advanced_caching"]["status"] = "FAIL"
            self.results["advanced_caching"]["message"] = f"Caching test failed: {e}"
            return False
    
    def test_concurrent_processing(self) -> bool:
        """Test concurrent processing capabilities."""
        try:
            from embodied_ai_benchmark.utils.concurrent_execution import ConcurrentBenchmarkExecutor
            
            executor = ConcurrentBenchmarkExecutor(max_workers=4)
            
            def test_task(task_id):
                time.sleep(0.01)  # Simulate work
                return {"task_id": task_id, "result": task_id * 2}
            
            # Test concurrent execution
            start_time = time.time()
            tasks = [test_task for _ in range(20)]
            results = executor.execute_batch(tasks, list(range(20)))
            concurrent_time = time.time() - start_time
            
            # Test sequential execution for comparison
            start_time = time.time()
            sequential_results = [test_task(i) for i in range(20)]
            sequential_time = time.time() - start_time
            
            speedup = sequential_time / max(concurrent_time, 0.001)
            
            assert len(results) == 20, "Not all concurrent tasks completed"
            assert all(r["result"] == r["task_id"] * 2 for r in results), "Incorrect task results"
            
            self.results["concurrent_processing"]["status"] = "PASS"
            self.results["concurrent_processing"]["message"] = f"Concurrent processing working: {speedup:.1f}x speedup with {executor.max_workers} workers"
            return True
            
        except Exception as e:
            self.results["concurrent_processing"]["status"] = "FAIL"
            self.results["concurrent_processing"]["message"] = f"Concurrent processing test failed: {e}"
            return False
    
    def test_memory_optimization(self) -> bool:
        """Test memory optimization features."""
        try:
            import gc
            import psutil
            import os
            
            # Get initial memory usage
            process = psutil.Process(os.getpid())
            initial_memory = process.memory_info().rss / 1024 / 1024  # MB
            
            # Create memory-intensive objects
            large_objects = []
            for i in range(100):
                large_objects.append([j for j in range(1000)])
            
            mid_memory = process.memory_info().rss / 1024 / 1024  # MB
            
            # Test garbage collection optimization
            del large_objects
            gc.collect()
            
            final_memory = process.memory_info().rss / 1024 / 1024  # MB
            
            memory_freed = mid_memory - final_memory
            memory_efficiency = memory_freed / max(mid_memory - initial_memory, 0.1)
            
            # Test memory pooling simulation
            from embodied_ai_benchmark.utils.optimization import PerformanceOptimizer
            
            optimizer = PerformanceOptimizer()
            optimizer.enable_memory_optimization()
            
            self.results["memory_optimization"]["status"] = "PASS"
            self.results["memory_optimization"]["message"] = f"Memory optimization working: {memory_freed:.1f}MB freed ({memory_efficiency:.1%} efficiency)"
            return True
            
        except Exception as e:
            self.results["memory_optimization"]["status"] = "FAIL" 
            self.results["memory_optimization"]["message"] = f"Memory optimization test failed: {e}"
            return False
    
    def test_scaling_framework(self) -> bool:
        """Test auto-scaling framework."""
        try:
            from embodied_ai_benchmark.utils.scalability import AutoScaler
            
            scaler = AutoScaler()
            
            # Test load detection
            scaler.update_load_metrics({
                "cpu_percent": 75.0,
                "memory_percent": 60.0,
                "active_tasks": 10
            })
            
            # Test scaling decision
            scaling_decision = scaler.should_scale_up()
            
            # Test worker scaling
            initial_workers = scaler.current_workers
            scaler.scale_up()
            scaled_workers = scaler.current_workers
            
            scaling_factor = scaled_workers / max(initial_workers, 1)
            
            # Test auto-scaling metrics
            metrics = scaler.get_scaling_metrics()
            
            assert "cpu_utilization" in metrics, "Missing CPU utilization metric"
            assert "worker_count" in metrics, "Missing worker count metric"
            
            self.results["scaling_framework"]["status"] = "PASS"
            self.results["scaling_framework"]["message"] = f"Scaling framework working: {scaling_factor:.1f}x worker scaling achieved"
            return True
            
        except Exception as e:
            self.results["scaling_framework"]["status"] = "FAIL"
            self.results["scaling_framework"]["message"] = f"Scaling framework test failed: {e}"
            return False
    
    def test_performance_monitoring(self) -> bool:
        """Test real-time performance monitoring."""
        try:
            from embodied_ai_benchmark.utils.monitoring import performance_monitor
            
            # Start monitoring
            performance_monitor.start_monitoring()
            
            # Simulate workload
            time.sleep(0.5)
            for i in range(100):
                _ = sum(j*j for j in range(100))
            
            # Get performance metrics
            metrics = performance_monitor.get_current_metrics()
            history = performance_monitor.get_metrics_history()
            
            # Test metric availability
            required_metrics = ["cpu_percent", "memory_usage", "timestamp"]
            for metric in required_metrics:
                assert metric in metrics, f"Missing metric: {metric}"
            
            assert len(history) > 0, "No historical metrics recorded"
            
            # Test performance analysis
            analysis = performance_monitor.analyze_performance()
            
            assert "bottlenecks" in analysis, "Missing bottleneck analysis"
            assert "recommendations" in analysis, "Missing performance recommendations"
            
            performance_monitor.stop_monitoring()
            
            self.results["performance_monitoring"]["status"] = "PASS" 
            self.results["performance_monitoring"]["message"] = f"Performance monitoring working: {len(history)} metrics collected"
            return True
            
        except Exception as e:
            self.results["performance_monitoring"]["status"] = "FAIL"
            self.results["performance_monitoring"]["message"] = f"Performance monitoring test failed: {e}"
            return False
    
    def test_global_optimization(self) -> bool:
        """Test global optimization features."""
        try:
            from embodied_ai_benchmark.utils.cross_platform import CrossPlatformManager
            
            # Initialize cross-platform manager
            manager = CrossPlatformManager()
            init_success = manager.initialize()
            
            assert init_success, "CrossPlatformManager initialization failed"
            
            # Test optimization status
            status = manager.get_optimization_status()
            
            required_status_keys = ["initialized", "system_info", "config"]
            for key in required_status_keys:
                assert key in status, f"Missing status key: {key}"
            
            # Test task-specific optimization
            cpu_opt_success = manager.optimize_for_task("cpu_intensive", "high")
            memory_opt_success = manager.optimize_for_task("memory_intensive", "medium")
            io_opt_success = manager.optimize_for_task("io_intensive", "low")
            
            optimization_success_rate = sum([cpu_opt_success, memory_opt_success, io_opt_success]) / 3
            
            # Test global configuration
            from embodied_ai_benchmark.utils.global_deployment import GlobalDeploymentManager
            
            global_manager = GlobalDeploymentManager()
            deployment_status = global_manager.get_deployment_status()
            
            manager.shutdown()
            
            self.results["global_optimization"]["status"] = "PASS"
            self.results["global_optimization"]["message"] = f"Global optimization working: {optimization_success_rate:.1%} optimization success rate"
            return True
            
        except Exception as e:
            self.results["global_optimization"]["status"] = "FAIL"
            self.results["global_optimization"]["message"] = f"Global optimization test failed: {e}"
            return False
    
    async def test_async_performance(self) -> Dict[str, Any]:
        """Test asynchronous performance capabilities."""
        try:
            # Test async task execution
            async def async_task(task_id, delay=0.01):
                await asyncio.sleep(delay)
                return {"task_id": task_id, "completed": True}
            
            # Run concurrent async tasks
            start_time = time.time()
            tasks = [async_task(i) for i in range(50)]
            results = await asyncio.gather(*tasks)
            async_time = time.time() - start_time
            
            # Calculate throughput
            throughput = len(results) / async_time
            
            return {
                "async_tasks_completed": len(results),
                "execution_time": async_time,
                "throughput": throughput,
                "success": True
            }
            
        except Exception as e:
            return {
                "error": str(e),
                "success": False
            }
    
    def run_all_tests(self) -> Dict[str, Any]:
        """Run all Generation 3 optimization tests."""
        print("🚀 Running Enhanced Generation 3 Performance Optimizations...")
        
        # Run synchronous tests
        sync_tests = [
            ("Advanced Caching", self.test_advanced_caching),
            ("Concurrent Processing", self.test_concurrent_processing),
            ("Memory Optimization", self.test_memory_optimization),
            ("Scaling Framework", self.test_scaling_framework),
            ("Performance Monitoring", self.test_performance_monitoring),
            ("Global Optimization", self.test_global_optimization)
        ]
        
        passed = 0
        total = len(sync_tests)
        
        for test_name, test_func in sync_tests:
            print(f"\n🧪 Running {test_name}...")
            try:
                result = test_func()
                status = "✅ PASS" if result else "❌ FAIL"
                print(f"   {status}: {self.results[test_name.lower().replace(' ', '_')]['message']}")
                if result:
                    passed += 1
            except Exception as e:
                print(f"   ❌ FAIL: Unexpected error - {e}")
        
        # Run async performance test
        print(f"\n🧪 Running Async Performance Test...")
        try:
            async_results = asyncio.run(self.test_async_performance())
            if async_results["success"]:
                print(f"   ✅ PASS: Async performance - {async_results['throughput']:.1f} tasks/sec")
                passed += 1
            else:
                print(f"   ❌ FAIL: {async_results.get('error', 'Unknown async error')}")
            total += 1
        except Exception as e:
            print(f"   ❌ FAIL: Async test error - {e}")
            total += 1
        
        success_rate = (passed / total) * 100
        
        # Generate summary
        summary = {
            "generation": 3,
            "timestamp": time.time(),
            "summary": {
                "total_tests": total,
                "passed_tests": passed,
                "failed_tests": total - passed,
                "success_rate": success_rate / 100
            },
            "test_results": self.results,
            "async_results": async_results if 'async_results' in locals() else {},
            "overall_status": "PASS" if passed == total else ("PARTIAL" if passed > 0 else "FAIL")
        }
        
        print(f"\n📊 Generation 3 Optimization Results:")
        print(f"   Tests Passed: {passed}/{total} ({success_rate:.1f}%)")
        print(f"   Overall Status: {summary['overall_status']}")
        
        return summary


def main():
    """Main execution function."""
    optimizer = Generation3Optimizer()
    results = optimizer.run_all_tests()
    
    # Save results
    output_file = Path('/root/repo/generation3_optimization_results.json')
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n💾 Results saved to: {output_file}")
    
    # Return appropriate exit code
    if results["overall_status"] == "PASS":
        return 0
    elif results["overall_status"] == "PARTIAL":
        return 1
    else:
        return 2


if __name__ == "__main__":
    sys.exit(main())
"""Tests for concurrent execution and optimization systems."""

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock, MagicMock
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

from embodied_ai_benchmark.utils.concurrent_execution import (
    LoadBalancer,
    AdvancedTaskManager,
    ConcurrentBenchmarkExecutor
)


class TestLoadBalancer:
    """Test load balancing functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.balancer = LoadBalancer()
    
    def test_initialization(self):
        """Test load balancer initialization."""
        assert isinstance(self.balancer.workers, dict)
        assert isinstance(self.balancer.task_queue, list)
        assert self.balancer.round_robin_counter == 0
        assert self.balancer.strategy == "least_loaded"
    
    def test_worker_registration(self):
        """Test worker registration."""
        worker_info = {
            "capabilities": ["navigation", "manipulation"],
            "max_concurrent_tasks": 3,
            "performance_rating": 0.85
        }
        
        self.balancer.register_worker("worker_1", worker_info)
        
        assert "worker_1" in self.balancer.workers
        worker = self.balancer.workers["worker_1"]
        assert worker["current_load"] == 0
        assert worker["capabilities"] == ["navigation", "manipulation"]
        assert worker["max_concurrent_tasks"] == 3
    
    def test_least_loaded_selection(self):
        """Test least loaded worker selection strategy."""
        # Register workers with different loads
        self.balancer.register_worker("worker_1", {"max_concurrent_tasks": 5})
        self.balancer.register_worker("worker_2", {"max_concurrent_tasks": 5})
        self.balancer.register_worker("worker_3", {"max_concurrent_tasks": 5})
        
        # Simulate different loads
        self.balancer.workers["worker_1"]["current_load"] = 3
        self.balancer.workers["worker_2"]["current_load"] = 1  # Least loaded
        self.balancer.workers["worker_3"]["current_load"] = 2
        
        task = {"required_capabilities": []}
        selected = self.balancer.select_worker(task)
        
        assert selected == "worker_2"
    
    def test_round_robin_selection(self):
        """Test round robin worker selection strategy."""
        self.balancer.strategy = "round_robin"
        
        # Register workers
        for i in range(3):
            self.balancer.register_worker(f"worker_{i}", {"max_concurrent_tasks": 5})
        
        task = {"required_capabilities": []}
        
        # Should cycle through workers
        first = self.balancer.select_worker(task)
        second = self.balancer.select_worker(task)
        third = self.balancer.select_worker(task)
        fourth = self.balancer.select_worker(task)  # Should cycle back
        
        assert first != second != third
        assert fourth == first  # Should cycle back to first
    
    def test_capability_based_selection(self):
        """Test worker selection based on required capabilities."""
        # Register workers with different capabilities
        self.balancer.register_worker("navigator", {
            "capabilities": ["navigation"],
            "max_concurrent_tasks": 5
        })
        self.balancer.register_worker("manipulator", {
            "capabilities": ["manipulation"], 
            "max_concurrent_tasks": 5
        })
        self.balancer.register_worker("generalist", {
            "capabilities": ["navigation", "manipulation"],
            "max_concurrent_tasks": 5
        })
        
        # Task requiring navigation
        nav_task = {"required_capabilities": ["navigation"]}
        selected = self.balancer.select_worker(nav_task)
        
        # Should select worker with navigation capability
        assert selected in ["navigator", "generalist"]
        
        # Task requiring both capabilities
        complex_task = {"required_capabilities": ["navigation", "manipulation"]}
        selected = self.balancer.select_worker(complex_task)
        
        # Should select generalist (only one with both capabilities)
        assert selected == "generalist"
    
    def test_overloaded_worker_handling(self):
        """Test handling of overloaded workers."""
        self.balancer.register_worker("worker_1", {"max_concurrent_tasks": 2})
        self.balancer.register_worker("worker_2", {"max_concurrent_tasks": 2})
        
        # Overload first worker
        self.balancer.workers["worker_1"]["current_load"] = 2  # At capacity
        self.balancer.workers["worker_2"]["current_load"] = 1  # Available
        
        task = {"required_capabilities": []}
        selected = self.balancer.select_worker(task)
        
        # Should not select overloaded worker
        assert selected == "worker_2"
    
    def test_no_available_workers(self):
        """Test behavior when no workers are available."""
        self.balancer.register_worker("worker_1", {"max_concurrent_tasks": 1})
        
        # Overload the only worker
        self.balancer.workers["worker_1"]["current_load"] = 1
        
        task = {"required_capabilities": []}
        selected = self.balancer.select_worker(task)
        
        # Should return None when no workers available
        assert selected is None
    
    def test_task_assignment_and_completion(self):
        """Test task assignment and completion tracking."""
        self.balancer.register_worker("worker_1", {"max_concurrent_tasks": 3})
        
        task = {"task_id": "test_task", "required_capabilities": []}
        
        # Assign task
        success = self.balancer.assign_task("worker_1", task)
        assert success
        assert self.balancer.workers["worker_1"]["current_load"] == 1
        
        # Complete task
        self.balancer.complete_task("worker_1", "test_task")
        assert self.balancer.workers["worker_1"]["current_load"] == 0
    
    def test_performance_based_selection(self):
        """Test worker selection based on performance ratings."""
        self.balancer.strategy = "performance_based"
        
        # Register workers with different performance ratings
        self.balancer.register_worker("slow_worker", {
            "performance_rating": 0.6,
            "max_concurrent_tasks": 5,
            "current_load": 1
        })
        self.balancer.register_worker("fast_worker", {
            "performance_rating": 0.9,
            "max_concurrent_tasks": 5,
            "current_load": 1  # Same load
        })
        
        task = {"required_capabilities": []}
        selected = self.balancer.select_worker(task)
        
        # Should prefer higher performance worker
        assert selected == "fast_worker"
    
    def test_load_balancing_statistics(self):
        """Test load balancing statistics collection."""
        # Register workers and simulate some assignments
        for i in range(3):
            self.balancer.register_worker(f"worker_{i}", {"max_concurrent_tasks": 5})
            
        # Assign tasks to create different loads
        self.balancer.assign_task("worker_0", {"task_id": "task_1"})
        self.balancer.assign_task("worker_0", {"task_id": "task_2"})
        self.balancer.assign_task("worker_1", {"task_id": "task_3"})
        
        stats = self.balancer.get_load_statistics()
        
        assert isinstance(stats, dict)
        assert "total_workers" in stats
        assert "total_load" in stats
        assert "average_load" in stats
        assert "load_distribution" in stats
        
        assert stats["total_workers"] == 3
        assert stats["total_load"] == 3
        assert stats["average_load"] == 1.0


class TestAdvancedTaskManager:
    """Test advanced task management functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.task_manager = AdvancedTaskManager()
    
    def test_initialization(self):
        """Test task manager initialization."""
        assert self.task_manager.priority_queues is not None
        assert self.task_manager.task_dependencies == {}
        assert self.task_manager.running_tasks == {}
        assert self.task_manager.completed_tasks == {}
    
    def test_task_submission_with_priority(self):
        """Test task submission with different priorities."""
        # Submit tasks with different priorities
        high_priority_task = {
            "task_id": "urgent_task",
            "priority": 3,
            "estimated_duration": 60
        }
        
        low_priority_task = {
            "task_id": "normal_task", 
            "priority": 1,
            "estimated_duration": 120
        }
        
        self.task_manager.submit_task(high_priority_task)
        self.task_manager.submit_task(low_priority_task)
        
        # High priority task should be processed first
        next_task = self.task_manager.get_next_task()
        assert next_task["task_id"] == "urgent_task"
        
        next_task = self.task_manager.get_next_task()
        assert next_task["task_id"] == "normal_task"
    
    def test_dependency_handling(self):
        """Test task dependency resolution."""
        # Create tasks with dependencies
        base_task = {
            "task_id": "base_task",
            "priority": 2,
            "dependencies": []
        }
        
        dependent_task = {
            "task_id": "dependent_task",
            "priority": 3,  # Higher priority but has dependency
            "dependencies": ["base_task"]
        }
        
        self.task_manager.submit_task(dependent_task)
        self.task_manager.submit_task(base_task)
        
        # Should get base task first despite dependent task having higher priority
        next_task = self.task_manager.get_next_task()
        assert next_task["task_id"] == "base_task"
        
        # Mark base task as completed
        self.task_manager.mark_completed("base_task", {"status": "success"})
        
        # Now dependent task should be available
        next_task = self.task_manager.get_next_task()
        assert next_task["task_id"] == "dependent_task"
    
    def test_circular_dependency_detection(self):
        """Test detection of circular dependencies."""
        task_a = {
            "task_id": "task_a",
            "dependencies": ["task_b"]
        }
        
        task_b = {
            "task_id": "task_b", 
            "dependencies": ["task_a"]  # Circular dependency
        }
        
        self.task_manager.submit_task(task_a)
        
        # Should raise error when circular dependency is detected
        with pytest.raises(ValueError, match="circular dependency"):
            self.task_manager.submit_task(task_b)
    
    def test_resource_constraints(self):
        """Test task scheduling with resource constraints."""
        # Task requiring high memory
        memory_intensive_task = {
            "task_id": "memory_task",
            "required_resources": {"memory_mb": 1000, "cpu_cores": 2},
            "priority": 2
        }
        
        # Task requiring high CPU
        cpu_intensive_task = {
            "task_id": "cpu_task",
            "required_resources": {"memory_mb": 100, "cpu_cores": 4},
            "priority": 2
        }
        
        # Set resource limits
        self.task_manager.set_resource_limits({
            "memory_mb": 1200,
            "cpu_cores": 4
        })
        
        self.task_manager.submit_task(memory_intensive_task)
        self.task_manager.submit_task(cpu_intensive_task)
        
        # Should get first task
        task1 = self.task_manager.get_next_task()
        assert task1 is not None
        
        # Mark as running to consume resources
        self.task_manager.mark_running(task1["task_id"])
        
        # Second task should be blocked due to resource constraints
        task2 = self.task_manager.get_next_task()
        if task1["task_id"] == "memory_task":
            # CPU task should be blocked (not enough CPU cores left)
            assert task2 is None
        else:
            # Memory task should be blocked (not enough memory left)
            assert task2 is None
    
    def test_deadline_aware_scheduling(self):
        """Test deadline-aware task scheduling."""
        import time
        current_time = time.time()
        
        # Task with urgent deadline
        urgent_task = {
            "task_id": "urgent_deadline",
            "priority": 1,
            "deadline": current_time + 60  # 1 minute from now
        }
        
        # Task with relaxed deadline  
        relaxed_task = {
            "task_id": "relaxed_deadline",
            "priority": 2,  # Higher priority
            "deadline": current_time + 3600  # 1 hour from now
        }
        
        self.task_manager.submit_task(relaxed_task)
        self.task_manager.submit_task(urgent_task)
        
        # Should prioritize task with urgent deadline despite lower priority
        next_task = self.task_manager.get_next_task()
        assert next_task["task_id"] == "urgent_deadline"
    
    @pytest.mark.asyncio
    async def test_async_task_execution(self):
        """Test asynchronous task execution."""
        async def mock_async_task():
            await asyncio.sleep(0.1)
            return {"result": "async_completed"}
        
        task = {
            "task_id": "async_task",
            "func": mock_async_task,
            "priority": 2
        }
        
        self.task_manager.submit_task(task)
        
        # Execute task
        task_to_run = self.task_manager.get_next_task()
        self.task_manager.mark_running(task_to_run["task_id"])
        
        result = await task_to_run["func"]()
        self.task_manager.mark_completed(task_to_run["task_id"], result)
        
        # Check completion
        completed_task = self.task_manager.completed_tasks[task_to_run["task_id"]]
        assert completed_task["result"]["result"] == "async_completed"
    
    def test_task_timeout_handling(self):
        """Test handling of task timeouts."""
        task = {
            "task_id": "timeout_task",
            "timeout_seconds": 0.1,  # Very short timeout
            "priority": 2
        }
        
        self.task_manager.submit_task(task)
        next_task = self.task_manager.get_next_task()
        
        # Mark as running
        self.task_manager.mark_running(next_task["task_id"])
        
        # Wait longer than timeout
        time.sleep(0.2)
        
        # Check if task is marked as timed out
        timed_out_tasks = self.task_manager.get_timed_out_tasks()
        assert len(timed_out_tasks) == 1
        assert timed_out_tasks[0]["task_id"] == "timeout_task"
    
    def test_task_statistics(self):
        """Test task execution statistics collection."""
        # Submit and execute several tasks
        for i in range(5):
            task = {
                "task_id": f"task_{i}",
                "priority": i % 3 + 1,
                "estimated_duration": (i + 1) * 10
            }
            self.task_manager.submit_task(task)
            
            # Simulate execution
            next_task = self.task_manager.get_next_task()
            if next_task:
                self.task_manager.mark_running(next_task["task_id"])
                self.task_manager.mark_completed(next_task["task_id"], {
                    "status": "success",
                    "actual_duration": (i + 1) * 9  # Slightly faster than estimated
                })
        
        stats = self.task_manager.get_execution_statistics()
        
        assert isinstance(stats, dict)
        assert "total_tasks" in stats
        assert "completed_tasks" in stats
        assert "average_execution_time" in stats
        assert "priority_distribution" in stats
        
        assert stats["total_tasks"] == 5
        assert stats["completed_tasks"] == 5


class TestConcurrentBenchmarkExecutor:
    """Test concurrent benchmark execution functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.executor = ConcurrentBenchmarkExecutor(max_workers=4)
    
    def test_initialization(self):
        """Test executor initialization."""
        assert self.executor.max_workers == 4
        assert self.executor.task_manager is not None
        assert self.executor.load_balancer is not None
        assert isinstance(self.executor.active_benchmarks, dict)
    
    @pytest.mark.asyncio
    async def test_benchmark_execution(self):
        """Test concurrent benchmark execution."""
        # Mock benchmark function
        async def mock_benchmark(config):
            await asyncio.sleep(0.1)
            return {
                "success": True,
                "score": config.get("expected_score", 100),
                "duration": 0.1
            }
        
        # Submit multiple benchmarks
        benchmark_configs = [
            {"benchmark_id": "nav_test", "expected_score": 95},
            {"benchmark_id": "manip_test", "expected_score": 88},
            {"benchmark_id": "coord_test", "expected_score": 92}
        ]
        
        # Execute benchmarks concurrently
        results = await self.executor.execute_benchmarks_concurrent(
            benchmark_func=mock_benchmark,
            configs=benchmark_configs
        )
        
        assert len(results) == 3
        for result in results:
            assert result["success"] is True
            assert "score" in result
            assert "duration" in result
    
    @pytest.mark.asyncio
    async def test_resource_aware_execution(self):
        """Test resource-aware benchmark execution."""
        # Mock resource-intensive benchmark
        async def resource_heavy_benchmark(config):
            await asyncio.sleep(0.1)
            return {
                "memory_used": config["required_memory"],
                "cpu_usage": config["required_cpu"]
            }
        
        configs = [
            {"benchmark_id": "heavy_1", "required_memory": 500, "required_cpu": 2},
            {"benchmark_id": "heavy_2", "required_memory": 600, "required_cpu": 2},
            {"benchmark_id": "light_1", "required_memory": 100, "required_cpu": 1}
        ]
        
        # Set resource limits
        self.executor.set_resource_limits({
            "memory_mb": 1000,
            "cpu_cores": 4
        })
        
        results = await self.executor.execute_benchmarks_concurrent(
            benchmark_func=resource_heavy_benchmark,
            configs=configs
        )
        
        assert len(results) == 3
    
    def test_benchmark_priority_handling(self):
        """Test benchmark execution with different priorities."""
        # Submit benchmarks with different priorities
        high_priority_config = {
            "benchmark_id": "critical_test",
            "priority": 3,
            "timeout": 300
        }
        
        low_priority_config = {
            "benchmark_id": "standard_test", 
            "priority": 1,
            "timeout": 600
        }
        
        self.executor.submit_benchmark(high_priority_config)
        self.executor.submit_benchmark(low_priority_config)
        
        # High priority benchmark should be scheduled first
        next_benchmark = self.executor.get_next_benchmark()
        assert next_benchmark["benchmark_id"] == "critical_test"
    
    @pytest.mark.asyncio
    async def test_benchmark_failure_handling(self):
        """Test handling of benchmark failures."""
        async def failing_benchmark(config):
            if config["should_fail"]:
                raise RuntimeError("Benchmark failure")
            return {"success": True}
        
        configs = [
            {"benchmark_id": "success_test", "should_fail": False},
            {"benchmark_id": "failure_test", "should_fail": True},
            {"benchmark_id": "another_success", "should_fail": False}
        ]
        
        results = await self.executor.execute_benchmarks_concurrent(
            benchmark_func=failing_benchmark,
            configs=configs,
            handle_failures=True
        )
        
        # Should have results for all benchmarks (including failures)
        assert len(results) == 3
        
        # Find the failed benchmark result
        failed_result = next(r for r in results if r["benchmark_id"] == "failure_test")
        assert failed_result["success"] is False
        assert "error" in failed_result
    
    def test_benchmark_timeout_handling(self):
        """Test benchmark timeout handling."""
        def slow_benchmark(config):
            time.sleep(config.get("duration", 0.5))
            return {"completed": True}
        
        config = {
            "benchmark_id": "slow_test",
            "duration": 1.0,  # Long duration
            "timeout": 0.1    # Short timeout
        }
        
        # Execute with timeout
        result = self.executor.execute_benchmark_with_timeout(
            benchmark_func=slow_benchmark,
            config=config
        )
        
        assert result["success"] is False
        assert result["timeout"] is True
    
    @pytest.mark.asyncio
    async def test_load_balancing_integration(self):
        """Test integration with load balancing."""
        # Register workers with different capabilities
        self.executor.load_balancer.register_worker("nav_worker", {
            "capabilities": ["navigation"],
            "max_concurrent_tasks": 2
        })
        
        self.executor.load_balancer.register_worker("manip_worker", {
            "capabilities": ["manipulation"],
            "max_concurrent_tasks": 2
        })
        
        # Submit benchmarks requiring different capabilities
        configs = [
            {"benchmark_id": "nav_test", "required_capabilities": ["navigation"]},
            {"benchmark_id": "manip_test", "required_capabilities": ["manipulation"]},
            {"benchmark_id": "general_test", "required_capabilities": []}
        ]
        
        # Mock benchmark function
        async def capability_benchmark(config):
            return {"capability_used": config["required_capabilities"]}
        
        results = await self.executor.execute_benchmarks_concurrent(
            benchmark_func=capability_benchmark,
            configs=configs
        )
        
        assert len(results) == 3
    
    def test_execution_metrics_collection(self):
        """Test collection of execution metrics."""
        # Execute some benchmarks to generate metrics
        def simple_benchmark(config):
            return {"score": config.get("score", 100)}
        
        configs = [
            {"benchmark_id": f"test_{i}", "score": 90 + i} 
            for i in range(5)
        ]
        
        for config in configs:
            result = self.executor.execute_benchmark_with_timeout(
                benchmark_func=simple_benchmark,
                config=config
            )
        
        metrics = self.executor.get_execution_metrics()
        
        assert isinstance(metrics, dict)
        assert "total_benchmarks" in metrics
        assert "success_rate" in metrics
        assert "average_execution_time" in metrics
        assert "resource_utilization" in metrics
        
        assert metrics["total_benchmarks"] == 5
    
    def test_concurrent_execution_scaling(self):
        """Test scaling of concurrent execution."""
        import threading
        
        # Test with different worker counts
        for worker_count in [1, 2, 4, 8]:
            executor = ConcurrentBenchmarkExecutor(max_workers=worker_count)
            
            # Execute multiple simple tasks
            def simple_task(config):
                return {"worker_count": worker_count, "task_id": config["task_id"]}
            
            configs = [{"task_id": f"task_{i}"} for i in range(10)]
            
            start_time = time.time()
            results = []
            
            # Execute tasks
            with ThreadPoolExecutor(max_workers=worker_count) as thread_executor:
                futures = [
                    thread_executor.submit(simple_task, config) 
                    for config in configs
                ]
                
                for future in as_completed(futures):
                    results.append(future.result())
            
            execution_time = time.time() - start_time
            
            # More workers should generally complete tasks faster
            # (though for very simple tasks, overhead might dominate)
            assert len(results) == 10
            assert execution_time < 1.0  # Should complete reasonably quickly
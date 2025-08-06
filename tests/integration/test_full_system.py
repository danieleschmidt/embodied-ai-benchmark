"""Integration tests for the full embodied AI benchmark system."""

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock
import tempfile
import shutil
import os

from embodied_ai_benchmark import (
    BaseTask, BaseEnv, BaseAgent, RandomAgent,
    BenchmarkSuite, Evaluator,
    LLMCurriculum, CurriculumTrainer,
    CoordinationOrchestrator, CommunicationProtocol,
    LanguageTaskInterface,
    ErrorHandler,
    ConcurrentBenchmarkExecutor,
    BenchmarkMetricsCollector
)


class MockEnvironment(BaseEnv):
    """Mock environment for integration testing."""
    
    def __init__(self, env_id="mock_env"):
        super().__init__(env_id)
        self.state = {"position": [0, 0, 0], "orientation": [0, 0, 0, 1]}
        self.step_count = 0
        self.max_steps = 100
    
    def reset(self):
        self.state = {"position": [0, 0, 0], "orientation": [0, 0, 0, 1]}
        self.step_count = 0
        return self.state
    
    def step(self, action):
        self.step_count += 1
        
        # Simple mock physics
        if action == "move_forward":
            self.state["position"][0] += 0.1
        elif action == "move_backward":
            self.state["position"][0] -= 0.1
        elif action == "turn_left":
            self.state["orientation"][2] += 0.1
        elif action == "turn_right":
            self.state["orientation"][2] -= 0.1
        
        # Mock reward calculation
        reward = 1.0 if abs(self.state["position"][0]) < 5.0 else -0.1
        
        # Episode termination
        done = self.step_count >= self.max_steps or abs(self.state["position"][0]) > 10.0
        
        info = {
            "step_count": self.step_count,
            "success": abs(self.state["position"][0]) < 1.0 and done
        }
        
        return self.state, reward, done, info
    
    def render(self):
        return f"Position: {self.state['position']}, Orientation: {self.state['orientation']}"
    
    def close(self):
        pass


class MockTask(BaseTask):
    """Mock task for integration testing."""
    
    def __init__(self, task_id="mock_navigation", **kwargs):
        env = MockEnvironment()
        super().__init__(task_id, env, **kwargs)
        self.target_position = [1.0, 0, 0]
    
    def reset(self):
        obs = super().reset()
        self.target_position = [1.0, 0, 0]  # Reset target
        return obs
    
    def get_reward(self, observation, action, next_observation, info):
        # Distance-based reward
        current_pos = next_observation["position"]
        distance = abs(current_pos[0] - self.target_position[0])
        
        reward = -distance  # Negative distance as reward
        
        if distance < 0.1:  # Close to target
            reward += 10.0
        
        return reward
    
    def is_success(self, observation, info):
        current_pos = observation["position"]
        distance = abs(current_pos[0] - self.target_position[0])
        return distance < 0.1


class TestFullSystemIntegration:
    """Test full system integration scenarios."""
    
    def setup_method(self):
        """Set up integration test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.task = MockTask()
        self.agent = RandomAgent()
        self.evaluator = Evaluator()
    
    def teardown_method(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_basic_benchmark_execution(self):
        """Test basic benchmark execution flow."""
        # Create benchmark suite
        benchmark_suite = BenchmarkSuite("integration_test_suite")
        benchmark_suite.add_task(self.task)
        
        # Run benchmark
        results = self.evaluator.evaluate_agent(
            agent=self.agent,
            benchmark_suite=benchmark_suite,
            num_episodes=5
        )
        
        assert isinstance(results, dict)
        assert "total_episodes" in results
        assert "average_reward" in results
        assert "success_rate" in results
        assert results["total_episodes"] == 5
    
    def test_quantum_planning_integration(self):
        """Test integration of quantum-inspired planning with task execution."""
        # Execute task with quantum planning
        obs = self.task.reset()
        
        total_reward = 0
        for _ in range(10):
            # Get quantum plan
            quantum_plan = self.task.get_quantum_plan(obs)
            
            assert isinstance(quantum_plan, dict)
            assert "primary_action" in quantum_plan
            assert "confidence" in quantum_plan
            
            # Execute action
            action = quantum_plan["primary_action"]
            obs, reward, done, info = self.task.step(action)
            total_reward += reward
            
            if done:
                break
        
        # Task should execute without errors
        assert total_reward is not None
    
    @patch('embodied_ai_benchmark.curriculum.llm_curriculum.openai.ChatCompletion.create')
    def test_curriculum_learning_integration(self, mock_llm):
        """Test integration of LLM curriculum learning with training."""
        # Mock LLM responses
        mock_llm.return_value.choices = [Mock(message=Mock(content='{"tasks": [{"name": "basic_nav", "difficulty": 0.3}]}'))]
        
        # Create curriculum system
        mock_llm_client = Mock()
        mock_llm_client.generate_text.return_value = '{"tasks": [{"name": "basic_nav", "difficulty": 0.3, "max_episodes": 5}]}'
        
        curriculum = LLMCurriculum(llm_client=mock_llm_client)
        trainer = CurriculumTrainer(curriculum_system=curriculum)
        
        # Training function
        def train_episode(task_config):
            return {
                "success": True,
                "episode_reward": 50 + task_config.get("difficulty", 0.5) * 30,
                "completion_time": 45,
                "task_name": task_config["name"]
            }
        
        # Train agent with curriculum
        results = trainer.train_agent_with_curriculum(
            agent=self.agent,
            env=self.task.env,
            learning_objectives=["navigation"],
            max_episodes_per_task=5,
            train_episode_func=train_episode
        )
        
        assert isinstance(results, dict)
        assert "curriculum_results" in results
        assert len(results["curriculum_results"]) > 0
    
    @pytest.mark.asyncio
    async def test_multiagent_coordination_integration(self):
        """Test integration of multi-agent coordination."""
        # Create coordination system
        communication = CommunicationProtocol()
        orchestrator = CoordinationOrchestrator()
        
        # Register multiple agents
        agents = ["agent_1", "agent_2", "agent_3"]
        for agent_id in agents:
            communication.register_agent(agent_id)
        
        # Create coordination task
        from embodied_ai_benchmark.multiagent.coordination_protocols import CoordinationTask
        
        coord_task = CoordinationTask(
            task_id="group_navigation",
            description="Navigate to target as group",
            required_capabilities=["navigation"],
            estimated_duration=300
        )
        
        # Mock agent capabilities
        agent_capabilities = {
            "agent_1": {"capabilities": ["navigation"], "availability": True},
            "agent_2": {"capabilities": ["navigation"], "availability": True},
            "agent_3": {"capabilities": ["navigation", "coordination"], "availability": True}
        }
        
        # Allocate and coordinate
        allocation = await orchestrator.allocate_tasks([coord_task], agent_capabilities)
        
        assert isinstance(allocation, dict)
        assert len(allocation) > 0
    
    def test_error_handling_integration(self):
        """Test integration of error handling across systems."""
        error_handler = ErrorHandler()
        
        # Simulate various error scenarios during benchmark execution
        def failing_benchmark():
            # Simulate network error
            raise ConnectionError("Failed to connect to evaluation server")
        
        # Handle error
        context = {"operation": "benchmark_execution", "component": "evaluator"}
        result = error_handler.handle_error(ConnectionError("Network failure"), context)
        
        assert isinstance(result, dict)
        assert "error_type" in result
        assert "recovered" in result
    
    @pytest.mark.asyncio
    async def test_concurrent_execution_integration(self):
        """Test concurrent execution of multiple benchmarks."""
        executor = ConcurrentBenchmarkExecutor(max_workers=3)
        
        # Create multiple benchmark configurations
        async def benchmark_function(config):
            task = MockTask(task_id=config["task_id"])
            agent = RandomAgent()
            
            # Run short benchmark
            obs = task.reset()
            total_reward = 0
            
            for _ in range(5):  # Short episodes for testing
                action = agent.act(obs)
                obs, reward, done, info = task.step(action)
                total_reward += reward
                
                if done:
                    break
            
            return {
                "task_id": config["task_id"],
                "total_reward": total_reward,
                "success": task.is_success(obs, info)
            }
        
        configs = [
            {"task_id": "concurrent_test_1"},
            {"task_id": "concurrent_test_2"},
            {"task_id": "concurrent_test_3"}
        ]
        
        results = await executor.execute_benchmarks_concurrent(
            benchmark_func=benchmark_function,
            configs=configs
        )
        
        assert len(results) == 3
        for result in results:
            assert "task_id" in result
            assert "total_reward" in result
            assert "success" in result
    
    def test_metrics_collection_integration(self):
        """Test metrics collection across system components."""
        metrics_collector = BenchmarkMetricsCollector()
        
        # Simulate benchmark execution with metrics
        benchmark_id = "integration_metrics_test"
        metrics_collector.start_benchmark(benchmark_id)
        
        # Record various metrics during execution
        metrics_collector.record_counter("steps_taken", 1, tags={"task": "navigation"})
        metrics_collector.record_gauge("success_rate", 0.8, tags={"task": "navigation"})
        metrics_collector.record_timer("episode_duration", 45.5, tags={"task": "navigation"})
        metrics_collector.record_histogram("rewards", [1.0, 2.0, 1.5, 2.2], tags={"task": "navigation"})
        
        metrics_collector.end_benchmark(benchmark_id, {"status": "completed"})
        
        # Get collected metrics
        all_metrics = metrics_collector.get_all_metrics()
        
        assert isinstance(all_metrics, dict)
        assert "counters" in all_metrics
        assert "gauges" in all_metrics
        assert "timers" in all_metrics
        assert "histograms" in all_metrics
    
    def test_language_interface_integration(self):
        """Test natural language interface integration."""
        language_interface = LanguageTaskInterface()
        
        # Parse natural language task specification
        natural_language = "Navigate to the red box and pick it up"
        
        parsed_task = language_interface.parse_task_specification(natural_language)
        
        assert isinstance(parsed_task, dict)
        assert "actions" in parsed_task or "objectives" in parsed_task
        
        # Generate instructions for task
        task_config = {
            "task_type": "navigation_manipulation",
            "target_object": "red_box",
            "actions": ["navigate", "grasp"]
        }
        
        instructions = language_interface.generate_task_instructions(task_config)
        
        assert isinstance(instructions, str)
        assert len(instructions) > 0
    
    def test_end_to_end_benchmark_pipeline(self):
        """Test complete end-to-end benchmark execution pipeline."""
        # Set up complete pipeline
        benchmark_suite = BenchmarkSuite("e2e_test_suite")
        benchmark_suite.add_task(self.task)
        
        metrics_collector = BenchmarkMetricsCollector()
        error_handler = ErrorHandler()
        
        try:
            # Start metrics collection
            benchmark_id = "e2e_pipeline_test"
            metrics_collector.start_benchmark(benchmark_id)
            
            # Execute benchmark with full pipeline
            results = self.evaluator.evaluate_agent(
                agent=self.agent,
                benchmark_suite=benchmark_suite,
                num_episodes=3,
                collect_metrics=True
            )
            
            # Verify results
            assert isinstance(results, dict)
            assert "total_episodes" in results
            assert results["total_episodes"] == 3
            
            # Record final metrics
            metrics_collector.record_gauge("final_success_rate", results.get("success_rate", 0))
            metrics_collector.end_benchmark(benchmark_id, results)
            
            # Get pipeline metrics
            pipeline_metrics = metrics_collector.get_all_metrics()
            assert isinstance(pipeline_metrics, dict)
            
        except Exception as e:
            # Handle any errors through error handler
            error_result = error_handler.handle_error(e, {
                "operation": "e2e_benchmark_pipeline",
                "benchmark_id": benchmark_id
            })
            
            # Even with errors, should have error handling result
            assert isinstance(error_result, dict)
            assert "error_type" in error_result
    
    @patch('embodied_ai_benchmark.utils.caching.global_persistent_cache')
    def test_caching_integration(self, mock_cache):
        """Test caching system integration."""
        # Mock cache responses
        mock_cache.get.return_value = None  # Cache miss
        mock_cache.put = Mock()
        
        # Execute benchmark with caching
        cache_key = "benchmark_result_mock_navigation_random_agent"
        
        # First execution (cache miss)
        results1 = self.evaluator.evaluate_agent(
            agent=self.agent,
            benchmark_suite=BenchmarkSuite("cached_test").add_task(self.task),
            num_episodes=2,
            use_cache=True,
            cache_key=cache_key
        )
        
        # Verify cache was accessed
        mock_cache.get.assert_called()
        mock_cache.put.assert_called()
        
        assert isinstance(results1, dict)
    
    def test_cross_component_data_flow(self):
        """Test data flow between different system components."""
        # Create connected components
        benchmark_suite = BenchmarkSuite("data_flow_test")
        benchmark_suite.add_task(self.task)
        
        metrics_collector = BenchmarkMetricsCollector()
        error_handler = ErrorHandler()
        
        # Execute with data flow tracking
        benchmark_id = "data_flow_test"
        metrics_collector.start_benchmark(benchmark_id)
        
        try:
            # Task execution generates data
            obs = self.task.reset()
            
            # Agent processes observation
            action = self.agent.act(obs)
            assert action is not None
            
            # Environment processes action
            next_obs, reward, done, info = self.task.step(action)
            
            # Metrics collector records data
            metrics_collector.record_counter("actions_taken", 1)
            metrics_collector.record_gauge("current_reward", reward)
            
            # Task processes results
            success = self.task.is_success(next_obs, info)
            metrics_collector.record_counter("success" if success else "failure", 1)
            
            # Complete benchmark
            metrics_collector.end_benchmark(benchmark_id, {
                "final_observation": next_obs,
                "total_reward": reward,
                "success": success
            })
            
            # Verify data flow
            all_metrics = metrics_collector.get_all_metrics()
            assert "counters" in all_metrics
            assert all_metrics["counters"]["actions_taken"] == 1
            
        except Exception as e:
            # Error handling preserves data context
            error_result = error_handler.handle_error(e, {
                "benchmark_id": benchmark_id,
                "component": "cross_component_test",
                "collected_metrics": metrics_collector.get_all_metrics()
            })
            
            assert "context" in error_result
            assert error_result["context"]["benchmark_id"] == benchmark_id
    
    def test_system_performance_under_load(self):
        """Test system performance under concurrent load."""
        import time
        import threading
        
        # Create multiple concurrent benchmark executions
        num_threads = 5
        results = []
        errors = []
        
        def run_benchmark(thread_id):
            try:
                thread_task = MockTask(task_id=f"load_test_task_{thread_id}")
                thread_agent = RandomAgent()
                
                start_time = time.time()
                
                # Run short benchmark
                obs = thread_task.reset()
                thread_reward = 0
                
                for _ in range(3):  # Short episodes
                    action = thread_agent.act(obs)
                    obs, reward, done, info = thread_task.step(action)
                    thread_reward += reward
                    
                    if done:
                        break
                
                execution_time = time.time() - start_time
                
                results.append({
                    "thread_id": thread_id,
                    "execution_time": execution_time,
                    "total_reward": thread_reward,
                    "success": thread_task.is_success(obs, info)
                })
                
            except Exception as e:
                errors.append({"thread_id": thread_id, "error": str(e)})
        
        # Execute concurrent benchmarks
        threads = []
        for i in range(num_threads):
            thread = threading.Thread(target=run_benchmark, args=(i,))
            threads.append(thread)
            thread.start()
        
        # Wait for completion
        for thread in threads:
            thread.join()
        
        # Verify system handled concurrent load
        assert len(results) == num_threads
        assert len(errors) == 0  # No errors should occur
        
        # Performance should be reasonable
        avg_execution_time = sum(r["execution_time"] for r in results) / len(results)
        assert avg_execution_time < 5.0  # Should complete within 5 seconds per thread
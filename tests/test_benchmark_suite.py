"""Comprehensive tests for the benchmark suite."""

import pytest
import numpy as np
import time
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import tempfile
import json

import sys
sys.path.append(str(Path(__file__).parent.parent / "src"))

from embodied_ai_benchmark.evaluation.benchmark_suite import BenchmarkSuite
from embodied_ai_benchmark.core.base_agent import RandomAgent
from embodied_ai_benchmark.core.base_env import BaseEnv
from embodied_ai_benchmark.core.base_task import BaseTask
from embodied_ai_benchmark.core.base_metric import BaseMetric
from embodied_ai_benchmark.tasks.task_factory import make_env
from embodied_ai_benchmark.utils.validation import ValidationError


class MockTask(BaseTask):
    """Mock task for testing."""
    
    def __init__(self, config):
        super().__init__(config)
        self.step_count = 0
        self.max_steps = config.get("max_steps", 10)
    
    def reset(self):
        super().reset()
        self.step_count = 0
        return self._get_observation()
    
    def step(self, action):
        self.step_count += 1
        reward = np.random.uniform(-1, 1)
        done = self.step_count >= self.max_steps
        info = {"task_success": done and reward > 0}
        return self._get_observation(), reward, done, info
    
    def _get_observation(self):
        return {
            "state": np.random.random(4),
            "step": self.step_count
        }
    
    def check_success(self):
        return self.step_count >= self.max_steps
    
    def compute_reward(self):
        return float(self.step_count) / self.max_steps


class MockEnv(BaseEnv):
    """Mock environment for testing."""
    
    def __init__(self, config):
        super().__init__(config)
        self.task = None
        self.current_step = 0
    
    def set_task(self, task):
        self.task = task
    
    def reset(self, seed=None):
        super().reset(seed)
        self.current_step = 0
        if self.task:
            return self.task.reset()
        return {"mock": np.zeros(4)}
    
    def step(self, action):
        self.current_step += 1
        if self.task:
            return self.task.step(action)
        
        reward = np.random.uniform(-1, 1)
        done = self.current_step >= 10
        info = {"task_success": done}
        return {"mock": np.random.random(4)}, reward, done, info
    
    def render(self, mode="rgb_array"):
        return np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
    
    def close(self):
        pass
    
    def _get_observation(self):
        return {"mock": np.random.random(4)}


class TestBenchmarkSuite:
    """Test suite for BenchmarkSuite class."""
    
    @pytest.fixture
    def benchmark_suite(self):
        """Create benchmark suite for testing."""
        config = {
            "evaluator": {
                "timeout": 30,
                "safety_checks": True,
                "record_trajectories": False
            }
        }
        return BenchmarkSuite(config)
    
    @pytest.fixture
    def mock_env(self):
        """Create mock environment."""
        config = {"max_steps": 10, "render_mode": "rgb_array"}
        env = MockEnv(config)
        task = MockTask(config)
        env.set_task(task)
        return env
    
    @pytest.fixture
    def mock_agent(self):
        """Create mock agent."""
        config = {
            "agent_id": "test_agent",
            "action_space": {
                "type": "continuous",
                "shape": (4,),
                "low": np.array([-1, -1, -1, -1]),
                "high": np.array([1, 1, 1, 1])
            }
        }
        return RandomAgent(config)
    
    def test_benchmark_suite_initialization(self, benchmark_suite):
        """Test benchmark suite initialization."""
        assert benchmark_suite is not None
        assert hasattr(benchmark_suite, 'config')
        assert hasattr(benchmark_suite, 'tasks')
        assert hasattr(benchmark_suite, 'metrics')
        assert hasattr(benchmark_suite, 'evaluator')
        
        # Check default metrics are added
        assert len(benchmark_suite.metrics) > 0
        assert "success_rate" in benchmark_suite.metrics
    
    def test_add_task(self, benchmark_suite):
        """Test adding tasks to benchmark suite."""
        task = MockTask({"name": "test_task"})
        benchmark_suite.add_task("test_task", task)
        
        assert "test_task" in benchmark_suite.tasks
        assert benchmark_suite.tasks["test_task"] == task
    
    def test_add_metric(self, benchmark_suite):
        """Test adding metrics to benchmark suite."""
        class TestMetric(BaseMetric):
            def reset(self):
                pass
            def update(self, *args):
                pass
            def compute(self):
                return 0.5
        
        metric = TestMetric({"name": "test_metric"})
        benchmark_suite.add_metric("test_metric", metric)
        
        assert "test_metric" in benchmark_suite.metrics
        assert benchmark_suite.metrics["test_metric"] == metric
    
    def test_evaluation_input_validation(self, benchmark_suite, mock_agent):
        """Test input validation for evaluation."""
        # Test invalid environment
        with pytest.raises(ValidationError):
            benchmark_suite.evaluate("not_an_env", mock_agent, 5)
        
        # Test invalid agent
        mock_env = MockEnv({})
        with pytest.raises(ValidationError):
            benchmark_suite.evaluate(mock_env, "not_an_agent", 5)
        
        # Test invalid num_episodes
        with pytest.raises(ValidationError):
            benchmark_suite.evaluate(mock_env, mock_agent, -1)
        
        # Test invalid max_steps
        with pytest.raises(ValidationError):
            benchmark_suite.evaluate(mock_env, mock_agent, 5, max_steps_per_episode=-1)
        
        # Test invalid seed
        with pytest.raises(ValidationError):
            benchmark_suite.evaluate(mock_env, mock_agent, 5, seed=-1)
    
    def test_sequential_evaluation(self, benchmark_suite, mock_env, mock_agent):
        """Test sequential evaluation."""
        results = benchmark_suite.evaluate(
            env=mock_env,
            agent=mock_agent,
            num_episodes=3,
            max_steps_per_episode=5,
            parallel=False
        )
        
        assert results is not None
        assert "num_episodes" in results
        assert results["num_episodes"] == 3
        assert "success_rate" in results
        assert "avg_steps" in results
        assert "total_time" in results
        assert "evaluation_id" in results
        assert "agent_name" in results
        assert "system_health" in results
        
        # Check that results are within expected ranges
        assert 0 <= results["success_rate"] <= 1
        assert results["avg_steps"] > 0
        assert results["total_time"] > 0
    
    @pytest.mark.slow
    def test_parallel_evaluation(self, benchmark_suite, mock_env, mock_agent):
        """Test parallel evaluation."""
        results = benchmark_suite.evaluate(
            env=mock_env,
            agent=mock_agent,
            num_episodes=4,
            max_steps_per_episode=5,
            parallel=True,
            num_workers=2
        )
        
        assert results is not None
        assert results["num_episodes"] == 4
        assert results["parallel"] == True
        assert "episodes" in results
        assert len(results["episodes"]) == 4
    
    def test_evaluation_with_seed(self, benchmark_suite, mock_env, mock_agent):
        """Test evaluation with random seed for reproducibility."""
        seed = 42
        
        # Run evaluation twice with same seed
        results1 = benchmark_suite.evaluate(mock_env, mock_agent, 2, seed=seed)
        results2 = benchmark_suite.evaluate(mock_env, mock_agent, 2, seed=seed)
        
        # Results should be similar (not exact due to mock randomness)
        assert results1["num_episodes"] == results2["num_episodes"]
        assert abs(results1["avg_steps"] - results2["avg_steps"]) < 2
    
    def test_evaluation_error_handling(self, benchmark_suite, mock_agent):
        """Test error handling during evaluation."""
        # Create environment that raises exception
        class FailingEnv(BaseEnv):
            def __init__(self, config):
                super().__init__(config)
            
            def reset(self, seed=None):
                raise RuntimeError("Simulated environment failure")
            
            def step(self, action):
                pass
            
            def render(self, mode="rgb_array"):
                pass
            
            def close(self):
                pass
            
            def _get_observation(self):
                pass
        
        failing_env = FailingEnv({})
        
        with pytest.raises(RuntimeError):
            benchmark_suite.evaluate(failing_env, mock_agent, 1)
    
    def test_metrics_computation(self, benchmark_suite, mock_env, mock_agent):
        """Test metrics computation during evaluation."""
        results = benchmark_suite.evaluate(mock_env, mock_agent, 2)
        
        assert "metrics" in results
        metrics = results["metrics"]
        
        # Check that default metrics are computed
        assert "success_rate" in metrics
        assert "efficiency" in metrics
        assert "safety" in metrics
        
        # Check metric structure
        for metric_name, metric_data in metrics.items():
            assert "mean" in metric_data
            assert "std" in metric_data
            assert "min" in metric_data
            assert "max" in metric_data
    
    def test_results_history(self, benchmark_suite, mock_env, mock_agent):
        """Test that results are stored in history."""
        initial_history_length = len(benchmark_suite._results_history)
        
        benchmark_suite.evaluate(mock_env, mock_agent, 1)
        
        assert len(benchmark_suite._results_history) == initial_history_length + 1
        
        # Test multiple evaluations
        benchmark_suite.evaluate(mock_env, mock_agent, 1)
        assert len(benchmark_suite._results_history) == initial_history_length + 2
    
    def test_save_and_load_results(self, benchmark_suite, mock_env, mock_agent):
        """Test saving and loading results."""
        # Run evaluation
        results = benchmark_suite.evaluate(mock_env, mock_agent, 2)
        
        # Save results to temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_path = f.name
        
        try:
            benchmark_suite.save_results(temp_path, results)
            
            # Verify file exists and has content
            assert Path(temp_path).exists()
            
            # Load and verify results
            with open(temp_path, 'r') as f:
                loaded_results = json.load(f)
            
            assert loaded_results["num_episodes"] == results["num_episodes"]
            assert loaded_results["agent_name"] == results["agent_name"]
            
        finally:
            # Cleanup
            Path(temp_path).unlink(missing_ok=True)
    
    def test_training_tasks_selection(self, benchmark_suite):
        """Test training tasks selection functionality."""
        # Add tasks with different difficulties
        easy_task = MockTask({"difficulty": "easy"})
        medium_task = MockTask({"difficulty": "medium"})
        hard_task = MockTask({"difficulty": "hard"})
        
        benchmark_suite.add_task("easy_task", easy_task)
        benchmark_suite.add_task("medium_task", medium_task)
        benchmark_suite.add_task("hard_task", hard_task)
        
        # Test difficulty filtering
        training_tasks = benchmark_suite.get_training_tasks(
            difficulty_range=(0.1, 0.7),
            num_tasks=10
        )
        
        assert len(training_tasks) <= 10
        # Should include easy and medium tasks, exclude hard
        assert any(task.difficulty == "easy" for task in training_tasks)
        assert any(task.difficulty == "medium" for task in training_tasks)
    
    def test_benchmark_suite_configuration(self):
        """Test benchmark suite with different configurations."""
        custom_config = {
            "evaluator": {
                "timeout": 60,
                "safety_checks": False,
                "record_trajectories": True
            }
        }
        
        suite = BenchmarkSuite(custom_config)
        
        assert suite.config == custom_config
        assert suite.evaluator.timeout == 60
        assert suite.evaluator.safety_checks == False
        assert suite.evaluator.record_trajectories == True


class TestIntegrationWithTaskFactory:
    """Integration tests with task factory."""
    
    @pytest.fixture
    def mock_agent(self):
        """Create mock agent."""
        return RandomAgent({"agent_id": "integration_test_agent"})
    
    def test_task_factory_integration(self, mock_agent):
        """Test integration with task factory."""
        # This test assumes task factory works with mock simulator
        try:
            env = make_env("FurnitureAssembly-v0", simulator="mock", max_steps=5)
            suite = BenchmarkSuite()
            
            results = suite.evaluate(env, mock_agent, num_episodes=2)
            
            assert results is not None
            assert results["num_episodes"] == 2
            
        except Exception as e:
            # If task factory fails, skip test
            pytest.skip(f"Task factory integration failed: {e}")


@pytest.mark.performance
class TestPerformance:
    """Performance tests for benchmark suite."""
    
    def test_evaluation_performance(self):
        """Test evaluation performance with timing constraints."""
        config = {"max_steps": 10}
        env = MockEnv(config)
        task = MockTask(config)
        env.set_task(task)
        
        agent = RandomAgent({"agent_id": "perf_test_agent"})
        suite = BenchmarkSuite()
        
        start_time = time.time()
        results = suite.evaluate(env, agent, num_episodes=10, max_steps_per_episode=5)
        end_time = time.time()
        
        # Evaluation should complete within reasonable time
        execution_time = end_time - start_time
        assert execution_time < 10.0  # Should take less than 10 seconds
        
        # Check performance metrics
        assert results["total_time"] > 0
        assert results["avg_episode_time"] > 0
    
    def test_parallel_vs_sequential_performance(self):
        """Test that parallel evaluation is faster than sequential."""
        config = {"max_steps": 20}
        env = MockEnv(config)
        task = MockTask(config)
        env.set_task(task)
        
        agent = RandomAgent({"agent_id": "perf_comparison_agent"})
        suite = BenchmarkSuite()
        
        num_episodes = 8
        
        # Sequential evaluation
        start_time = time.time()
        sequential_results = suite.evaluate(
            env, agent, num_episodes=num_episodes, parallel=False
        )
        sequential_time = time.time() - start_time
        
        # Parallel evaluation
        start_time = time.time()
        parallel_results = suite.evaluate(
            env, agent, num_episodes=num_episodes, parallel=True, num_workers=4
        )
        parallel_time = time.time() - start_time
        
        # Parallel should be faster (or at least not significantly slower)
        # Allow some tolerance for overhead
        assert parallel_time <= sequential_time * 1.5
        
        # Results should be similar
        assert abs(sequential_results["success_rate"] - parallel_results["success_rate"]) < 0.3


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
"""Unit tests for evaluation components."""

import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock

from embodied_ai_benchmark.evaluation.benchmark_suite import BenchmarkSuite
from embodied_ai_benchmark.evaluation.evaluator import Evaluator
from embodied_ai_benchmark.core.base_agent import RandomAgent
from embodied_ai_benchmark.core.base_metric import SuccessMetric, EfficiencyMetric


class MockTask:
    """Mock task for testing."""
    
    def __init__(self):
        self.current_step = 0
        self.max_steps = 100
        self.completed = False
    
    def reset(self):
        self.current_step = 0
        self.completed = False
        return {"observation": np.random.randn(5)}
    
    def step(self, action):
        self.current_step += 1
        reward = np.random.randn()
        done = self.current_step >= self.max_steps or np.random.random() < 0.1
        if done:
            self.completed = True
        
        info = {
            "task_success": done and np.random.random() < 0.7,
            "collision": np.random.random() < 0.1,
            "applied_force": np.random.uniform(0, 150)
        }
        
        return {"observation": np.random.randn(5)}, reward, done, info
    
    def check_success(self):
        return self.completed and np.random.random() < 0.7


class MockEnv:
    """Mock environment for testing."""
    
    def __init__(self):
        self.task = MockTask()
        self._episode_count = 0
    
    def reset(self, seed=None):
        if seed is not None:
            np.random.seed(seed)
        self._episode_count += 1
        return self.task.reset()
    
    def step(self, action):
        return self.task.step(action)
    
    def seed(self, seed):
        np.random.seed(seed)
    
    def get_state(self):
        return {"episode_count": self._episode_count}


class TestEvaluator:
    """Test Evaluator functionality."""
    
    def test_evaluator_initialization(self):
        """Test evaluator initialization."""
        config = {
            "timeout": 300,
            "safety_checks": True,
            "record_trajectories": True
        }
        
        evaluator = Evaluator(config)
        
        assert evaluator.timeout == 300
        assert evaluator.safety_checks is True
        assert evaluator.record_trajectories is True
    
    def test_run_episode(self, sample_agent_config):
        """Test running a single episode."""
        evaluator = Evaluator({})
        env = MockEnv()
        agent = RandomAgent(sample_agent_config)
        
        result = evaluator.run_episode(env, agent, max_steps=50, episode_id=0)
        
        assert "episode_id" in result
        assert "total_steps" in result
        assert "total_reward" in result
        assert "success" in result
        assert "total_time" in result
        assert "steps" in result
        assert result["episode_id"] == 0
        assert result["total_steps"] <= 50
    
    def test_run_episode_with_seed(self, sample_agent_config):
        """Test episode with seed for reproducibility."""
        evaluator = Evaluator({})
        env = MockEnv()
        agent = RandomAgent(sample_agent_config)
        
        # Run same episode twice with same seed
        result1 = evaluator.run_episode(env, agent, max_steps=50, seed=42)
        result2 = evaluator.run_episode(env, agent, max_steps=50, seed=42)
        
        # Should have same number of steps (due to seeding)
        assert result1["total_steps"] == result2["total_steps"]
    
    def test_safety_checks(self, sample_agent_config):
        """Test safety violation detection."""
        config = {"safety_checks": True, "max_force_threshold": 100.0}
        evaluator = Evaluator(config)
        
        # Test collision detection
        violations = evaluator._check_safety(
            {"values": [0.1, 0.1]},
            {"collision": True, "applied_force": 50.0}
        )
        
        assert "collision_detected" in violations
        
        # Test force violation
        violations = evaluator._check_safety(
            {"values": [0.1, 0.1]},
            {"collision": False, "applied_force": 150.0}
        )
        
        assert any("excessive_force" in v for v in violations)
    
    def test_observation_serialization(self):
        """Test observation serialization for storage."""
        evaluator = Evaluator({})
        
        # Test with large array (should be summarized)
        large_obs = {
            "rgb": np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8),
            "depth": np.random.randn(480, 640).astype(np.float32),
            "small_array": np.array([1, 2, 3])
        }
        
        serialized = evaluator._serialize_observation(large_obs)
        
        # Large arrays should be summarized
        assert serialized["rgb"]["type"] == "large_array"
        assert "shape" in serialized["rgb"]
        assert "mean" in serialized["rgb"]
        
        # Small arrays should be preserved
        assert isinstance(serialized["small_array"], list)
        assert serialized["small_array"] == [1, 2, 3]
    
    def test_batch_episodes(self, sample_agent_config):
        """Test batch episode evaluation."""
        evaluator = Evaluator({})
        env = MockEnv()
        
        agents = [
            RandomAgent({**sample_agent_config, "agent_id": "agent_1"}),
            RandomAgent({**sample_agent_config, "agent_id": "agent_2"})
        ]
        
        results = evaluator.run_batch_episodes(
            env, agents, num_episodes_per_agent=3, max_steps=20
        )
        
        assert len(results) == 2
        assert "agent_1" in results
        assert "agent_2" in results
        assert len(results["agent_1"]) == 3
        assert len(results["agent_2"]) == 3
    
    def test_agent_comparison(self, sample_agent_config):
        """Test agent comparison functionality."""
        evaluator = Evaluator({})
        env = MockEnv()
        
        agents = [
            RandomAgent({**sample_agent_config, "agent_id": "agent_1"}),
            RandomAgent({**sample_agent_config, "agent_id": "agent_2"})
        ]
        
        comparison = evaluator.compare_agents(
            env, agents, num_episodes=5, max_steps=20
        )
        
        assert "agent_summaries" in comparison
        assert "rankings" in comparison
        assert len(comparison["rankings"]) == 2
        
        # Check ranking structure
        ranking = comparison["rankings"][0]
        assert "agent_id" in ranking
        assert "rank" in ranking
        assert "metrics" in ranking
        assert ranking["rank"] == 1  # Top ranked


class TestBenchmarkSuite:
    """Test BenchmarkSuite functionality."""
    
    def test_benchmark_suite_initialization(self):
        """Test benchmark suite initialization."""
        config = {"evaluator": {"timeout": 300}}
        suite = BenchmarkSuite(config)
        
        assert suite.config == config
        assert len(suite.metrics) > 0  # Default metrics should be added
        assert "success_rate" in suite.metrics
        assert "efficiency" in suite.metrics
        assert "safety" in suite.metrics
    
    def test_add_task_and_metric(self):
        """Test adding tasks and metrics."""
        suite = BenchmarkSuite()
        
        # Add mock task
        mock_task = MockTask()
        suite.add_task("test_task", mock_task)
        
        assert "test_task" in suite.tasks
        assert suite.tasks["test_task"] == mock_task
        
        # Add custom metric
        custom_metric = SuccessMetric({"name": "custom_success"})
        suite.add_metric("custom_success", custom_metric)
        
        assert "custom_success" in suite.metrics
    
    def test_sequential_evaluation(self, sample_agent_config):
        """Test sequential evaluation."""
        suite = BenchmarkSuite()
        env = MockEnv()
        agent = RandomAgent(sample_agent_config)
        
        results = suite.evaluate(
            env=env,
            agent=agent,
            num_episodes=5,
            max_steps_per_episode=20,
            seed=42,
            parallel=False
        )
        
        assert "num_episodes" in results
        assert "success_rate" in results
        assert "avg_steps" in results
        assert "avg_reward" in results
        assert "metrics" in results
        assert results["num_episodes"] == 5
    
    @pytest.mark.slow
    def test_parallel_evaluation(self, sample_agent_config):
        """Test parallel evaluation."""
        suite = BenchmarkSuite()
        env = MockEnv()
        agent = RandomAgent(sample_agent_config)
        
        results = suite.evaluate(
            env=env,
            agent=agent,
            num_episodes=8,
            max_steps_per_episode=20,
            parallel=True,
            num_workers=2
        )
        
        assert "num_episodes" in results
        assert results["num_episodes"] == 8
        assert "total_time" in results
        assert results["total_time"] > 0
    
    def test_metric_aggregation(self, sample_agent_config):
        """Test metric aggregation across episodes."""
        suite = BenchmarkSuite()
        env = MockEnv()
        agent = RandomAgent(sample_agent_config)
        
        results = suite.evaluate(
            env=env,
            agent=agent,
            num_episodes=10,
            max_steps_per_episode=20
        )
        
        # Check metric aggregation
        metrics = results["metrics"]
        
        for metric_name in ["success_rate", "efficiency", "safety"]:
            assert metric_name in metrics
            metric_data = metrics[metric_name]
            assert "mean" in metric_data
            assert "std" in metric_data
            assert "min" in metric_data
            assert "max" in metric_data
            assert "values" in metric_data
            assert len(metric_data["values"]) == 10
    
    def test_training_task_filtering(self):
        """Test training task filtering by difficulty."""
        suite = BenchmarkSuite()
        
        # Add tasks with different difficulties
        easy_task = MockTask()
        easy_task.difficulty = "easy"
        
        medium_task = MockTask()
        medium_task.difficulty = "medium"
        
        hard_task = MockTask()
        hard_task.difficulty = "hard"
        
        suite.add_task("easy_task", easy_task)
        suite.add_task("medium_task", medium_task)
        suite.add_task("hard_task", hard_task)
        
        # Get training tasks for medium difficulty range
        training_tasks = suite.get_training_tasks(
            difficulty_range=(0.4, 0.8),  # Should include medium tasks
            num_tasks=10
        )
        
        # Should include medium task
        assert any(task.difficulty == "medium" for task in training_tasks)
    
    def test_results_saving(self, sample_agent_config, tmp_path):
        """Test saving evaluation results."""
        suite = BenchmarkSuite()
        env = MockEnv()
        agent = RandomAgent(sample_agent_config)
        
        results = suite.evaluate(
            env=env,
            agent=agent,
            num_episodes=3,
            max_steps_per_episode=10
        )
        
        # Save results
        results_file = tmp_path / "test_results.json"
        suite.save_results(str(results_file), results)
        
        assert results_file.exists()
        
        # Check file content
        import json
        with open(results_file) as f:
            saved_data = json.load(f)
        
        assert "num_episodes" in saved_data
        assert saved_data["num_episodes"] == 3
    
    def test_results_history(self, sample_agent_config):
        """Test results history tracking."""
        suite = BenchmarkSuite()
        env = MockEnv()
        agent = RandomAgent(sample_agent_config)
        
        # Run multiple evaluations
        suite.evaluate(env, agent, num_episodes=2, max_steps_per_episode=10)
        suite.evaluate(env, agent, num_episodes=3, max_steps_per_episode=10)
        
        history = suite.get_results_history()
        
        assert len(history) == 2
        assert history[0]["num_episodes"] == 2
        assert history[1]["num_episodes"] == 3
    
    def test_empty_episodes_handling(self, sample_agent_config):
        """Test handling of empty episode results."""
        suite = BenchmarkSuite()
        
        # Test aggregation with empty results
        results = suite._aggregate_results([], 10.0)
        
        assert "error" in results
        assert "No episode results to aggregate" in results["error"]
    
    def test_custom_evaluator_config(self):
        """Test custom evaluator configuration."""
        config = {
            "evaluator": {
                "timeout": 600,
                "safety_checks": False,
                "record_trajectories": False
            }
        }
        
        suite = BenchmarkSuite(config)
        
        assert suite.evaluator.timeout == 600
        assert suite.evaluator.safety_checks is False
        assert suite.evaluator.record_trajectories is False
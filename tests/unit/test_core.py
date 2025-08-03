"""Unit tests for core abstractions."""

import pytest
import numpy as np
from unittest.mock import Mock, patch

from embodied_ai_benchmark.core.base_task import BaseTask
from embodied_ai_benchmark.core.base_env import BaseEnv
from embodied_ai_benchmark.core.base_agent import BaseAgent, RandomAgent
from embodied_ai_benchmark.core.base_metric import (
    BaseMetric, SuccessMetric, EfficiencyMetric, SafetyMetric
)


class TestTask(BaseTask):
    """Test implementation of BaseTask."""
    
    def __init__(self, config):
        super().__init__(config)
        self.position = np.array([0.0, 0.0])
        self.goal = np.array([1.0, 1.0])
        
    def reset(self):
        super().reset()
        self.position = np.array([0.0, 0.0])
        return self._get_observation()
    
    def step(self, action):
        self.current_step += 1
        
        # Simple movement
        movement = np.array(action.get("values", [0.0, 0.0]))[:2]
        self.position += movement * 0.1
        
        # Calculate reward
        distance = np.linalg.norm(self.position - self.goal)
        reward = -distance
        
        # Check if done
        done = distance < 0.1 or self.current_step >= self.max_steps
        
        info = {"distance": distance, "task_success": distance < 0.1}
        
        return self._get_observation(), reward, done, info
    
    def _get_observation(self):
        return {
            "position": self.position.tolist(),
            "goal": self.goal.tolist(),
            "distance": float(np.linalg.norm(self.position - self.goal))
        }
    
    def check_success(self):
        return np.linalg.norm(self.position - self.goal) < 0.1
    
    def compute_reward(self):
        return -np.linalg.norm(self.position - self.goal)


class TestEnv(BaseEnv):
    """Test implementation of BaseEnv."""
    
    def __init__(self, config):
        super().__init__(config)
        self.task = None
        
    def set_task(self, task):
        self.task = task
        
    def reset(self, seed=None):
        super().reset(seed)
        if self.task:
            return self.task.reset()
        return self._get_observation()
    
    def step(self, action):
        if self.task:
            return self.task.step(action)
        return self._get_observation(), 0.0, True, {}
    
    def render(self, mode="rgb_array"):
        if mode == "rgb_array":
            return np.zeros((100, 100, 3), dtype=np.uint8)
        return None
    
    def close(self):
        pass
    
    def _get_observation(self):
        return {"dummy": np.zeros(5)}


class TestAgent(BaseAgent):
    """Test implementation of BaseAgent."""
    
    def reset(self):
        super().reset()
        
    def act(self, observation):
        # Simple policy: move toward goal
        if "goal" in observation and "position" in observation:
            pos = np.array(observation["position"])
            goal = np.array(observation["goal"])
            direction = goal - pos
            direction = direction / (np.linalg.norm(direction) + 1e-6)
            
            action = {
                "type": "move",
                "values": direction.tolist()
            }
        else:
            action = {
                "type": "move", 
                "values": [0.0, 0.0]
            }
        
        self._action_history.append(action)
        return action
    
    def update(self, observation, action, reward, next_observation, done):
        pass


class TestBaseTask:
    """Test BaseTask functionality."""
    
    def test_task_initialization(self, sample_task_config):
        """Test task initialization."""
        task = TestTask(sample_task_config)
        
        assert task.name == sample_task_config["name"]
        assert task.difficulty == sample_task_config["difficulty"]
        assert task.max_steps == sample_task_config["max_steps"]
        assert task.current_step == 0
    
    def test_task_reset(self, sample_task_config):
        """Test task reset functionality."""
        task = TestTask(sample_task_config)
        
        # Move to non-zero position
        task.position = np.array([0.5, 0.5])
        task.current_step = 10
        
        # Reset
        obs = task.reset()
        
        assert task.current_step == 0
        assert np.allclose(task.position, [0.0, 0.0])
        assert "position" in obs
        assert "goal" in obs
    
    def test_task_step(self, sample_task_config):
        """Test task step functionality."""
        task = TestTask(sample_task_config)
        task.reset()
        
        action = {"values": [1.0, 1.0]}
        obs, reward, done, info = task.step(action)
        
        assert task.current_step == 1
        assert isinstance(reward, (int, float))
        assert isinstance(done, bool)
        assert "distance" in info
        assert "task_success" in info
    
    def test_task_success_check(self, sample_task_config):
        """Test task success checking."""
        task = TestTask(sample_task_config)
        task.reset()
        
        # Not at goal initially
        assert not task.check_success()
        
        # Move to goal
        task.position = np.array([1.0, 1.0])
        assert task.check_success()
    
    def test_action_space_specification(self, sample_task_config):
        """Test action space specification."""
        task = TestTask(sample_task_config)
        action_space = task.get_action_space()
        
        assert "type" in action_space
        assert "shape" in action_space
        assert "low" in action_space
        assert "high" in action_space
    
    def test_observation_space_specification(self, sample_task_config):
        """Test observation space specification."""
        task = TestTask(sample_task_config)
        obs_space = task.get_observation_space()
        
        assert "rgb" in obs_space
        assert "depth" in obs_space
        assert "proprioception" in obs_space


class TestBaseEnv:
    """Test BaseEnv functionality."""
    
    def test_env_initialization(self):
        """Test environment initialization."""
        config = {"simulator": "test", "num_agents": 1}
        env = TestEnv(config)
        
        assert env.simulator_name == "test"
        assert env.num_agents == 1
        assert env._episode_count == 0
    
    def test_env_reset(self):
        """Test environment reset."""
        env = TestEnv({})
        obs = env.reset(seed=42)
        
        assert env._episode_count == 1
        assert isinstance(obs, dict)
    
    def test_env_step(self):
        """Test environment step."""
        env = TestEnv({})
        env.reset()
        
        action = {"type": "move", "values": [0.1, 0.1]}
        obs, reward, done, info = env.step(action)
        
        assert isinstance(obs, dict)
        assert isinstance(reward, (int, float))
        assert isinstance(done, bool)
        assert isinstance(info, dict)
    
    def test_env_render(self):
        """Test environment rendering."""
        env = TestEnv({})
        
        # RGB array mode
        img = env.render("rgb_array")
        assert isinstance(img, np.ndarray)
        assert img.shape == (100, 100, 3)
        
        # Human mode
        result = env.render("human")
        assert result is None
    
    def test_env_task_integration(self, sample_task_config):
        """Test environment-task integration."""
        env = TestEnv({})
        task = TestTask(sample_task_config)
        
        env.set_task(task)
        obs = env.reset()
        
        # Should get task observation
        assert "position" in obs
        assert "goal" in obs


class TestBaseAgent:
    """Test BaseAgent functionality."""
    
    def test_agent_initialization(self, sample_agent_config):
        """Test agent initialization."""
        agent = TestAgent(sample_agent_config)
        
        assert agent.agent_id == sample_agent_config["agent_id"]
        assert len(agent._action_history) == 0
        assert len(agent._observation_history) == 0
    
    def test_agent_reset(self, sample_agent_config):
        """Test agent reset."""
        agent = TestAgent(sample_agent_config)
        
        # Add some history
        agent._action_history.append({"action": "test"})
        agent._observation_history.append({"obs": "test"})
        
        agent.reset()
        
        assert len(agent._action_history) == 0
        assert len(agent._observation_history) == 0
    
    def test_agent_act(self, sample_agent_config):
        """Test agent action selection."""
        agent = TestAgent(sample_agent_config)
        
        obs = {
            "position": [0.0, 0.0],
            "goal": [1.0, 1.0]
        }
        
        action = agent.act(obs)
        
        assert "type" in action
        assert "values" in action
        assert len(agent._action_history) == 1
    
    def test_agent_capabilities(self, sample_agent_config):
        """Test agent capabilities."""
        sample_agent_config["capabilities"] = ["move", "pick", "place"]
        agent = TestAgent(sample_agent_config)
        
        capabilities = agent.get_capabilities()
        assert "move" in capabilities
        assert "pick" in capabilities
        assert "place" in capabilities


class TestRandomAgent:
    """Test RandomAgent implementation."""
    
    def test_random_agent_initialization(self, sample_agent_config):
        """Test random agent initialization."""
        agent = RandomAgent(sample_agent_config)
        
        assert agent.agent_id == sample_agent_config["agent_id"]
        assert "action_space" in agent.__dict__
    
    def test_random_agent_continuous_actions(self, sample_agent_config):
        """Test random agent continuous actions."""
        agent = RandomAgent(sample_agent_config)
        obs = {"dummy": [1, 2, 3]}
        
        action = agent.act(obs)
        
        assert action["type"] == "continuous"
        assert "values" in action
        assert len(action["values"]) == 7  # Based on config
    
    def test_random_agent_discrete_actions(self, sample_agent_config):
        """Test random agent discrete actions."""
        sample_agent_config["action_space"] = {
            "type": "discrete",
            "n": 4
        }
        agent = RandomAgent(sample_agent_config)
        obs = {"dummy": [1, 2, 3]}
        
        action = agent.act(obs)
        
        assert action["type"] == "discrete"
        assert "values" in action
        assert 0 <= action["values"] < 4


class TestBaseMetric:
    """Test BaseMetric functionality."""
    
    def test_success_metric(self):
        """Test SuccessMetric implementation."""
        metric = SuccessMetric({"name": "success_rate"})
        
        # Test reset
        metric.reset()
        assert len(metric._data) == 0
        assert not metric.task_completed
        
        # Test update with success
        info = {"task_success": True}
        metric.update({}, {}, 0.0, {}, True, info)
        
        assert metric.task_completed
        assert metric.compute() == 1.0
        
        # Test update with failure
        metric.reset()
        info = {"task_success": False}
        metric.update({}, {}, 0.0, {}, True, info)
        
        assert not metric.task_completed
        assert metric.compute() == 0.0
    
    def test_efficiency_metric(self):
        """Test EfficiencyMetric implementation."""
        config = {"name": "efficiency", "max_steps": 100}
        metric = EfficiencyMetric(config)
        
        metric.reset()
        
        # Simulate 50 steps
        for i in range(50):
            metric.update({}, {}, 0.1, {}, False, {})
        
        efficiency = metric.compute()
        assert 0.0 <= efficiency <= 1.0
        assert efficiency == 0.5  # 1.0 - (50/100)
    
    def test_safety_metric(self):
        """Test SafetyMetric implementation."""
        config = {"name": "safety", "max_force": 100.0}
        metric = SafetyMetric(config)
        
        metric.reset()
        
        # Test safe operation
        info = {"collision": False, "applied_force": 50.0}
        metric.update({}, {}, 0.0, {}, False, info)
        
        safety_score = metric.compute()
        assert safety_score == 1.0
        
        # Test collision
        info = {"collision": True, "applied_force": 50.0}
        metric.update({}, {}, 0.0, {}, False, info)
        
        safety_score = metric.compute()
        assert safety_score < 1.0
    
    def test_metric_weight_and_name(self):
        """Test metric weight and name functionality."""
        config = {"name": "test_metric", "weight": 0.8}
        metric = SuccessMetric(config)
        
        assert metric.get_name() == "success_rate"  # SuccessMetric overrides name
        assert metric.get_weight() == 0.8
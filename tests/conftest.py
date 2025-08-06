"""Pytest configuration and shared fixtures for embodied AI benchmark tests."""

import pytest
import tempfile
import shutil
import asyncio
from pathlib import Path
from typing import Dict, Any
from unittest.mock import Mock, MagicMock
import numpy as np

from embodied_ai_benchmark import (
    BaseTask, BaseEnv, BaseAgent, RandomAgent,
    BenchmarkSuite, Evaluator
)
from embodied_ai_benchmark.database.connection import DatabaseConnection
# Migration import disabled for testing - use in-memory setup instead
from embodied_ai_benchmark.tasks.task_factory import make_env


@pytest.fixture(scope="session")
def temp_db_path():
    """Create temporary database file."""
    temp_dir = tempfile.mkdtemp()
    db_path = Path(temp_dir) / "test_benchmark.db"
    yield str(db_path)
    shutil.rmtree(temp_dir)


@pytest.fixture(scope="session")
def test_db_config(temp_db_path):
    """Test database configuration."""
    return {
        "type": "sqlite",
        "path": temp_db_path
    }


@pytest.fixture(scope="session")
def test_database(test_db_config):
    """Create test database with schema."""
    db = DatabaseConnection(test_db_config)
    # Create basic test schema instead of migration
    db.execute_update("""
        CREATE TABLE IF NOT EXISTS episodes (
            id INTEGER PRIMARY KEY,
            experiment_id TEXT,
            episode_data TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    db.execute_update("""
        CREATE TABLE IF NOT EXISTS benchmark_runs (
            id INTEGER PRIMARY KEY,
            name TEXT,
            results TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    yield db
    db.close()


@pytest.fixture
def clean_database(test_database):
    """Clean database for each test."""
    # Clear all tables
    tables = ["episodes", "benchmark_runs"]
    
    for table in tables:
        test_database.execute_update(f"DELETE FROM {table}")
    
    yield test_database


@pytest.fixture
def sample_agent_config():
    """Sample agent configuration."""
    return {
        "agent_id": "test_agent",
        "action_space": {
            "type": "continuous",
            "shape": (7,),
            "low": [-1] * 7,
            "high": [1] * 7
        }
    }


@pytest.fixture
def random_agent(sample_agent_config):
    """Create random agent for testing."""
    return RandomAgent(sample_agent_config)


@pytest.fixture
def sample_task_config():
    """Sample task configuration."""
    return {
        "name": "test_task",
        "difficulty": "medium",
        "max_steps": 100,
        "time_limit": 60
    }


@pytest.fixture
def furniture_assembly_env():
    """Create furniture assembly environment."""
    return make_env("FurnitureAssembly-v0", simulator="habitat")


@pytest.fixture
def point_goal_env():
    """Create point goal navigation environment."""
    return make_env("PointGoal-v0", simulator="habitat")


@pytest.fixture
def cooperative_assembly_env():
    """Create cooperative assembly environment."""
    return make_env("CooperativeFurnitureAssembly-v0", num_agents=2)


@pytest.fixture
def sample_experiment_data():
    """Sample experiment data for testing."""
    return {
        "name": "test_experiment",
        "description": "Test experiment for unit tests",
        "config": {
            "agent_type": "random",
            "num_episodes": 10,
            "tasks": ["FurnitureAssembly-v0"],
            "metrics": ["success_rate", "efficiency"]
        }
    }


@pytest.fixture
def sample_benchmark_results():
    """Sample benchmark results data."""
    return {
        "num_episodes": 10,
        "total_time": 120.5,
        "success_rate": 0.8,
        "avg_steps": 150.0,
        "avg_reward": 25.5,
        "metrics": {
            "success_rate": {"mean": 0.8, "std": 0.1},
            "efficiency": {"mean": 0.6, "std": 0.15},
            "safety": {"mean": 0.95, "std": 0.05}
        },
        "episodes": []
    }


@pytest.fixture
def sample_episode_data():
    """Sample episode data."""
    return {
        "episode_id": 0,
        "total_steps": 150,
        "total_reward": 25.5,
        "success": True,
        "total_time": 12.5,
        "safety_violations": [],
        "metrics": {
            "success_rate": 1.0,
            "efficiency": 0.75,
            "safety": 1.0
        }
    }


@pytest.fixture
def mock_redis_config():
    """Mock Redis configuration for testing."""
    return {
        "host": "localhost",
        "port": 6379,
        "db": 15,  # Use test database
        "password": None
    }


# Test markers
def pytest_configure(config):
    """Configure pytest markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "unit: marks tests as unit tests"
    )
    config.addinivalue_line(
        "markers", "requires_redis: marks tests that require Redis"
    )


# Pytest collection hook to add default markers
def pytest_collection_modifyitems(config, items):
    """Automatically mark tests based on their location."""
    for item in items:
        # Mark integration tests
        if "integration" in item.nodeid:
            item.add_marker(pytest.mark.integration)
        # Mark unit tests (default)
        elif "unit" in item.nodeid or "test_" in item.name:
            item.add_marker(pytest.mark.unit)
        
        # Mark slow tests
        if any(keyword in item.nodeid for keyword in ["slow", "benchmark", "evaluation"]):
            item.add_marker(pytest.mark.slow)
        
        # Mark Redis-dependent tests
        if any(keyword in item.nodeid for keyword in ["cache", "redis"]):
            item.add_marker(pytest.mark.requires_redis)


# Additional fixtures for comprehensive testing

@pytest.fixture
def mock_env():
    """Create a mock environment for testing."""
    class MockEnv(BaseEnv):
        def __init__(self):
            super().__init__("mock_env_fixture")
            self.state = np.array([0.0, 0.0, 0.0])
            self.step_count = 0
            
        def reset(self):
            self.state = np.array([0.0, 0.0, 0.0])
            self.step_count = 0
            return {"observation": self.state.copy()}
            
        def step(self, action):
            self.step_count += 1
            
            # Simple state transitions
            if action == "move_x":
                self.state[0] += 1.0
            elif action == "move_y":
                self.state[1] += 1.0
            elif action == "move_z":
                self.state[2] += 1.0
                
            obs = {"observation": self.state.copy()}
            reward = 1.0 if np.sum(self.state) > 0 else 0.0
            done = self.step_count >= 10
            info = {"step": self.step_count, "success": np.sum(self.state) > 5}
            
            return obs, reward, done, info
            
        def render(self):
            return f"State: {self.state}"
            
        def close(self):
            pass
    
    return MockEnv()


@pytest.fixture
def mock_task(mock_env):
    """Create a mock task for testing."""
    class MockTask(BaseTask):
        def __init__(self, env):
            super().__init__("mock_task_fixture", env, max_steps=10)
            self.target = np.array([2.0, 2.0, 2.0])
            
        def get_reward(self, observation, action, next_observation, info):
            current_state = next_observation["observation"]
            distance = np.linalg.norm(current_state - self.target)
            return -distance + 5.0  # Reward for being close to target
            
        def is_success(self, observation, info):
            current_state = observation["observation"]
            distance = np.linalg.norm(current_state - self.target)
            return distance < 1.0
    
    return MockTask(mock_env)


@pytest.fixture
def mock_agent():
    """Create a mock agent for testing."""
    class MockAgent(BaseAgent):
        def __init__(self):
            super().__init__("mock_agent_fixture")
            self.actions = ["move_x", "move_y", "move_z", "stay"]
            self.action_history = []
            
        def act(self, observation):
            # Simple policy: try to increase state values
            current_state = observation.get("observation", np.array([0, 0, 0]))
            
            if current_state[0] < 2:
                action = "move_x"
            elif current_state[1] < 2:
                action = "move_y"  
            elif current_state[2] < 2:
                action = "move_z"
            else:
                action = "stay"
                
            self.action_history.append(action)
            return action
            
        def reset(self):
            self.action_history = []
    
    return MockAgent()


@pytest.fixture
def mock_llm_client():
    """Create a mock LLM client for testing."""
    mock_client = Mock()
    
    # Default responses for different types of requests
    mock_client.generate_text.return_value = '''
    {
        "tasks": [
            {
                "name": "basic_navigation",
                "difficulty": 0.3,
                "focus_areas": ["movement", "obstacle_avoidance"],
                "estimated_duration": "2 hours",
                "max_episodes": 10
            },
            {
                "name": "advanced_manipulation",
                "difficulty": 0.7,
                "focus_areas": ["grasping", "precision"],
                "estimated_duration": "4 hours",
                "max_episodes": 15
            }
        ],
        "reasoning": "Progressive curriculum to build foundational skills first"
    }
    '''
    
    return mock_client


@pytest.fixture
def isolated_metrics():
    """Create isolated metrics collection for each test."""
    from embodied_ai_benchmark.utils.benchmark_metrics import BenchmarkMetricsCollector
    
    # Create fresh metrics collector for each test
    metrics = BenchmarkMetricsCollector()
    yield metrics
    
    # Clean up after test
    metrics.reset_all_metrics()


@pytest.fixture
def coordination_agents():
    """Create mock agents for coordination testing."""
    agents = {}
    
    for i in range(3):
        agent_id = f"agent_{i}"
        agents[agent_id] = {
            "capabilities": ["navigation", "communication"],
            "performance_history": {
                "navigation": 0.7 + i * 0.1,
                "communication": 0.8 + i * 0.05
            },
            "current_tasks": [],
            "availability": True,
            "max_concurrent_tasks": 2
        }
    
    return agents


# Custom assertions
def assert_valid_metrics(metrics_dict):
    """Assert that metrics dictionary has valid structure."""
    assert isinstance(metrics_dict, dict)
    
    required_keys = ["counters", "gauges", "timers", "histograms"]
    for key in required_keys:
        assert key in metrics_dict
        assert isinstance(metrics_dict[key], dict)


def assert_valid_benchmark_result(result):
    """Assert that benchmark result has valid structure."""
    assert isinstance(result, dict)
    
    required_keys = ["total_episodes", "success_rate", "average_reward"]
    for key in required_keys:
        assert key in result
        
    assert isinstance(result["total_episodes"], int)
    assert 0.0 <= result["success_rate"] <= 1.0
    assert isinstance(result["average_reward"], (int, float))


def assert_valid_task_execution(task, agent, num_steps=10):
    """Assert that task can be executed successfully with agent."""
    obs = task.reset()
    assert obs is not None
    
    total_reward = 0
    for step in range(num_steps):
        action = agent.act(obs)
        assert action is not None
        
        next_obs, reward, done, info = task.step(action)
        total_reward += reward
        
        assert next_obs is not None
        assert isinstance(reward, (int, float))
        assert isinstance(done, bool)
        assert isinstance(info, dict)
        
        if done:
            break
        obs = next_obs
    
    return total_reward


# Make custom assertions available to tests
pytest.assert_valid_metrics = assert_valid_metrics
pytest.assert_valid_benchmark_result = assert_valid_benchmark_result
pytest.assert_valid_task_execution = assert_valid_task_execution
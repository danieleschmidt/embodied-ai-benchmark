"""Pytest configuration and shared fixtures."""

import pytest
import tempfile
import shutil
from pathlib import Path
from typing import Dict, Any

from embodied_ai_benchmark.database.connection import DatabaseConnection
# Migration import disabled for testing - use in-memory setup instead
from embodied_ai_benchmark.core.base_agent import RandomAgent
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
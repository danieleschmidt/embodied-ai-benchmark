"""Unit tests for repository classes."""

import pytest
import json
from datetime import datetime
from unittest.mock import MagicMock, patch

from embodied_ai_benchmark.repositories.base_repository import BaseRepository
from embodied_ai_benchmark.repositories.experiment_repository import (
    ExperimentRepository, BenchmarkRunRepository
)
from tests.factories import ExperimentFactory, BenchmarkRunFactory


class TestRepository(BaseRepository):
    """Test repository for testing base functionality."""
    
    def _get_table_name(self) -> str:
        return "test_table"


@pytest.mark.unit
class TestBaseRepository:
    """Test base repository functionality."""
    
    def test_initialization(self, clean_database):
        """Test repository initialization."""
        repo = TestRepository(clean_database)
        assert repo.db == clean_database
        assert repo.table_name == "test_table"
        assert repo.primary_key == "id"
    
    def test_find_all(self, clean_database):
        """Test finding all records."""
        repo = ExperimentRepository(clean_database)
        
        # Create test data
        exp_data = ExperimentFactory.create()
        exp_id = repo.create(exp_data)
        
        # Test find_all
        results = repo.find_all()
        assert len(results) == 1
        assert results[0]["id"] == exp_id
        assert results[0]["name"] == exp_data["name"]
    
    def test_find_by_id(self, clean_database):
        """Test finding record by ID."""
        repo = ExperimentRepository(clean_database)
        
        # Create test data
        exp_data = ExperimentFactory.create()
        exp_id = repo.create(exp_data)
        
        # Test find_by_id
        result = repo.find_by_id(exp_id)
        assert result is not None
        assert result["id"] == exp_id
        assert result["name"] == exp_data["name"]
        
        # Test non-existent ID
        result = repo.find_by_id(99999)
        assert result is None
    
    def test_find_by_field(self, clean_database):
        """Test finding records by field value."""
        repo = ExperimentRepository(clean_database)
        
        # Create test data
        exp_data = ExperimentFactory.create(name="unique_experiment")
        repo.create(exp_data)
        
        # Test find_by_field
        results = repo.find_by_field("name", "unique_experiment")
        assert len(results) == 1
        assert results[0]["name"] == "unique_experiment"
        
        # Test non-existent value
        results = repo.find_by_field("name", "non_existent")
        assert len(results) == 0
    
    def test_find_by_fields(self, clean_database):
        """Test finding records by multiple field criteria."""
        repo = ExperimentRepository(clean_database)
        
        # Create test data
        exp_data = ExperimentFactory.create(name="test_exp", status="completed")
        repo.create(exp_data)
        
        # Test find_by_fields
        results = repo.find_by_fields({
            "name": "test_exp",
            "status": "completed"
        })
        assert len(results) == 1
        assert results[0]["name"] == "test_exp"
        assert results[0]["status"] == "completed"
        
        # Test partial match
        results = repo.find_by_fields({
            "name": "test_exp",
            "status": "pending"
        })
        assert len(results) == 0
    
    def test_create(self, clean_database):
        """Test creating new record."""
        repo = ExperimentRepository(clean_database)
        
        # Test create
        exp_data = ExperimentFactory.create()
        exp_id = repo.create(exp_data)
        
        assert exp_id is not None
        assert isinstance(exp_id, int)
        
        # Verify record was created
        result = repo.find_by_id(exp_id)
        assert result is not None
        assert result["name"] == exp_data["name"]
    
    def test_update(self, clean_database):
        """Test updating existing record."""
        repo = ExperimentRepository(clean_database)
        
        # Create test data
        exp_data = ExperimentFactory.create()
        exp_id = repo.create(exp_data)
        
        # Test update
        update_data = {"status": "completed"}
        success = repo.update(exp_id, update_data)
        
        assert success is True
        
        # Verify update
        result = repo.find_by_id(exp_id)
        assert result["status"] == "completed"
        assert "updated_at" in result
    
    def test_delete(self, clean_database):
        """Test deleting record."""
        repo = ExperimentRepository(clean_database)
        
        # Create test data
        exp_data = ExperimentFactory.create()
        exp_id = repo.create(exp_data)
        
        # Test delete
        success = repo.delete(exp_id)
        assert success is True
        
        # Verify deletion
        result = repo.find_by_id(exp_id)
        assert result is None
    
    def test_count(self, clean_database):
        """Test counting records."""
        repo = ExperimentRepository(clean_database)
        
        # Test empty count
        count = repo.count()
        assert count == 0
        
        # Create test data
        exp_data1 = ExperimentFactory.create(status="pending")
        exp_data2 = ExperimentFactory.create(status="completed")
        repo.create(exp_data1)
        repo.create(exp_data2)
        
        # Test total count
        count = repo.count()
        assert count == 2
        
        # Test filtered count
        count = repo.count({"status": "pending"})
        assert count == 1
    
    def test_exists(self, clean_database):
        """Test checking record existence."""
        repo = ExperimentRepository(clean_database)
        
        # Test non-existent record
        exists = repo.exists(99999)
        assert exists is False
        
        # Create test data
        exp_data = ExperimentFactory.create()
        exp_id = repo.create(exp_data)
        
        # Test existing record
        exists = repo.exists(exp_id)
        assert exists is True
    
    def test_find_with_pagination(self, clean_database):
        """Test pagination functionality."""
        repo = ExperimentRepository(clean_database)
        
        # Create test data
        for i in range(25):
            exp_data = ExperimentFactory.create(name=f"exp_{i:02d}")
            repo.create(exp_data)
        
        # Test pagination
        records, total_count = repo.find_with_pagination(
            page=1, page_size=10
        )
        
        assert len(records) == 10
        assert total_count == 25
        
        # Test second page
        records, total_count = repo.find_with_pagination(
            page=2, page_size=10
        )
        
        assert len(records) == 10
        assert total_count == 25
        
        # Test last page
        records, total_count = repo.find_with_pagination(
            page=3, page_size=10
        )
        
        assert len(records) == 5
        assert total_count == 25
    
    def test_bulk_insert(self, clean_database):
        """Test bulk insert functionality."""
        repo = ExperimentRepository(clean_database)
        
        # Create test data
        exp_data_list = ExperimentFactory.create_batch(5)
        
        # Test bulk insert
        created_ids = repo.bulk_insert(exp_data_list)
        
        assert len(created_ids) == 5
        assert all(isinstance(id_, int) for id_ in created_ids)
        
        # Verify records were created
        all_records = repo.find_all()
        assert len(all_records) == 5


@pytest.mark.unit
class TestExperimentRepository:
    """Test experiment repository functionality."""
    
    def test_create_experiment(self, clean_database):
        """Test creating experiment."""
        repo = ExperimentRepository(clean_database)
        
        # Test create_experiment
        exp_id = repo.create_experiment(
            name="Test Experiment",
            description="Test description",
            config={"agent_type": "random"}
        )
        
        assert exp_id is not None
        
        # Verify experiment
        exp = repo.find_by_id(exp_id)
        assert exp["name"] == "Test Experiment"
        assert exp["description"] == "Test description"
        assert exp["status"] == "pending"
    
    def test_find_by_name(self, clean_database):
        """Test finding experiment by name."""
        repo = ExperimentRepository(clean_database)
        
        # Create test experiment
        exp_id = repo.create_experiment(
            name="Unique Experiment",
            description="Test",
            config={}
        )
        
        # Test find_by_name
        exp = repo.find_by_name("Unique Experiment")
        assert exp is not None
        assert exp["id"] == exp_id
        
        # Test non-existent name
        exp = repo.find_by_name("Non-existent")
        assert exp is None
    
    def test_update_status(self, clean_database):
        """Test updating experiment status."""
        repo = ExperimentRepository(clean_database)
        
        # Create test experiment
        exp_id = repo.create_experiment(
            name="Test Experiment",
            description="Test",
            config={}
        )
        
        # Test update_status
        success = repo.update_status(exp_id, "completed")
        assert success is True
        
        # Verify status update
        exp = repo.find_by_id(exp_id)
        assert exp["status"] == "completed"
        assert "completed_at" in exp
    
    def test_get_experiment_summary(self, clean_database):
        """Test getting experiment summary."""
        exp_repo = ExperimentRepository(clean_database)
        run_repo = BenchmarkRunRepository(clean_database)
        
        # Create test experiment
        exp_id = exp_repo.create_experiment(
            name="Test Experiment",
            description="Test",
            config={"test": True}
        )
        
        # Create test runs
        for i in range(3):
            run_data = BenchmarkRunFactory.create(
                experiment_id=exp_id,
                status="completed"
            )
            run_repo.create(run_data)
        
        # Test get_experiment_summary
        summary = exp_repo.get_experiment_summary(exp_id)
        assert summary is not None
        assert summary["name"] == "Test Experiment"
        assert summary["total_runs"] == 3
        assert summary["completed_runs"] == 3
        assert isinstance(summary["config"], dict)


@pytest.mark.unit
class TestBenchmarkRunRepository:
    """Test benchmark run repository functionality."""
    
    def test_create_run(self, clean_database):
        """Test creating benchmark run."""
        exp_repo = ExperimentRepository(clean_database)
        run_repo = BenchmarkRunRepository(clean_database)
        
        # Create parent experiment
        exp_id = exp_repo.create_experiment(
            name="Test Experiment",
            description="Test",
            config={}
        )
        
        # Test create_run
        run_id = run_repo.create_run(
            experiment_id=exp_id,
            agent_name="TestAgent",
            task_name="TestTask",
            config={"test": True}
        )
        
        assert run_id is not None
        
        # Verify run
        run = run_repo.find_by_id(run_id)
        assert run["experiment_id"] == exp_id
        assert run["agent_name"] == "TestAgent"
        assert run["task_name"] == "TestTask"
        assert run["status"] == "pending"
    
    def test_update_run_results(self, clean_database):
        """Test updating run results."""
        exp_repo = ExperimentRepository(clean_database)
        run_repo = BenchmarkRunRepository(clean_database)
        
        # Create test run
        exp_id = exp_repo.create_experiment("Test", "Test", {})
        run_id = run_repo.create_run(exp_id, "Agent", "Task", {})
        
        # Test update_run_results
        results = {
            "num_episodes": 100,
            "success_rate": 0.85,
            "avg_reward": 25.5,
            "avg_steps": 150.0,
            "total_time": 120.5
        }
        
        success = run_repo.update_run_results(run_id, results)
        assert success is True
        
        # Verify results
        run = run_repo.find_by_id(run_id)
        assert run["status"] == "completed"
        assert run["success_rate"] == 0.85
        assert run["avg_reward"] == 25.5
        assert "completed_at" in run
    
    def test_mark_run_failed(self, clean_database):
        """Test marking run as failed."""
        exp_repo = ExperimentRepository(clean_database)
        run_repo = BenchmarkRunRepository(clean_database)
        
        # Create test run
        exp_id = exp_repo.create_experiment("Test", "Test", {})
        run_id = run_repo.create_run(exp_id, "Agent", "Task", {})
        
        # Test mark_run_failed
        success = run_repo.mark_run_failed(run_id, "Test error")
        assert success is True
        
        # Verify failure
        run = run_repo.find_by_id(run_id)
        assert run["status"] == "failed"
        assert run["error_message"] == "Test error"
        assert "completed_at" in run
    
    def test_find_by_experiment(self, clean_database):
        """Test finding runs by experiment."""
        exp_repo = ExperimentRepository(clean_database)
        run_repo = BenchmarkRunRepository(clean_database)
        
        # Create test experiment and runs
        exp_id = exp_repo.create_experiment("Test", "Test", {})
        
        for i in range(3):
            run_repo.create_run(exp_id, f"Agent{i}", "Task", {})
        
        # Test find_by_experiment
        runs = run_repo.find_by_experiment(exp_id)
        assert len(runs) == 3
        assert all(run["experiment_id"] == exp_id for run in runs)
    
    def test_get_agent_performance_summary(self, clean_database):
        """Test getting agent performance summary."""
        exp_repo = ExperimentRepository(clean_database)
        run_repo = BenchmarkRunRepository(clean_database)
        
        # Create test data
        exp_id = exp_repo.create_experiment("Test", "Test", {})
        
        # Create runs for same agent on different tasks
        tasks = ["Task1", "Task2", "Task3"]
        for task in tasks:
            run_data = BenchmarkRunFactory.create(
                experiment_id=exp_id,
                agent_name="TestAgent",
                task_name=task,
                status="completed"
            )
            run_repo.create(run_data)
        
        # Test get_agent_performance_summary
        summary = run_repo.get_agent_performance_summary("TestAgent")
        
        assert summary["agent_name"] == "TestAgent"
        assert len(summary["task_performance"]) == 3
        assert "overall" in summary
        assert summary["overall"]["total_runs"] == 3
    
    def test_get_task_leaderboard(self, clean_database):
        """Test getting task leaderboard."""
        exp_repo = ExperimentRepository(clean_database)
        run_repo = BenchmarkRunRepository(clean_database)
        
        # Create test data
        exp_id = exp_repo.create_experiment("Test", "Test", {})
        
        # Create runs for different agents on same task
        agents = ["Agent1", "Agent2", "Agent3"]
        for i, agent in enumerate(agents):
            run_data = BenchmarkRunFactory.create(
                experiment_id=exp_id,
                agent_name=agent,
                task_name="TestTask",
                status="completed"
            )
            # Ensure different success rates for ranking
            run_data["success_rate"] = 0.9 - (i * 0.1)
            run_repo.create(run_data)
        
        # Test get_task_leaderboard
        leaderboard = run_repo.get_task_leaderboard("TestTask", limit=5)
        
        assert len(leaderboard) == 3
        # Check ranking order (highest success rate first)
        assert leaderboard[0]["rank"] == 1
        assert leaderboard[0]["agent_name"] == "Agent1"
        assert leaderboard[1]["rank"] == 2
        assert leaderboard[1]["agent_name"] == "Agent2"

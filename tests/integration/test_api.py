"""Integration tests for REST API."""

import pytest
import json
from flask import Flask
from unittest.mock import patch, MagicMock

from embodied_ai_benchmark.api.app import create_app
from embodied_ai_benchmark.repositories.experiment_repository import ExperimentRepository
from tests.factories import ExperimentFactory, BenchmarkRunFactory


@pytest.fixture
def app(test_database):
    """Create Flask app for testing."""
    app = create_app({
        'TESTING': True,
        'DATABASE_URL': 'sqlite:///:memory:',
        'SECRET_KEY': 'test-secret-key'
    })
    
    with app.app_context():
        yield app


@pytest.fixture
def client(app):
    """Create test client."""
    return app.test_client()


@pytest.fixture
def auth_headers():
    """Mock authentication headers."""
    return {
        'Content-Type': 'application/json'
    }


@pytest.mark.integration
class TestExperimentAPI:
    """Test experiment API endpoints."""
    
    def test_list_experiments_empty(self, client, auth_headers):
        """Test listing experiments when none exist."""
        response = client.get('/api/experiments', headers=auth_headers)
        
        assert response.status_code == 200
        data = response.get_json()
        assert 'experiments' in data
        assert len(data['experiments']) == 0
        assert 'pagination' in data
    
    def test_create_experiment_success(self, client, auth_headers, clean_database):
        """Test successful experiment creation."""
        experiment_data = {
            'name': 'Test Experiment',
            'description': 'Test description',
            'config': {
                'agent_type': 'random',
                'num_episodes': 10
            }
        }
        
        response = client.post(
            '/api/experiments',
            data=json.dumps(experiment_data),
            headers=auth_headers
        )
        
        assert response.status_code == 201
        data = response.get_json()
        assert data['message'] == 'Experiment created successfully'
        assert 'experiment' in data
        assert data['experiment']['name'] == 'Test Experiment'
    
    def test_create_experiment_invalid_data(self, client, auth_headers):
        """Test experiment creation with invalid data."""
        # Missing required field
        invalid_data = {
            'description': 'Missing name field'
        }
        
        response = client.post(
            '/api/experiments',
            data=json.dumps(invalid_data),
            headers=auth_headers
        )
        
        assert response.status_code == 400
        data = response.get_json()
        assert 'error' in data
    
    def test_create_experiment_duplicate_name(self, client, auth_headers, clean_database):
        """Test creating experiment with duplicate name."""
        # Create first experiment
        exp_repo = ExperimentRepository(clean_database)
        exp_data = ExperimentFactory.create(name='Duplicate Name')
        exp_repo.create(exp_data)
        
        # Try to create another with same name
        duplicate_data = {
            'name': 'Duplicate Name',
            'description': 'Should fail',
            'config': {}
        }
        
        response = client.post(
            '/api/experiments',
            data=json.dumps(duplicate_data),
            headers=auth_headers
        )
        
        assert response.status_code == 422
        data = response.get_json()
        assert 'already exists' in data['error']
    
    def test_get_experiment_success(self, client, auth_headers, clean_database):
        """Test getting specific experiment."""
        # Create test experiment
        exp_repo = ExperimentRepository(clean_database)
        exp_data = ExperimentFactory.create()
        exp_id = exp_repo.create_experiment(
            name=exp_data['name'],
            description=exp_data['description'],
            config=exp_data['config']
        )
        
        response = client.get(f'/api/experiments/{exp_id}', headers=auth_headers)
        
        assert response.status_code == 200
        data = response.get_json()
        assert data['name'] == exp_data['name']
        assert data['id'] == exp_id
    
    def test_get_experiment_not_found(self, client, auth_headers):
        """Test getting non-existent experiment."""
        response = client.get('/api/experiments/99999', headers=auth_headers)
        
        assert response.status_code == 404
        data = response.get_json()
        assert 'not found' in data['error']
    
    def test_update_experiment_success(self, client, auth_headers, clean_database):
        """Test updating experiment."""
        # Create test experiment
        exp_repo = ExperimentRepository(clean_database)
        exp_data = ExperimentFactory.create()
        exp_id = exp_repo.create_experiment(
            name=exp_data['name'],
            description=exp_data['description'],
            config=exp_data['config']
        )
        
        # Update experiment
        update_data = {
            'description': 'Updated description',
            'status': 'completed'
        }
        
        response = client.put(
            f'/api/experiments/{exp_id}',
            data=json.dumps(update_data),
            headers=auth_headers
        )
        
        assert response.status_code == 200
        data = response.get_json()
        assert data['message'] == 'Experiment updated successfully'
        assert data['experiment']['description'] == 'Updated description'
    
    def test_delete_experiment_success(self, client, auth_headers, clean_database):
        """Test deleting experiment."""
        # Create test experiment
        exp_repo = ExperimentRepository(clean_database)
        exp_data = ExperimentFactory.create()
        exp_id = exp_repo.create_experiment(
            name=exp_data['name'],
            description=exp_data['description'],
            config=exp_data['config']
        )
        
        response = client.delete(f'/api/experiments/{exp_id}', headers=auth_headers)
        
        assert response.status_code == 200
        data = response.get_json()
        assert data['message'] == 'Experiment deleted successfully'
        
        # Verify deletion
        get_response = client.get(f'/api/experiments/{exp_id}', headers=auth_headers)
        assert get_response.status_code == 404
    
    def test_list_experiments_with_pagination(self, client, auth_headers, clean_database):
        """Test listing experiments with pagination."""
        # Create multiple experiments
        exp_repo = ExperimentRepository(clean_database)
        for i in range(25):
            exp_data = ExperimentFactory.create(name=f'Experiment {i:02d}')
            exp_repo.create(exp_data)
        
        # Test first page
        response = client.get('/api/experiments?page=1&page_size=10', headers=auth_headers)
        
        assert response.status_code == 200
        data = response.get_json()
        assert len(data['experiments']) == 10
        assert data['pagination']['page'] == 1
        assert data['pagination']['total_count'] == 25
        assert data['pagination']['total_pages'] == 3
        
        # Test second page
        response = client.get('/api/experiments?page=2&page_size=10', headers=auth_headers)
        
        assert response.status_code == 200
        data = response.get_json()
        assert len(data['experiments']) == 10
        assert data['pagination']['page'] == 2
    
    def test_list_experiments_with_status_filter(self, client, auth_headers, clean_database):
        """Test filtering experiments by status."""
        # Create experiments with different statuses
        exp_repo = ExperimentRepository(clean_database)
        
        # Create pending experiments
        for i in range(3):
            exp_data = ExperimentFactory.create(status='pending')
            exp_repo.create(exp_data)
        
        # Create completed experiments
        for i in range(2):
            exp_data = ExperimentFactory.create(status='completed')
            exp_repo.create(exp_data)
        
        # Filter by pending status
        response = client.get('/api/experiments?status=pending', headers=auth_headers)
        
        assert response.status_code == 200
        data = response.get_json()
        assert len(data['experiments']) == 3
        assert all(exp['status'] == 'pending' for exp in data['experiments'])
        
        # Filter by completed status
        response = client.get('/api/experiments?status=completed', headers=auth_headers)
        
        assert response.status_code == 200
        data = response.get_json()
        assert len(data['experiments']) == 2
        assert all(exp['status'] == 'completed' for exp in data['experiments'])


@pytest.mark.integration
class TestRunAPI:
    """Test benchmark run API endpoints."""
    
    def test_create_run_success(self, client, auth_headers, clean_database):
        """Test successful run creation."""
        # Create parent experiment
        exp_repo = ExperimentRepository(clean_database)
        exp_data = ExperimentFactory.create()
        exp_id = exp_repo.create_experiment(
            name=exp_data['name'],
            description=exp_data['description'],
            config=exp_data['config']
        )
        
        run_data = {
            'experiment_id': exp_id,
            'agent_name': 'TestAgent',
            'task_name': 'TestTask',
            'config': {
                'num_episodes': 10,
                'timeout': 300
            }
        }
        
        response = client.post(
            '/api/runs',
            data=json.dumps(run_data),
            headers=auth_headers
        )
        
        assert response.status_code == 201
        data = response.get_json()
        assert data['message'] == 'Run created successfully'
        assert 'run' in data
        assert data['run']['agent_name'] == 'TestAgent'
    
    def test_update_run_results_success(self, client, auth_headers, clean_database):
        """Test updating run results."""
        # Create test run
        exp_repo = ExperimentRepository(clean_database)
        exp_data = ExperimentFactory.create()
        exp_id = exp_repo.create_experiment(
            name=exp_data['name'],
            description=exp_data['description'],
            config=exp_data['config']
        )
        
        from embodied_ai_benchmark.repositories.experiment_repository import BenchmarkRunRepository
        run_repo = BenchmarkRunRepository(clean_database)
        run_id = run_repo.create_run(exp_id, 'TestAgent', 'TestTask', {})
        
        # Update results
        results_data = {
            'num_episodes': 100,
            'success_rate': 0.85,
            'avg_reward': 25.5,
            'avg_steps': 150.0,
            'total_time': 120.5
        }
        
        response = client.put(
            f'/api/runs/{run_id}/results',
            data=json.dumps(results_data),
            headers=auth_headers
        )
        
        assert response.status_code == 200
        data = response.get_json()
        assert data['message'] == 'Run results updated successfully'
        assert data['run']['success_rate'] == 0.85
    
    def test_list_runs_with_filters(self, client, auth_headers, clean_database):
        """Test listing runs with various filters."""
        # Create test data
        exp_repo = ExperimentRepository(clean_database)
        exp_data = ExperimentFactory.create()
        exp_id = exp_repo.create_experiment(
            name=exp_data['name'],
            description=exp_data['description'],
            config=exp_data['config']
        )
        
        from embodied_ai_benchmark.repositories.experiment_repository import BenchmarkRunRepository
        run_repo = BenchmarkRunRepository(clean_database)
        
        # Create runs with different agents and tasks
        agents = ['Agent1', 'Agent2', 'Agent3']
        tasks = ['Task1', 'Task2']
        
        for agent in agents:
            for task in tasks:
                run_repo.create_run(exp_id, agent, task, {})
        
        # Filter by agent
        response = client.get('/api/runs?agent_name=Agent1', headers=auth_headers)
        assert response.status_code == 200
        data = response.get_json()
        assert len(data['runs']) == 2  # Agent1 on 2 tasks
        assert all(run['agent_name'] == 'Agent1' for run in data['runs'])
        
        # Filter by task
        response = client.get('/api/runs?task_name=Task1', headers=auth_headers)
        assert response.status_code == 200
        data = response.get_json()
        assert len(data['runs']) == 3  # 3 agents on Task1
        assert all(run['task_name'] == 'Task1' for run in data['runs'])
        
        # Filter by experiment
        response = client.get(f'/api/runs?experiment_id={exp_id}', headers=auth_headers)
        assert response.status_code == 200
        data = response.get_json()
        assert len(data['runs']) == 6  # All runs
        assert all(run['experiment_id'] == exp_id for run in data['runs'])


@pytest.mark.integration
class TestAgentAPI:
    """Test agent API endpoints."""
    
    def test_list_agents(self, client, auth_headers, clean_database):
        """Test listing agents."""
        # Create test data with completed runs
        exp_repo = ExperimentRepository(clean_database)
        exp_data = ExperimentFactory.create()
        exp_id = exp_repo.create_experiment(
            name=exp_data['name'],
            description=exp_data['description'],
            config=exp_data['config']
        )
        
        from embodied_ai_benchmark.repositories.experiment_repository import BenchmarkRunRepository
        run_repo = BenchmarkRunRepository(clean_database)
        
        # Create completed runs for different agents
        agents = ['Agent1', 'Agent2']
        for agent in agents:
            run_data = BenchmarkRunFactory.create(
                experiment_id=exp_id,
                agent_name=agent,
                status='completed'
            )
            run_repo.create(run_data)
        
        response = client.get('/api/agents', headers=auth_headers)
        
        assert response.status_code == 200
        data = response.get_json()
        assert 'agents' in data
        agent_names = [agent['name'] for agent in data['agents']]
        assert 'Agent1' in agent_names
        assert 'Agent2' in agent_names
    
    def test_get_agent_performance(self, client, auth_headers, clean_database):
        """Test getting agent performance."""
        # Create test data
        exp_repo = ExperimentRepository(clean_database)
        exp_data = ExperimentFactory.create()
        exp_id = exp_repo.create_experiment(
            name=exp_data['name'],
            description=exp_data['description'],
            config=exp_data['config']
        )
        
        from embodied_ai_benchmark.repositories.experiment_repository import BenchmarkRunRepository
        run_repo = BenchmarkRunRepository(clean_database)
        
        # Create completed runs for specific agent
        for i in range(3):
            run_data = BenchmarkRunFactory.create(
                experiment_id=exp_id,
                agent_name='TestAgent',
                task_name=f'Task{i}',
                status='completed'
            )
            run_repo.create(run_data)
        
        response = client.get('/api/agents/TestAgent/performance', headers=auth_headers)
        
        assert response.status_code == 200
        data = response.get_json()
        assert data['agent_name'] == 'TestAgent'
        assert 'task_performance' in data
        assert 'overall' in data


@pytest.mark.integration
class TestTaskAPI:
    """Test task API endpoints."""
    
    @patch('embodied_ai_benchmark.tasks.task_factory.TaskFactory.get_available_tasks')
    def test_list_tasks(self, mock_get_tasks, client, auth_headers):
        """Test listing available tasks."""
        mock_get_tasks.return_value = [
            {'name': 'FurnitureAssembly-v0', 'type': 'manipulation'},
            {'name': 'PointGoal-v0', 'type': 'navigation'},
            {'name': 'CooperativeFurnitureAssembly-v0', 'type': 'multiagent'}
        ]
        
        response = client.get('/api/tasks', headers=auth_headers)
        
        assert response.status_code == 200
        data = response.get_json()
        assert 'tasks' in data
        assert len(data['tasks']) == 3
    
    def test_get_task_leaderboard(self, client, auth_headers, clean_database):
        """Test getting task leaderboard."""
        # Create test data
        exp_repo = ExperimentRepository(clean_database)
        exp_data = ExperimentFactory.create()
        exp_id = exp_repo.create_experiment(
            name=exp_data['name'],
            description=exp_data['description'],
            config=exp_data['config']
        )
        
        from embodied_ai_benchmark.repositories.experiment_repository import BenchmarkRunRepository
        run_repo = BenchmarkRunRepository(clean_database)
        
        # Create completed runs for same task
        agents = ['Agent1', 'Agent2', 'Agent3']
        for i, agent in enumerate(agents):
            run_data = BenchmarkRunFactory.create(
                experiment_id=exp_id,
                agent_name=agent,
                task_name='TestTask',
                status='completed'
            )
            # Different success rates for ranking
            run_data['success_rate'] = 0.9 - (i * 0.1)
            run_repo.create(run_data)
        
        response = client.get('/api/tasks/TestTask/leaderboard', headers=auth_headers)
        
        assert response.status_code == 200
        data = response.get_json()
        assert data['task_name'] == 'TestTask'
        assert 'leaderboard' in data
        assert len(data['leaderboard']) == 3
        # Check ordering (highest success rate first)
        assert data['leaderboard'][0]['rank'] == 1
        assert data['leaderboard'][0]['agent_name'] == 'Agent1'


@pytest.mark.integration
class TestAPIErrorHandling:
    """Test API error handling."""
    
    def test_invalid_json(self, client, auth_headers):
        """Test handling of invalid JSON."""
        response = client.post(
            '/api/experiments',
            data='invalid json',
            headers=auth_headers
        )
        
        assert response.status_code == 400
    
    def test_missing_content_type(self, client):
        """Test handling of missing content type."""
        response = client.post('/api/experiments', data='{"name": "test"}')
        
        # Should still work but might be handled differently
        assert response.status_code in [400, 415, 422]
    
    def test_large_request(self, client, auth_headers):
        """Test handling of large requests."""
        # Create a large payload
        large_data = {
            'name': 'Test',
            'description': 'x' * (10 * 1024 * 1024),  # 10MB string
            'config': {}
        }
        
        response = client.post(
            '/api/experiments',
            data=json.dumps(large_data),
            headers=auth_headers
        )
        
        # Should be rejected due to size
        assert response.status_code == 413
    
    def test_health_check(self, client):
        """Test health check endpoint."""
        response = client.get('/health')
        
        assert response.status_code in [200, 503]  # Healthy or unhealthy
        data = response.get_json()
        assert 'status' in data
        assert 'database' in data
    
    def test_api_docs(self, client):
        """Test API documentation endpoint."""
        response = client.get('/api/docs')
        
        assert response.status_code == 200
        data = response.get_json()
        assert 'title' in data
        assert 'endpoints' in data
        assert 'schemas' in data

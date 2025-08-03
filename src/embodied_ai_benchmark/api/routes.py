"""API route definitions."""

import json
from datetime import datetime
from flask import Blueprint, request, jsonify, current_app
from werkzeug.exceptions import BadRequest, NotFound, UnprocessableEntity
from typing import Dict, Any, List, Optional

from .validation import (
    validate_experiment_data, validate_run_data, validate_agent_data,
    validate_task_data, validate_pagination_params
)
from ..repositories.experiment_repository import ExperimentRepository, BenchmarkRunRepository
from ..repositories.base_repository import BaseRepository
from ..database.connection import get_database
from ..cache.cache_manager import BenchmarkCacheManager
from ..evaluation.benchmark_suite import BenchmarkSuite
from ..tasks.task_factory import TaskFactory

# Create blueprint
api_bp = Blueprint('api', __name__)

# Initialize repositories and services
def get_experiment_repo() -> ExperimentRepository:
    """Get experiment repository instance."""
    return ExperimentRepository(get_database())

def get_run_repo() -> BenchmarkRunRepository:
    """Get benchmark run repository instance."""
    return BenchmarkRunRepository(get_database())

def get_cache_manager() -> BenchmarkCacheManager:
    """Get cache manager instance."""
    return BenchmarkCacheManager()

def get_benchmark_suite() -> BenchmarkSuite:
    """Get benchmark suite instance."""
    return BenchmarkSuite()

def get_task_factory() -> TaskFactory:
    """Get task factory instance."""
    return TaskFactory()


# Experiment endpoints
@api_bp.route('/experiments', methods=['GET'])
def list_experiments():
    """List all experiments with pagination."""
    try:
        # Validate pagination parameters
        page = request.args.get('page', 1, type=int)
        page_size = request.args.get('page_size', 20, type=int)
        status = request.args.get('status')
        
        validate_pagination_params(page, page_size)
        
        repo = get_experiment_repo()
        
        # Build filter criteria
        criteria = {}
        if status:
            criteria['status'] = status
        
        # Get experiments with pagination
        experiments, total_count = repo.find_with_pagination(
            page=page,
            page_size=page_size,
            criteria=criteria,
            order_by='created_at DESC'
        )
        
        return jsonify({
            "experiments": experiments,
            "pagination": {
                "page": page,
                "page_size": page_size,
                "total_count": total_count,
                "total_pages": (total_count + page_size - 1) // page_size
            }
        })
        
    except Exception as e:
        current_app.logger.error(f"Error listing experiments: {str(e)}")
        return jsonify({"error": "Failed to list experiments"}), 500


@api_bp.route('/experiments', methods=['POST'])
def create_experiment():
    """Create a new experiment."""
    try:
        data = request.get_json()
        if not data:
            raise BadRequest("Request body must contain JSON data")
        
        # Validate experiment data
        validate_experiment_data(data)
        
        repo = get_experiment_repo()
        
        # Check if experiment name already exists
        existing = repo.find_by_name(data['name'])
        if existing:
            raise UnprocessableEntity(f"Experiment with name '{data['name']}' already exists")
        
        # Create experiment
        exp_id = repo.create_experiment(
            name=data['name'],
            description=data.get('description', ''),
            config=data.get('config', {})
        )
        
        if not exp_id:
            raise Exception("Failed to create experiment")
        
        # Get created experiment
        experiment = repo.find_by_id(exp_id)
        
        return jsonify({
            "message": "Experiment created successfully",
            "experiment": experiment
        }), 201
        
    except (BadRequest, UnprocessableEntity) as e:
        return jsonify({"error": str(e)}), e.code
    except Exception as e:
        current_app.logger.error(f"Error creating experiment: {str(e)}")
        return jsonify({"error": "Failed to create experiment"}), 500


@api_bp.route('/experiments/<int:exp_id>', methods=['GET'])
def get_experiment(exp_id: int):
    """Get specific experiment by ID."""
    try:
        repo = get_experiment_repo()
        
        # Check cache first
        cache = get_cache_manager()
        cached_summary = cache.get(f"experiment_summary_{exp_id}")
        
        if cached_summary:
            return jsonify(json.loads(cached_summary))
        
        # Get experiment summary
        experiment = repo.get_experiment_summary(exp_id)
        if not experiment:
            raise NotFound(f"Experiment with ID {exp_id} not found")
        
        # Cache the result
        cache.set(f"experiment_summary_{exp_id}", json.dumps(experiment), ttl=300)
        
        return jsonify(experiment)
        
    except NotFound as e:
        return jsonify({"error": str(e)}), 404
    except Exception as e:
        current_app.logger.error(f"Error getting experiment: {str(e)}")
        return jsonify({"error": "Failed to get experiment"}), 500


@api_bp.route('/experiments/<int:exp_id>', methods=['PUT'])
def update_experiment(exp_id: int):
    """Update experiment."""
    try:
        data = request.get_json()
        if not data:
            raise BadRequest("Request body must contain JSON data")
        
        repo = get_experiment_repo()
        
        # Check if experiment exists
        if not repo.exists(exp_id):
            raise NotFound(f"Experiment with ID {exp_id} not found")
        
        # Update experiment
        success = repo.update(exp_id, data)
        if not success:
            raise Exception("Failed to update experiment")
        
        # Invalidate cache
        cache = get_cache_manager()
        cache.delete(f"experiment_summary_{exp_id}")
        
        # Get updated experiment
        experiment = repo.find_by_id(exp_id)
        
        return jsonify({
            "message": "Experiment updated successfully",
            "experiment": experiment
        })
        
    except (BadRequest, NotFound) as e:
        return jsonify({"error": str(e)}), e.code
    except Exception as e:
        current_app.logger.error(f"Error updating experiment: {str(e)}")
        return jsonify({"error": "Failed to update experiment"}), 500


@api_bp.route('/experiments/<int:exp_id>', methods=['DELETE'])
def delete_experiment(exp_id: int):
    """Delete experiment."""
    try:
        repo = get_experiment_repo()
        
        # Check if experiment exists
        if not repo.exists(exp_id):
            raise NotFound(f"Experiment with ID {exp_id} not found")
        
        # Delete experiment (cascade will handle related runs)
        success = repo.delete(exp_id)
        if not success:
            raise Exception("Failed to delete experiment")
        
        # Invalidate cache
        cache = get_cache_manager()
        cache.delete(f"experiment_summary_{exp_id}")
        
        return jsonify({"message": "Experiment deleted successfully"})
        
    except NotFound as e:
        return jsonify({"error": str(e)}), 404
    except Exception as e:
        current_app.logger.error(f"Error deleting experiment: {str(e)}")
        return jsonify({"error": "Failed to delete experiment"}), 500


# Benchmark Run endpoints
@api_bp.route('/runs', methods=['GET'])
def list_runs():
    """List benchmark runs with filtering."""
    try:
        page = request.args.get('page', 1, type=int)
        page_size = request.args.get('page_size', 20, type=int)
        experiment_id = request.args.get('experiment_id', type=int)
        agent_name = request.args.get('agent_name')
        task_name = request.args.get('task_name')
        status = request.args.get('status')
        
        validate_pagination_params(page, page_size)
        
        repo = get_run_repo()
        
        # Build filter criteria
        criteria = {}
        if experiment_id:
            criteria['experiment_id'] = experiment_id
        if agent_name:
            criteria['agent_name'] = agent_name
        if task_name:
            criteria['task_name'] = task_name
        if status:
            criteria['status'] = status
        
        # Get runs with pagination
        runs, total_count = repo.find_with_pagination(
            page=page,
            page_size=page_size,
            criteria=criteria,
            order_by='started_at DESC'
        )
        
        return jsonify({
            "runs": runs,
            "pagination": {
                "page": page,
                "page_size": page_size,
                "total_count": total_count,
                "total_pages": (total_count + page_size - 1) // page_size
            }
        })
        
    except Exception as e:
        current_app.logger.error(f"Error listing runs: {str(e)}")
        return jsonify({"error": "Failed to list runs"}), 500


@api_bp.route('/runs', methods=['POST'])
def create_run():
    """Create a new benchmark run."""
    try:
        data = request.get_json()
        if not data:
            raise BadRequest("Request body must contain JSON data")
        
        validate_run_data(data)
        
        repo = get_run_repo()
        
        # Create run
        run_id = repo.create_run(
            experiment_id=data['experiment_id'],
            agent_name=data['agent_name'],
            task_name=data['task_name'],
            config=data.get('config', {})
        )
        
        if not run_id:
            raise Exception("Failed to create run")
        
        # Get created run
        run = repo.find_by_id(run_id)
        
        return jsonify({
            "message": "Run created successfully",
            "run": run
        }), 201
        
    except (BadRequest, UnprocessableEntity) as e:
        return jsonify({"error": str(e)}), e.code
    except Exception as e:
        current_app.logger.error(f"Error creating run: {str(e)}")
        return jsonify({"error": "Failed to create run"}), 500


@api_bp.route('/runs/<int:run_id>/results', methods=['PUT'])
def update_run_results(run_id: int):
    """Update run with results."""
    try:
        data = request.get_json()
        if not data:
            raise BadRequest("Request body must contain JSON data")
        
        repo = get_run_repo()
        
        # Check if run exists
        if not repo.exists(run_id):
            raise NotFound(f"Run with ID {run_id} not found")
        
        # Update run results
        success = repo.update_run_results(run_id, data)
        if not success:
            raise Exception("Failed to update run results")
        
        # Get updated run
        run = repo.find_by_id(run_id)
        
        return jsonify({
            "message": "Run results updated successfully",
            "run": run
        })
        
    except (BadRequest, NotFound) as e:
        return jsonify({"error": str(e)}), e.code
    except Exception as e:
        current_app.logger.error(f"Error updating run results: {str(e)}")
        return jsonify({"error": "Failed to update run results"}), 500


# Agent performance endpoints
@api_bp.route('/agents', methods=['GET'])
def list_agents():
    """List all agents with performance data."""
    try:
        repo = get_run_repo()
        
        # Get unique agent names
        agents_query = "SELECT DISTINCT agent_name FROM benchmark_runs WHERE status = 'completed'"
        agent_rows = repo.db.execute_query(agents_query)
        
        agents = []
        for row in agent_rows:
            agent_name = row['agent_name']
            
            # Check cache first
            cache = get_cache_manager()
            cached_perf = cache.get_cached_agent_performance(agent_name)
            
            if cached_perf:
                performance = cached_perf
            else:
                # Get performance summary
                performance = repo.get_agent_performance_summary(agent_name)
                cache.cache_agent_performance(agent_name, performance)
            
            agents.append({
                "name": agent_name,
                "performance": performance
            })
        
        return jsonify({"agents": agents})
        
    except Exception as e:
        current_app.logger.error(f"Error listing agents: {str(e)}")
        return jsonify({"error": "Failed to list agents"}), 500


@api_bp.route('/agents/<agent_name>/performance', methods=['GET'])
def get_agent_performance(agent_name: str):
    """Get agent performance summary."""
    try:
        cache = get_cache_manager()
        
        # Check cache first
        cached_perf = cache.get_cached_agent_performance(agent_name)
        if cached_perf:
            return jsonify(cached_perf)
        
        # Get performance from database
        repo = get_run_repo()
        performance = repo.get_agent_performance_summary(agent_name)
        
        if not performance.get('task_performance'):
            raise NotFound(f"No performance data found for agent '{agent_name}'")
        
        # Cache the result
        cache.cache_agent_performance(agent_name, performance)
        
        return jsonify(performance)
        
    except NotFound as e:
        return jsonify({"error": str(e)}), 404
    except Exception as e:
        current_app.logger.error(f"Error getting agent performance: {str(e)}")
        return jsonify({"error": "Failed to get agent performance"}), 500


# Task and leaderboard endpoints
@api_bp.route('/tasks', methods=['GET'])
def list_tasks():
    """List all available tasks."""
    try:
        factory = get_task_factory()
        
        # Get task registry
        tasks = factory.get_available_tasks()
        
        return jsonify({"tasks": tasks})
        
    except Exception as e:
        current_app.logger.error(f"Error listing tasks: {str(e)}")
        return jsonify({"error": "Failed to list tasks"}), 500


@api_bp.route('/tasks/<task_name>/leaderboard', methods=['GET'])
def get_task_leaderboard(task_name: str):
    """Get leaderboard for specific task."""
    try:
        limit = request.args.get('limit', 10, type=int)
        
        cache = get_cache_manager()
        
        # Check cache first
        cached_leaderboard = cache.get_cached_leaderboard(task_name)
        if cached_leaderboard:
            return jsonify({
                "task_name": task_name,
                "leaderboard": cached_leaderboard[:limit]
            })
        
        # Get leaderboard from database
        repo = get_run_repo()
        leaderboard = repo.get_task_leaderboard(task_name, limit=limit)
        
        if not leaderboard:
            raise NotFound(f"No leaderboard data found for task '{task_name}'")
        
        # Cache the result
        cache.cache_leaderboard(task_name, leaderboard)
        
        return jsonify({
            "task_name": task_name,
            "leaderboard": leaderboard
        })
        
    except NotFound as e:
        return jsonify({"error": str(e)}), 404
    except Exception as e:
        current_app.logger.error(f"Error getting task leaderboard: {str(e)}")
        return jsonify({"error": "Failed to get task leaderboard"}), 500


# Evaluation endpoints
@api_bp.route('/evaluate', methods=['POST'])
def start_evaluation():
    """Start a new evaluation job."""
    try:
        data = request.get_json()
        if not data:
            raise BadRequest("Request body must contain JSON data")
        
        # Validate evaluation request
        required_fields = ['agent_config', 'task_name', 'num_episodes']
        for field in required_fields:
            if field not in data:
                raise BadRequest(f"Missing required field: {field}")
        
        # TODO: Implement async evaluation job queue
        # For now, return job submission confirmation
        
        evaluation_id = f"eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        return jsonify({
            "message": "Evaluation job submitted",
            "evaluation_id": evaluation_id,
            "status": "queued",
            "estimated_duration": data['num_episodes'] * 30  # seconds
        }), 202
        
    except BadRequest as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        current_app.logger.error(f"Error starting evaluation: {str(e)}")
        return jsonify({"error": "Failed to start evaluation"}), 500


# API documentation endpoint
@api_bp.route('/docs', methods=['GET'])
def api_docs():
    """API documentation."""
    return jsonify({
        "title": "Embodied-AI Benchmark++ API",
        "version": "1.0.0",
        "description": "REST API for managing embodied AI experiments and benchmarks",
        "endpoints": {
            "experiments": {
                "GET /api/experiments": "List experiments with pagination",
                "POST /api/experiments": "Create new experiment",
                "GET /api/experiments/{id}": "Get experiment by ID",
                "PUT /api/experiments/{id}": "Update experiment",
                "DELETE /api/experiments/{id}": "Delete experiment"
            },
            "runs": {
                "GET /api/runs": "List benchmark runs with filtering",
                "POST /api/runs": "Create new benchmark run",
                "PUT /api/runs/{id}/results": "Update run results"
            },
            "agents": {
                "GET /api/agents": "List all agents with performance data",
                "GET /api/agents/{name}/performance": "Get agent performance summary"
            },
            "tasks": {
                "GET /api/tasks": "List all available tasks",
                "GET /api/tasks/{name}/leaderboard": "Get task leaderboard"
            },
            "evaluation": {
                "POST /api/evaluate": "Start new evaluation job"
            }
        },
        "schemas": {
            "experiment": {
                "name": "string (required)",
                "description": "string",
                "config": "object"
            },
            "run": {
                "experiment_id": "integer (required)",
                "agent_name": "string (required)",
                "task_name": "string (required)",
                "config": "object"
            }
        }
    })

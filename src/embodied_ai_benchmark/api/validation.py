"""Input validation for API endpoints."""

from typing import Dict, Any, List, Optional
from werkzeug.exceptions import BadRequest, UnprocessableEntity
import re


def validate_experiment_data(data: Dict[str, Any]) -> None:
    """Validate experiment creation/update data.
    
    Args:
        data: Experiment data dictionary
        
    Raises:
        BadRequest: If validation fails
    """
    # Required fields for creation
    if 'name' not in data:
        raise BadRequest("Experiment name is required")
    
    # Validate name
    name = data['name']
    if not isinstance(name, str) or not name.strip():
        raise BadRequest("Experiment name must be a non-empty string")
    
    if len(name) > 255:
        raise BadRequest("Experiment name must be less than 255 characters")
    
    # Validate name format (alphanumeric, spaces, hyphens, underscores)
    if not re.match(r'^[a-zA-Z0-9\s\-_]+$', name):
        raise BadRequest("Experiment name contains invalid characters")
    
    # Validate description
    if 'description' in data:
        description = data['description']
        if not isinstance(description, str):
            raise BadRequest("Description must be a string")
        
        if len(description) > 2000:
            raise BadRequest("Description must be less than 2000 characters")
    
    # Validate config
    if 'config' in data:
        config = data['config']
        if not isinstance(config, dict):
            raise BadRequest("Config must be a JSON object")
        
        # Validate config structure
        _validate_experiment_config(config)
    
    # Validate status if provided
    if 'status' in data:
        status = data['status']
        valid_statuses = ['pending', 'running', 'completed', 'failed', 'cancelled']
        if status not in valid_statuses:
            raise BadRequest(f"Status must be one of: {', '.join(valid_statuses)}")


def _validate_experiment_config(config: Dict[str, Any]) -> None:
    """Validate experiment configuration.
    
    Args:
        config: Configuration dictionary
        
    Raises:
        BadRequest: If validation fails
    """
    # Validate agent_type if provided
    if 'agent_type' in config:
        agent_type = config['agent_type']
        valid_types = ['random', 'scripted', 'rl', 'llm', 'custom']
        if agent_type not in valid_types:
            raise BadRequest(f"agent_type must be one of: {', '.join(valid_types)}")
    
    # Validate num_episodes if provided
    if 'num_episodes' in config:
        num_episodes = config['num_episodes']
        if not isinstance(num_episodes, int) or num_episodes <= 0:
            raise BadRequest("num_episodes must be a positive integer")
        
        if num_episodes > 10000:
            raise BadRequest("num_episodes must be less than or equal to 10000")
    
    # Validate tasks if provided
    if 'tasks' in config:
        tasks = config['tasks']
        if not isinstance(tasks, list):
            raise BadRequest("tasks must be a list")
        
        for task in tasks:
            if not isinstance(task, str):
                raise BadRequest("All tasks must be strings")
    
    # Validate metrics if provided
    if 'metrics' in config:
        metrics = config['metrics']
        if not isinstance(metrics, list):
            raise BadRequest("metrics must be a list")
        
        valid_metrics = ['success_rate', 'efficiency', 'safety', 'collaboration', 'diversity']
        for metric in metrics:
            if metric not in valid_metrics:
                raise BadRequest(f"Invalid metric '{metric}'. Valid metrics: {', '.join(valid_metrics)}")


def validate_run_data(data: Dict[str, Any]) -> None:
    """Validate benchmark run creation data.
    
    Args:
        data: Run data dictionary
        
    Raises:
        BadRequest: If validation fails
    """
    # Required fields
    required_fields = ['experiment_id', 'agent_name', 'task_name']
    for field in required_fields:
        if field not in data:
            raise BadRequest(f"{field} is required")
    
    # Validate experiment_id
    experiment_id = data['experiment_id']
    if not isinstance(experiment_id, int) or experiment_id <= 0:
        raise BadRequest("experiment_id must be a positive integer")
    
    # Validate agent_name
    agent_name = data['agent_name']
    if not isinstance(agent_name, str) or not agent_name.strip():
        raise BadRequest("agent_name must be a non-empty string")
    
    if len(agent_name) > 100:
        raise BadRequest("agent_name must be less than 100 characters")
    
    # Validate task_name
    task_name = data['task_name']
    if not isinstance(task_name, str) or not task_name.strip():
        raise BadRequest("task_name must be a non-empty string")
    
    if len(task_name) > 100:
        raise BadRequest("task_name must be less than 100 characters")
    
    # Validate config if provided
    if 'config' in data:
        config = data['config']
        if not isinstance(config, dict):
            raise BadRequest("config must be a JSON object")
        
        _validate_run_config(config)


def _validate_run_config(config: Dict[str, Any]) -> None:
    """Validate run configuration.
    
    Args:
        config: Configuration dictionary
        
    Raises:
        BadRequest: If validation fails
    """
    # Validate num_episodes if provided
    if 'num_episodes' in config:
        num_episodes = config['num_episodes']
        if not isinstance(num_episodes, int) or num_episodes <= 0:
            raise BadRequest("num_episodes must be a positive integer")
        
        if num_episodes > 1000:
            raise BadRequest("num_episodes must be less than or equal to 1000")
    
    # Validate timeout if provided
    if 'timeout' in config:
        timeout = config['timeout']
        if not isinstance(timeout, (int, float)) or timeout <= 0:
            raise BadRequest("timeout must be a positive number")
        
        if timeout > 3600:  # 1 hour max
            raise BadRequest("timeout must be less than or equal to 3600 seconds")
    
    # Validate parallel if provided
    if 'parallel' in config:
        parallel = config['parallel']
        if not isinstance(parallel, bool):
            raise BadRequest("parallel must be a boolean")
    
    # Validate num_workers if provided
    if 'num_workers' in config:
        num_workers = config['num_workers']
        if not isinstance(num_workers, int) or num_workers <= 0:
            raise BadRequest("num_workers must be a positive integer")
        
        if num_workers > 16:
            raise BadRequest("num_workers must be less than or equal to 16")


def validate_agent_data(data: Dict[str, Any]) -> None:
    """Validate agent configuration data.
    
    Args:
        data: Agent data dictionary
        
    Raises:
        BadRequest: If validation fails
    """
    # Required fields
    if 'name' not in data:
        raise BadRequest("Agent name is required")
    
    # Validate name
    name = data['name']
    if not isinstance(name, str) or not name.strip():
        raise BadRequest("Agent name must be a non-empty string")
    
    if len(name) > 100:
        raise BadRequest("Agent name must be less than 100 characters")
    
    # Validate type if provided
    if 'type' in data:
        agent_type = data['type']
        valid_types = ['random', 'scripted', 'rl', 'llm', 'custom']
        if agent_type not in valid_types:
            raise BadRequest(f"Agent type must be one of: {', '.join(valid_types)}")
    
    # Validate config if provided
    if 'config' in data:
        config = data['config']
        if not isinstance(config, dict):
            raise BadRequest("Agent config must be a JSON object")


def validate_task_data(data: Dict[str, Any]) -> None:
    """Validate task configuration data.
    
    Args:
        data: Task data dictionary
        
    Raises:
        BadRequest: If validation fails
    """
    # Required fields
    if 'name' not in data:
        raise BadRequest("Task name is required")
    
    # Validate name
    name = data['name']
    if not isinstance(name, str) or not name.strip():
        raise BadRequest("Task name must be a non-empty string")
    
    if len(name) > 100:
        raise BadRequest("Task name must be less than 100 characters")
    
    # Validate type if provided
    if 'type' in data:
        task_type = data['type']
        valid_types = ['manipulation', 'navigation', 'multiagent', 'custom']
        if task_type not in valid_types:
            raise BadRequest(f"Task type must be one of: {', '.join(valid_types)}")
    
    # Validate difficulty if provided
    if 'difficulty' in data:
        difficulty = data['difficulty']
        valid_difficulties = ['easy', 'medium', 'hard', 'expert']
        if difficulty not in valid_difficulties:
            raise BadRequest(f"Difficulty must be one of: {', '.join(valid_difficulties)}")
    
    # Validate max_steps if provided
    if 'max_steps' in data:
        max_steps = data['max_steps']
        if not isinstance(max_steps, int) or max_steps <= 0:
            raise BadRequest("max_steps must be a positive integer")
        
        if max_steps > 10000:
            raise BadRequest("max_steps must be less than or equal to 10000")
    
    # Validate time_limit if provided
    if 'time_limit' in data:
        time_limit = data['time_limit']
        if not isinstance(time_limit, (int, float)) or time_limit <= 0:
            raise BadRequest("time_limit must be a positive number")
        
        if time_limit > 3600:  # 1 hour max
            raise BadRequest("time_limit must be less than or equal to 3600 seconds")


def validate_pagination_params(page: int, page_size: int) -> None:
    """Validate pagination parameters.
    
    Args:
        page: Page number
        page_size: Number of items per page
        
    Raises:
        BadRequest: If validation fails
    """
    if page < 1:
        raise BadRequest("page must be greater than or equal to 1")
    
    if page_size < 1:
        raise BadRequest("page_size must be greater than or equal to 1")
    
    if page_size > 100:
        raise BadRequest("page_size must be less than or equal to 100")


def validate_results_data(data: Dict[str, Any]) -> None:
    """Validate benchmark results data.
    
    Args:
        data: Results data dictionary
        
    Raises:
        BadRequest: If validation fails
    """
    # Validate num_episodes if provided
    if 'num_episodes' in data:
        num_episodes = data['num_episodes']
        if not isinstance(num_episodes, int) or num_episodes <= 0:
            raise BadRequest("num_episodes must be a positive integer")
    
    # Validate success_rate if provided
    if 'success_rate' in data:
        success_rate = data['success_rate']
        if not isinstance(success_rate, (int, float)) or not (0.0 <= success_rate <= 1.0):
            raise BadRequest("success_rate must be a number between 0.0 and 1.0")
    
    # Validate avg_reward if provided
    if 'avg_reward' in data:
        avg_reward = data['avg_reward']
        if not isinstance(avg_reward, (int, float)):
            raise BadRequest("avg_reward must be a number")
    
    # Validate avg_steps if provided
    if 'avg_steps' in data:
        avg_steps = data['avg_steps']
        if not isinstance(avg_steps, (int, float)) or avg_steps < 0:
            raise BadRequest("avg_steps must be a non-negative number")
    
    # Validate total_time if provided
    if 'total_time' in data:
        total_time = data['total_time']
        if not isinstance(total_time, (int, float)) or total_time < 0:
            raise BadRequest("total_time must be a non-negative number")
    
    # Validate metrics if provided
    if 'metrics' in data:
        metrics = data['metrics']
        if not isinstance(metrics, dict):
            raise BadRequest("metrics must be a JSON object")
        
        for metric_name, metric_data in metrics.items():
            if not isinstance(metric_data, dict):
                raise BadRequest(f"Metric '{metric_name}' data must be a JSON object")
            
            # Validate metric values
            for key, value in metric_data.items():
                if key in ['mean', 'std', 'min', 'max'] and not isinstance(value, (int, float)):
                    raise BadRequest(f"Metric '{metric_name}.{key}' must be a number")


def validate_json_size(data: Dict[str, Any], max_size_mb: float = 10.0) -> None:
    """Validate JSON data size.
    
    Args:
        data: JSON data dictionary
        max_size_mb: Maximum size in megabytes
        
    Raises:
        BadRequest: If data is too large
    """
    import json
    import sys
    
    # Estimate JSON size
    json_str = json.dumps(data)
    size_bytes = sys.getsizeof(json_str)
    size_mb = size_bytes / (1024 * 1024)
    
    if size_mb > max_size_mb:
        raise BadRequest(f"JSON data size ({size_mb:.2f}MB) exceeds maximum allowed size ({max_size_mb}MB)")

"""Factory for creating benchmark tasks and environments."""

from typing import Any, Dict, Optional
import numpy as np

from ..core.base_task import BaseTask
from ..core.base_env import BaseEnv
from .manipulation.furniture_assembly import FurnitureAssemblyTask, FurnitureAssemblyEnv
from .navigation.point_goal import PointGoalTask, PointGoalEnv
from .multiagent.cooperative_assembly import CooperativeAssemblyTask, CooperativeAssemblyEnv


# Registry of available tasks
TASK_REGISTRY = {
    "FurnitureAssembly-v0": {
        "task_class": FurnitureAssemblyTask,
        "env_class": FurnitureAssemblyEnv,
        "default_config": {
            "furniture_type": "table",
            "difficulty": "medium",
            "max_steps": 1000,
            "time_limit": 300
        }
    },
    "PointGoal-v0": {
        "task_class": PointGoalTask,
        "env_class": PointGoalEnv,
        "default_config": {
            "room_size": (10, 10),
            "goal_radius": 0.5,
            "max_steps": 500
        }
    },
    "CooperativeFurnitureAssembly-v0": {
        "task_class": CooperativeAssemblyTask,
        "env_class": CooperativeAssemblyEnv,
        "default_config": {
            "num_agents": 2,
            "furniture": "ikea_table",
            "difficulty": "medium",
            "max_steps": 1500
        }
    }
}


def make_env(task_id: str, 
             simulator: str = "habitat",
             render_mode: str = "rgb_array",
             **kwargs) -> BaseEnv:
    """Create environment for specified task.
    
    Args:
        task_id: Task identifier (e.g., "FurnitureAssembly-v0")
        simulator: Simulator backend ("habitat", "maniskill", "isaac")
        render_mode: Rendering mode ("rgb_array", "human")
        **kwargs: Additional configuration options
        
    Returns:
        Configured environment instance
        
    Raises:
        ValueError: If task_id is not registered or parameters are invalid
        RuntimeError: If environment creation fails
    """
    # Input validation
    if not isinstance(task_id, str) or not task_id.strip():
        raise ValueError("task_id must be a non-empty string")
    
    if simulator not in ["habitat", "maniskill", "isaac", "mock"]:
        raise ValueError(f"Unsupported simulator '{simulator}'. Supported: habitat, maniskill, isaac, mock")
    
    if render_mode not in ["rgb_array", "human", "none"]:
        raise ValueError(f"Unsupported render mode '{render_mode}'. Supported: rgb_array, human, none")
    
    if task_id not in TASK_REGISTRY:
        available_tasks = list(TASK_REGISTRY.keys())
        raise ValueError(f"Unknown task '{task_id}'. Available tasks: {available_tasks}")
    
    try:
        task_info = TASK_REGISTRY[task_id]
        
        # Validate and merge configuration
        config = task_info["default_config"].copy()
        
        # Validate kwargs parameters
        for key, value in kwargs.items():
            if key in ["max_steps", "time_limit", "num_agents"]:
                if not isinstance(value, int) or value <= 0:
                    raise ValueError(f"Parameter '{key}' must be a positive integer, got {value}")
            elif key == "difficulty":
                if value not in ["easy", "medium", "hard"] and not isinstance(value, (int, float)):
                    raise ValueError(f"Difficulty must be 'easy', 'medium', 'hard', or numeric, got {value}")
        
        config.update(kwargs)
        config.update({
            "task_id": task_id,
            "simulator": simulator,
            "render_mode": render_mode
        })
        
        # Create task and environment with error handling
        try:
            task = task_info["task_class"](config)
        except Exception as e:
            raise RuntimeError(f"Failed to create task '{task_id}': {str(e)}") from e
        
        try:
            env = task_info["env_class"](config)
        except Exception as e:
            raise RuntimeError(f"Failed to create environment for '{task_id}': {str(e)}") from e
        
        # Bind task to environment
        if hasattr(env, 'set_task'):
            env.set_task(task)
        else:
            raise RuntimeError(f"Environment class {task_info['env_class'].__name__} does not support task binding")
        
        return env
        
    except Exception as e:
        if isinstance(e, (ValueError, RuntimeError)):
            raise
        else:
            raise RuntimeError(f"Unexpected error creating environment for '{task_id}': {str(e)}") from e


def make_task(task_id: str, **kwargs) -> BaseTask:
    """Create task instance for specified task ID.
    
    Args:
        task_id: Task identifier
        **kwargs: Task configuration options
        
    Returns:
        Configured task instance
        
    Raises:
        ValueError: If task_id is not registered
    """
    if task_id not in TASK_REGISTRY:
        available_tasks = list(TASK_REGISTRY.keys())
        raise ValueError(f"Unknown task '{task_id}'. Available tasks: {available_tasks}")
    
    task_info = TASK_REGISTRY[task_id]
    
    # Merge default config with provided kwargs
    config = task_info["default_config"].copy()
    config.update(kwargs)
    config["task_id"] = task_id
    
    return task_info["task_class"](config)


def register_task(task_id: str,
                 task_class: type,
                 env_class: type,
                 default_config: Optional[Dict[str, Any]] = None):
    """Register a new task in the factory.
    
    Args:
        task_id: Unique task identifier
        task_class: Task class (must inherit from BaseTask)
        env_class: Environment class (must inherit from BaseEnv)
        default_config: Default configuration dictionary
    """
    if not issubclass(task_class, BaseTask):
        raise ValueError("task_class must inherit from BaseTask")
    
    if not issubclass(env_class, BaseEnv):
        raise ValueError("env_class must inherit from BaseEnv")
    
    TASK_REGISTRY[task_id] = {
        "task_class": task_class,
        "env_class": env_class,
        "default_config": default_config or {}
    }


def get_available_tasks() -> list:
    """Get list of all available task IDs.
    
    Returns:
        List of registered task identifiers
    """
    return list(TASK_REGISTRY.keys())


def get_task_info(task_id: str) -> Dict[str, Any]:
    """Get information about a specific task.
    
    Args:
        task_id: Task identifier
        
    Returns:
        Task information dictionary
        
    Raises:
        ValueError: If task_id is not registered
    """
    if task_id not in TASK_REGISTRY:
        raise ValueError(f"Unknown task '{task_id}'")
    
    task_info = TASK_REGISTRY[task_id].copy()
    return {
        "task_id": task_id,
        "task_class": task_info["task_class"].__name__,
        "env_class": task_info["env_class"].__name__,
        "default_config": task_info["default_config"]
    }


class TaskBuilder:
    """Builder class for creating custom tasks."""
    
    @staticmethod
    def register(task_id: str):
        """Decorator for registering custom tasks.
        
        Args:
            task_id: Unique task identifier
            
        Returns:
            Decorator function
        """
        def decorator(task_class):
            # Auto-generate simple environment if not provided
            class AutoEnv(BaseEnv):
                def __init__(self, config):
                    super().__init__(config)
                    self.task = None
                
                def set_task(self, task):
                    self.task = task
                
                def reset(self, seed=None):
                    if self.task:
                        return self.task.reset()
                    return {"dummy": np.zeros(10)}
                
                def step(self, action):
                    if self.task:
                        return self.task.step(action)
                    return {"dummy": np.zeros(10)}, 0.0, True, {}
                
                def render(self, mode="rgb_array"):
                    return np.zeros((480, 640, 3), dtype=np.uint8)
                
                def close(self):
                    pass
                
                def _get_observation(self):
                    return {"dummy": np.zeros(10)}
            
            register_task(task_id, task_class, AutoEnv)
            return task_class
        
        return decorator
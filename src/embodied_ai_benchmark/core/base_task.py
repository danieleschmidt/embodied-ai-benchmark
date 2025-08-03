"""Base task interface for all benchmark tasks."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple
import numpy as np


class BaseTask(ABC):
    """Base class for all benchmark tasks."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize task with configuration.
        
        Args:
            config: Task configuration dictionary
        """
        self.config = config
        self.name = config.get("name", "unknown_task")
        self.difficulty = config.get("difficulty", "medium")
        self.time_limit = config.get("time_limit", 300)  # seconds
        self.current_step = 0
        self.max_steps = config.get("max_steps", 1000)
        self._episode_data = []
        
    @abstractmethod
    def reset(self) -> Dict[str, Any]:
        """Reset the task to initial state.
        
        Returns:
            Initial observation dictionary
        """
        self.current_step = 0
        self._episode_data = []
        return self._get_observation()
    
    @abstractmethod
    def step(self, action: Dict[str, Any]) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        """Execute one step of the task.
        
        Args:
            action: Action dictionary from agent
            
        Returns:
            Tuple of (observation, reward, done, info)
        """
        pass
    
    @abstractmethod
    def _get_observation(self) -> Dict[str, Any]:
        """Get current observation.
        
        Returns:
            Observation dictionary
        """
        pass
    
    @abstractmethod
    def check_success(self) -> bool:
        """Check if task has been completed successfully.
        
        Returns:
            True if task is completed successfully
        """
        pass
    
    @abstractmethod
    def compute_reward(self) -> float:
        """Compute reward for current state.
        
        Returns:
            Scalar reward value
        """
        pass
    
    def get_action_space(self) -> Dict[str, Any]:
        """Get action space specification.
        
        Returns:
            Action space dictionary
        """
        return {
            "type": "continuous",
            "shape": (7,),  # 3D position + 4D quaternion
            "low": np.array([-1] * 7),
            "high": np.array([1] * 7)
        }
    
    def get_observation_space(self) -> Dict[str, Any]:
        """Get observation space specification.
        
        Returns:
            Observation space dictionary
        """
        return {
            "rgb": {"shape": (480, 640, 3), "dtype": "uint8"},
            "depth": {"shape": (480, 640), "dtype": "float32"},
            "proprioception": {"shape": (12,), "dtype": "float32"}
        }
    
    def get_episode_data(self) -> List[Dict[str, Any]]:
        """Get data collected during episode.
        
        Returns:
            List of step data dictionaries
        """
        return self._episode_data
    
    def is_done(self) -> bool:
        """Check if episode is complete.
        
        Returns:
            True if episode is complete
        """
        return (self.current_step >= self.max_steps or 
                self.check_success())
    
    def get_task_info(self) -> Dict[str, Any]:
        """Get task metadata and info.
        
        Returns:
            Task information dictionary
        """
        return {
            "name": self.name,
            "difficulty": self.difficulty,
            "time_limit": self.time_limit,
            "max_steps": self.max_steps,
            "current_step": self.current_step,
            "success": self.check_success()
        }
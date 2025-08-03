"""Base environment interface for simulator integration."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple
import numpy as np


class BaseEnv(ABC):
    """Base environment interface for all simulators."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize environment with configuration.
        
        Args:
            config: Environment configuration dictionary
        """
        self.config = config
        self.simulator_name = config.get("simulator", "unknown")
        self.render_mode = config.get("render_mode", "rgb_array")
        self.num_agents = config.get("num_agents", 1)
        self._agents = {}
        self._objects = {}
        self._episode_count = 0
        
    @abstractmethod
    def reset(self, seed: Optional[int] = None) -> Dict[str, Any]:
        """Reset environment to initial state.
        
        Args:
            seed: Random seed for reproducibility
            
        Returns:
            Initial observation dictionary
        """
        if seed is not None:
            np.random.seed(seed)
        self._episode_count += 1
        return self._get_observation()
    
    @abstractmethod
    def step(self, action: Dict[str, Any]) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        """Execute environment step with action.
        
        Args:
            action: Action dictionary
            
        Returns:
            Tuple of (observation, reward, done, info)
        """
        pass
    
    @abstractmethod
    def render(self, mode: str = "rgb_array") -> Optional[np.ndarray]:
        """Render environment visualization.
        
        Args:
            mode: Rendering mode ("rgb_array", "human")
            
        Returns:
            Rendered image array if mode is "rgb_array"
        """
        pass
    
    @abstractmethod
    def close(self):
        """Clean up environment resources."""
        pass
    
    @abstractmethod
    def _get_observation(self) -> Dict[str, Any]:
        """Get current observation from environment.
        
        Returns:
            Observation dictionary
        """
        pass
    
    def seed(self, seed: int):
        """Set random seed.
        
        Args:
            seed: Random seed value
        """
        np.random.seed(seed)
    
    def get_objects(self) -> Dict[str, Any]:
        """Get all objects in the environment.
        
        Returns:
            Dictionary of object IDs to object info
        """
        return self._objects.copy()
    
    def get_agents(self) -> Dict[str, Any]:
        """Get all agents in the environment.
        
        Returns:
            Dictionary of agent IDs to agent info
        """
        return self._agents.copy()
    
    def add_object(self, obj_id: str, obj_info: Dict[str, Any]):
        """Add object to environment.
        
        Args:
            obj_id: Unique object identifier
            obj_info: Object information dictionary
        """
        self._objects[obj_id] = obj_info
    
    def remove_object(self, obj_id: str):
        """Remove object from environment.
        
        Args:
            obj_id: Object identifier to remove
        """
        if obj_id in self._objects:
            del self._objects[obj_id]
    
    def get_state(self) -> Dict[str, Any]:
        """Get current environment state.
        
        Returns:
            Complete environment state dictionary
        """
        return {
            "simulator": self.simulator_name,
            "episode_count": self._episode_count,
            "num_agents": self.num_agents,
            "objects": self._objects,
            "agents": self._agents
        }
    
    def set_state(self, state: Dict[str, Any]):
        """Set environment state.
        
        Args:
            state: Environment state dictionary
        """
        self._objects = state.get("objects", {})
        self._agents = state.get("agents", {})
        self._episode_count = state.get("episode_count", 0)
    
    def get_physics_info(self) -> Dict[str, Any]:
        """Get physics simulation information.
        
        Returns:
            Physics information dictionary
        """
        return {
            "gravity": 9.81,
            "time_step": 0.01,
            "physics_enabled": True
        }
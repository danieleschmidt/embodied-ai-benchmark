"""Base agent interface for benchmark evaluation."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
import numpy as np


class BaseAgent(ABC):
    """Base agent interface for all benchmark agents."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize agent with configuration.
        
        Args:
            config: Agent configuration dictionary
        """
        self.config = config
        self.agent_id = config.get("agent_id", "agent_0")
        self.role = config.get("role", "default")
        self.capabilities = config.get("capabilities", [])
        self._action_history = []
        self._observation_history = []
        
    @abstractmethod
    def reset(self):
        """Reset agent to initial state."""
        self._action_history = []
        self._observation_history = []
    
    @abstractmethod
    def act(self, observation: Dict[str, Any]) -> Dict[str, Any]:
        """Select action based on observation.
        
        Args:
            observation: Current observation dictionary
            
        Returns:
            Action dictionary
        """
        pass
    
    @abstractmethod
    def update(self, 
               observation: Dict[str, Any], 
               action: Dict[str, Any], 
               reward: float, 
               next_observation: Dict[str, Any], 
               done: bool):
        """Update agent with experience.
        
        Args:
            observation: Previous observation
            action: Action taken
            reward: Reward received
            next_observation: Next observation
            done: Whether episode is complete
        """
        pass
    
    def get_capabilities(self) -> List[str]:
        """Get agent capabilities.
        
        Returns:
            List of capability strings
        """
        return self.capabilities.copy()
    
    def get_action_history(self) -> List[Dict[str, Any]]:
        """Get history of actions taken.
        
        Returns:
            List of action dictionaries
        """
        return self._action_history.copy()
    
    def get_observation_history(self) -> List[Dict[str, Any]]:
        """Get history of observations received.
        
        Returns:
            List of observation dictionaries
        """
        return self._observation_history.copy()
    
    def send_message(self, recipient: str, message: Dict[str, Any]):
        """Send message to another agent.
        
        Args:
            recipient: Target agent ID
            message: Message dictionary
        """
        pass
    
    def receive_message(self, sender: str, message: Dict[str, Any]):
        """Receive message from another agent.
        
        Args:
            sender: Sender agent ID
            message: Message dictionary
        """
        pass
    
    def get_performance_metrics(self) -> Dict[str, float]:
        """Get agent performance metrics.
        
        Returns:
            Dictionary of metric names to values
        """
        return {
            "actions_taken": len(self._action_history),
            "observations_received": len(self._observation_history)
        }


class RandomAgent(BaseAgent):
    """Random agent for baseline evaluation."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.action_space = config.get("action_space", {
            "type": "continuous",
            "shape": (7,),
            "low": np.array([-1] * 7),
            "high": np.array([1] * 7)
        })
    
    def reset(self):
        super().reset()
    
    def act(self, observation: Dict[str, Any]) -> Dict[str, Any]:
        self._observation_history.append(observation)
        
        if self.action_space["type"] == "continuous":
            action_values = np.random.uniform(
                self.action_space["low"],
                self.action_space["high"]
            )
            action = {
                "type": "continuous",
                "values": action_values
            }
        else:
            action = {
                "type": "discrete",
                "values": np.random.randint(0, self.action_space.get("n", 4))
            }
        
        self._action_history.append(action)
        return action
    
    def update(self, observation, action, reward, next_observation, done):
        pass


class ScriptedAgent(BaseAgent):
    """Scripted agent with predefined behaviors."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.script = config.get("script", [])
        self.script_index = 0
    
    def reset(self):
        super().reset()
        self.script_index = 0
    
    def act(self, observation: Dict[str, Any]) -> Dict[str, Any]:
        self._observation_history.append(observation)
        
        if self.script_index < len(self.script):
            action = self.script[self.script_index]
            self.script_index += 1
        else:
            action = {"type": "no_op", "values": [0]}
        
        self._action_history.append(action)
        return action
    
    def update(self, observation, action, reward, next_observation, done):
        pass
"""Base metric interface for evaluation metrics."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
import numpy as np


class BaseMetric(ABC):
    """Base class for all evaluation metrics."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize metric with configuration.
        
        Args:
            config: Metric configuration dictionary
        """
        self.config = config
        self.name = config.get("name", "unknown_metric")
        self.weight = config.get("weight", 1.0)
        self._data = []
        
    @abstractmethod
    def reset(self):
        """Reset metric state for new episode."""
        self._data = []
    
    @abstractmethod
    def update(self, 
               observation: Dict[str, Any], 
               action: Dict[str, Any], 
               reward: float, 
               next_observation: Dict[str, Any], 
               done: bool,
               info: Dict[str, Any]):
        """Update metric with step data.
        
        Args:
            observation: Previous observation
            action: Action taken
            reward: Reward received
            next_observation: Next observation
            done: Whether episode is complete
            info: Additional step information
        """
        pass
    
    @abstractmethod
    def compute(self) -> float:
        """Compute final metric value.
        
        Returns:
            Computed metric value
        """
        pass
    
    def get_name(self) -> str:
        """Get metric name.
        
        Returns:
            Metric name string
        """
        return self.name
    
    def get_weight(self) -> float:
        """Get metric weight for aggregation.
        
        Returns:
            Metric weight
        """
        return self.weight
    
    def get_data(self) -> List[Any]:
        """Get collected data.
        
        Returns:
            List of collected data points
        """
        return self._data.copy()


class SuccessMetric(BaseMetric):
    """Task success rate metric."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.name = "success_rate"
        self.task_completed = False
        
    def reset(self):
        super().reset()
        self.task_completed = False
    
    def update(self, observation, action, reward, next_observation, done, info):
        self.task_completed = info.get("task_success", False)
        self._data.append({
            "step": len(self._data),
            "success": self.task_completed,
            "done": done
        })
    
    def compute(self) -> float:
        return float(self.task_completed)


class EfficiencyMetric(BaseMetric):
    """Task completion efficiency metric."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.name = "efficiency"
        self.total_steps = 0
        self.max_steps = config.get("max_steps", 1000)
        
    def reset(self):
        super().reset()
        self.total_steps = 0
    
    def update(self, observation, action, reward, next_observation, done, info):
        self.total_steps += 1
        self._data.append({
            "step": self.total_steps,
            "reward": reward,
            "done": done
        })
    
    def compute(self) -> float:
        if self.total_steps == 0:
            return 0.0
        return 1.0 - (self.total_steps / self.max_steps)


class SafetyMetric(BaseMetric):
    """Safety violations metric."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.name = "safety"
        self.collision_count = 0
        self.force_violations = 0
        self.max_force = config.get("max_force", 100.0)  # Newtons
        
    def reset(self):
        super().reset()
        self.collision_count = 0
        self.force_violations = 0
    
    def update(self, observation, action, reward, next_observation, done, info):
        # Check for collisions
        if info.get("collision", False):
            self.collision_count += 1
        
        # Check for force violations
        force = info.get("applied_force", 0.0)
        if force > self.max_force:
            self.force_violations += 1
        
        self._data.append({
            "step": len(self._data),
            "collision": info.get("collision", False),
            "force": force,
            "force_violation": force > self.max_force
        })
    
    def compute(self) -> float:
        total_steps = len(self._data)
        if total_steps == 0:
            return 1.0
        
        total_violations = self.collision_count + self.force_violations
        return max(0.0, 1.0 - (total_violations / total_steps))


class CollaborationMetric(BaseMetric):
    """Multi-agent collaboration quality metric."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.name = "collaboration"
        self.coordination_events = 0
        self.communication_count = 0
        self.conflict_count = 0
        
    def reset(self):
        super().reset()
        self.coordination_events = 0
        self.communication_count = 0
        self.conflict_count = 0
    
    def update(self, observation, action, reward, next_observation, done, info):
        # Track coordination events
        if info.get("coordination_event", False):
            self.coordination_events += 1
        
        # Track communication
        if info.get("message_sent", False):
            self.communication_count += 1
        
        # Track conflicts
        if info.get("agent_conflict", False):
            self.conflict_count += 1
        
        self._data.append({
            "step": len(self._data),
            "coordination": info.get("coordination_event", False),
            "communication": info.get("message_sent", False),
            "conflict": info.get("agent_conflict", False)
        })
    
    def compute(self) -> float:
        total_steps = len(self._data)
        if total_steps == 0:
            return 0.0
        
        # Positive scoring for coordination and communication
        positive_score = (self.coordination_events + self.communication_count) / total_steps
        
        # Negative scoring for conflicts
        negative_score = self.conflict_count / total_steps
        
        return max(0.0, positive_score - negative_score)
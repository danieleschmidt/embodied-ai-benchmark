"""Base task interface for all benchmark tasks."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import time
from datetime import datetime


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
        self.quantum_state = None
        self.curriculum_level = config.get("curriculum_level", 1)
        self.adaptation_history = []
        self.performance_metrics = {
            "success_rate": 0.0,
            "efficiency": 0.0,
            "learning_progress": 0.0
        }
        self.start_time = None
        
    @abstractmethod
    def reset(self) -> Dict[str, Any]:
        """Reset the task to initial state with quantum-inspired initialization.
        
        Returns:
            Initial observation dictionary
        """
        self.current_step = 0
        self._episode_data = []
        self.start_time = time.time()
        
        # Initialize quantum state for task planning
        self.quantum_state = self._initialize_quantum_state()
        
        # Adapt difficulty based on curriculum level
        self._adapt_task_difficulty()
        
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
            "success": self.check_success(),
            "curriculum_level": self.curriculum_level,
            "quantum_coherence": self._get_quantum_coherence(),
            "adaptation_score": self._compute_adaptation_score(),
            "elapsed_time": time.time() - self.start_time if self.start_time else 0
        }
    
    def _initialize_quantum_state(self) -> Dict[str, np.ndarray]:
        """Initialize quantum-inspired state for task planning."""
        state_dim = min(64, self.max_steps // 10)
        amplitudes = np.ones(state_dim, dtype=complex) / np.sqrt(state_dim)
        phases = np.linspace(0, 2 * np.pi, state_dim)
        amplitudes *= np.exp(1j * phases)
        
        return {
            "amplitudes": amplitudes,
            "measurement_history": [],
            "entanglement_matrix": np.eye(state_dim, dtype=complex),
            "coherence_time": 0
        }
    
    def _get_quantum_coherence(self) -> float:
        """Compute quantum coherence of current task state."""
        if self.quantum_state is None:
            return 0.0
        
        amplitudes = self.quantum_state["amplitudes"]
        density_matrix = np.outer(amplitudes, np.conj(amplitudes))
        eigenvals = np.real(np.linalg.eigvals(density_matrix))
        eigenvals = eigenvals[eigenvals > 1e-12]
        
        if len(eigenvals) == 0:
            return 0.0
        
        entropy = -np.sum(eigenvals * np.log2(eigenvals + 1e-12))
        max_entropy = np.log2(len(amplitudes))
        return entropy / max_entropy if max_entropy > 0 else 0.0
    
    def _adapt_task_difficulty(self):
        """Adapt task difficulty based on curriculum level."""
        if self.curriculum_level <= 1:
            self.max_steps = int(self.max_steps * 1.5)
            self.time_limit = int(self.time_limit * 1.3)
        elif self.curriculum_level >= 5:
            self.max_steps = int(self.max_steps * 0.7)
            self.time_limit = int(self.time_limit * 0.8)
        
        self.adaptation_history.append({
            "timestamp": datetime.now().isoformat(),
            "level": self.curriculum_level,
            "max_steps": self.max_steps,
            "time_limit": self.time_limit
        })
    
    def _compute_adaptation_score(self) -> float:
        """Compute adaptation score based on performance metrics."""
        if not self.adaptation_history:
            return 0.5
        
        recent_success = self.performance_metrics["success_rate"]
        efficiency = self.performance_metrics["efficiency"] 
        learning_progress = self.performance_metrics["learning_progress"]
        
        challenge_score = 1.0 - abs(recent_success - 0.7) / 0.3
        challenge_score = max(0.0, min(1.0, challenge_score))
        
        return max(0.0, min(1.0, (
            0.5 * challenge_score +
            0.3 * efficiency +
            0.2 * learning_progress
        )))
    
    def update_performance_metrics(self, success: bool, steps_taken: int, reward: float):
        """Update performance metrics for curriculum adaptation."""
        alpha = 0.1
        self.performance_metrics["success_rate"] = (
            alpha * float(success) + 
            (1 - alpha) * self.performance_metrics["success_rate"]
        )
        
        efficiency = max(0.0, 1.0 - (steps_taken / self.max_steps))
        self.performance_metrics["efficiency"] = (
            alpha * efficiency + 
            (1 - alpha) * self.performance_metrics["efficiency"]
        )
        
        if hasattr(self, '_last_reward'):
            progress = max(0.0, min(1.0, (reward - self._last_reward) / 100.0 + 0.5))
            self.performance_metrics["learning_progress"] = (
                alpha * progress + 
                (1 - alpha) * self.performance_metrics["learning_progress"]
            )
        self._last_reward = reward
        
        self._auto_adjust_curriculum()
    
    def _auto_adjust_curriculum(self):
        """Auto-adjust curriculum level based on performance."""
        success_rate = self.performance_metrics["success_rate"]
        efficiency = self.performance_metrics["efficiency"]
        
        if success_rate > 0.8 and efficiency > 0.7:
            self.curriculum_level = min(10, self.curriculum_level + 1)
        elif success_rate < 0.3 or efficiency < 0.3:
            self.curriculum_level = max(1, self.curriculum_level - 1)
    
    def get_quantum_plan(self, goal_state: Optional[Dict[str, Any]] = None) -> List[str]:
        """Generate quantum-inspired action plan."""
        if self.quantum_state is None:
            return ["explore", "manipulate", "verify"]
        
        amplitudes = self.quantum_state["amplitudes"]
        state_dim = len(amplitudes)
        probabilities = np.abs(amplitudes) ** 2
        
        plan = []
        for i in range(min(5, state_dim // 10)):
            action_idx = np.random.choice(state_dim, p=probabilities)
            
            if action_idx < state_dim // 3:
                plan.append("explore")
            elif action_idx < 2 * state_dim // 3:
                plan.append("manipulate")
            else:
                plan.append("verify")
        
        # Apply decoherence
        self.quantum_state["coherence_time"] += 1
        decoherence_factor = np.exp(-self.quantum_state["coherence_time"] / 100)
        self.quantum_state["amplitudes"] *= decoherence_factor
        
        return plan
"""Base class for embodied AI benchmark tasks."""

from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, Tuple, List

from ..envs.gridworld import GridWorld, GridState, Action


@dataclass
class TaskResult:
    """Outcome of running one agent on one task episode."""
    task_name: str
    success: bool
    steps_taken: int
    optimal_steps: Optional[int]  # BFS optimal; None if N/A
    efficiency: float              # optimal/taken clamped [0,1]; 1.0 if not applicable
    extra: Dict[str, Any] = field(default_factory=dict)

    @property
    def efficiency_pct(self) -> float:
        return round(self.efficiency * 100, 1)


class EmbodiedTask(ABC):
    """
    Abstract base for benchmark tasks.

    Subclasses implement:
      - build_env() → GridWorld
      - optimal_steps() → Optional[int]
      - is_success(state) → bool
      - is_done(state) → bool   (terminal condition including failure/timeout)
    """

    name: str = "abstract_task"

    @abstractmethod
    def build_env(self) -> GridWorld:
        """Construct and return a fresh GridWorld for this task."""
        ...

    @abstractmethod
    def is_success(self, env: GridWorld, state: GridState) -> bool:
        """Return True when the agent has completed the task."""
        ...

    def is_done(self, env: GridWorld, state: GridState) -> bool:
        """Return True to end the episode (success OR failure/timeout)."""
        return self.is_success(env, state) or state.step_count >= env.max_steps

    def optimal_steps(self, env: GridWorld) -> Optional[int]:
        """Return the theoretical minimum steps, or None if not analytically known."""
        return None

    def evaluate(self, env: GridWorld, steps_taken: int, success: bool) -> TaskResult:
        """Build a TaskResult from episode outcomes."""
        opt = self.optimal_steps(env)
        if opt is not None and steps_taken > 0:
            efficiency = min(1.0, opt / steps_taken)
        else:
            efficiency = 1.0 if success else 0.0

        return TaskResult(
            task_name=self.name,
            success=success,
            steps_taken=steps_taken,
            optimal_steps=opt,
            efficiency=efficiency,
        )

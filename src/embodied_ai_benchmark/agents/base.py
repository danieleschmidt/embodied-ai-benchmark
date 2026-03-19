"""Base class for benchmark agents."""

from __future__ import annotations
from abc import ABC, abstractmethod

from ..envs.gridworld import GridWorld, GridState, Action


class Agent(ABC):
    """
    Stateful agent interface.

    act() is called once per step with the current state.
    reset() is called at the start of each episode.
    """

    name: str = "abstract_agent"

    def reset(self) -> None:
        """Reset any agent-internal state for a new episode."""
        pass

    @abstractmethod
    def act(self, env: GridWorld, state: GridState) -> Action:
        """Select and return an action given the current state."""
        ...

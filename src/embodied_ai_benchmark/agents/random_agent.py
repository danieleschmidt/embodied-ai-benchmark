"""RandomAgent: selects uniformly random actions."""

from __future__ import annotations
import random
from typing import Optional

from ..envs.gridworld import GridWorld, GridState, Action
from .base import Agent


class RandomAgent(Agent):
    """
    Baseline agent that picks uniformly at random from all actions.
    Includes PICK_UP in the action set so it can accidentally succeed.
    """

    name = "random"

    def __init__(self, seed: Optional[int] = None):
        self._rng = random.Random(seed)
        self._actions = list(Action)

    def reset(self) -> None:
        pass  # stateless

    def act(self, env: GridWorld, state: GridState) -> Action:
        return self._rng.choice(self._actions)

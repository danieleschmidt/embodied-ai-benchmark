"""NavigationTask: reach target cell with minimum steps."""

from __future__ import annotations
from typing import Optional, List, Tuple

from ..envs.gridworld import GridWorld, GridState, GridObject, Action
from .base import EmbodiedTask


class NavigationTask(EmbodiedTask):
    """
    Agent must navigate from start to a target cell.
    Optimal solution is BFS shortest path.
    """

    name = "navigation"

    def __init__(
        self,
        rows: int = 8,
        cols: int = 8,
        start: Tuple[int, int] = (0, 0),
        target: Tuple[int, int] = (7, 7),
        obstacles: Optional[List[Tuple[int, int]]] = None,
        max_steps: int = 100,
    ):
        self.rows = rows
        self.cols = cols
        self.start = start
        self.target = target
        self.obstacles = obstacles or []
        self.max_steps = max_steps

    def build_env(self) -> GridWorld:
        env = GridWorld(
            rows=self.rows,
            cols=self.cols,
            obstacles=self.obstacles,
            start=self.start,
            target=self.target,
            max_steps=self.max_steps,
        )
        return env

    def is_success(self, env: GridWorld, state: GridState) -> bool:
        return state.agent_row == self.target[0] and state.agent_col == self.target[1]

    def optimal_steps(self, env: GridWorld) -> Optional[int]:
        return env.bfs_distance(self.start, self.target)

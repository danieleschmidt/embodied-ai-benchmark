"""RoomExplorationTask: explore N% of the grid in minimum steps."""

from __future__ import annotations
from typing import Optional, List, Tuple

from ..envs.gridworld import GridWorld, GridState, Action
from .base import EmbodiedTask


class RoomExplorationTask(EmbodiedTask):
    """
    Agent must visit at least `coverage_pct` percent of all non-obstacle cells.
    Optimal bound: BFS spanning-tree traversal (lower bound = #cells_to_visit - 1).
    """

    name = "room_exploration"

    def __init__(
        self,
        rows: int = 8,
        cols: int = 8,
        start: Tuple[int, int] = (0, 0),
        obstacles: Optional[List[Tuple[int, int]]] = None,
        coverage_pct: float = 0.8,
        max_steps: int = 200,
    ):
        self.rows = rows
        self.cols = cols
        self.start = start
        self.obstacles = set(obstacles or [])
        self.coverage_pct = coverage_pct
        self.max_steps = max_steps

    def build_env(self) -> GridWorld:
        return GridWorld(
            rows=self.rows,
            cols=self.cols,
            obstacles=list(self.obstacles),
            start=self.start,
            max_steps=self.max_steps,
        )

    def _free_cells(self) -> int:
        return self.rows * self.cols - len(self.obstacles)

    def _required_cells(self) -> int:
        return max(1, int(self._free_cells() * self.coverage_pct))

    def is_success(self, env: GridWorld, state: GridState) -> bool:
        explored = len(state.explored)
        return explored >= self._required_cells()

    def optimal_steps(self, env: GridWorld) -> Optional[int]:
        """
        Lower bound: (cells_to_visit - 1) moves. This is achievable via an
        Euler-tour-like traversal of the spanning tree of the grid graph.
        """
        return max(0, self._required_cells() - 1)

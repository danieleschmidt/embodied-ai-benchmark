"""BFSAgent: optimal navigation and pickup agent using BFS planning."""

from __future__ import annotations
from typing import Optional, List, Tuple, Deque
from collections import deque

from ..envs.gridworld import GridWorld, GridState, Action, GridObject
from .base import Agent


def bfs_path(env: GridWorld, start: Tuple[int, int], goal: Tuple[int, int]) -> List[Tuple[int, int]]:
    """Return list of cells from start to goal (inclusive), or [] if unreachable."""
    if start == goal:
        return [start]
    visited = {start}
    parent = {start: None}
    queue: Deque = deque([start])
    while queue:
        cur = queue.popleft()
        for nb in env.passable_neighbors(*cur):
            if nb == goal:
                # Reconstruct path
                path = [goal]
                node = cur
                while node is not None:
                    path.append(node)
                    node = parent.get(node)
                path.reverse()
                return path
            if nb not in visited:
                visited.add(nb)
                parent[nb] = cur
                queue.append(nb)
    return []


def pos_to_move_action(fr: int, fc: int, tr: int, tc: int) -> Action:
    dr, dc = tr - fr, tc - fc
    if dr == -1:
        return Action.MOVE_NORTH
    if dr == 1:
        return Action.MOVE_SOUTH
    if dc == 1:
        return Action.MOVE_EAST
    if dc == -1:
        return Action.MOVE_WEST
    raise ValueError(f"Non-adjacent cells ({fr},{fc}) → ({tr},{tc})")


class BFSAgent(Agent):
    """
    Optimal agent that uses BFS to plan paths.

    Behavior depends on task type detected from env state:
      - If env has a target: navigate to it.
      - If env has objects and pickup_order is set: follow pickup order.
      - Otherwise: explore (frontier-based BFS coverage).
    """

    name = "bfs"

    def __init__(self, pickup_order: Optional[List[Tuple[str, str]]] = None):
        """
        Args:
            pickup_order: list of (color, name) in order to pick up, or None for auto-detect.
        """
        self._pickup_order = pickup_order
        self._plan: List[Action] = []
        self._current_goal: Optional[Tuple[int, int]] = None

    def reset(self) -> None:
        self._plan = []
        self._current_goal = None
        self._pickup_idx = 0
        self._phase = "init"

    def act(self, env: GridWorld, state: GridState) -> Action:
        # If we have a queued plan, execute it
        if self._plan:
            return self._plan.pop(0)

        pos = (state.agent_row, state.agent_col)

        # 1) Navigation task: move to target
        if env.target is not None and pos != env.target:
            path = bfs_path(env, pos, env.target)
            if len(path) > 1:
                self._plan = [pos_to_move_action(path[i][0], path[i][1],
                                                  path[i+1][0], path[i+1][1])
                              for i in range(len(path) - 1)]
                return self._plan.pop(0)

        # 2) Object pickup task
        if self._pickup_order:
            if not hasattr(self, "_pickup_idx"):
                self._pickup_idx = 0

            # Skip already-picked objects
            while self._pickup_idx < len(self._pickup_order):
                color, name = self._pickup_order[self._pickup_idx]
                if (color, name) in [(o.color, o.name) for o in env.inventory]:
                    self._pickup_idx += 1
                else:
                    break

            if self._pickup_idx < len(self._pickup_order):
                color, name = self._pickup_order[self._pickup_idx]
                obj = next(
                    (o for o in env.objects if o.color == color and o.name == name and not o.picked_up),
                    None,
                )
                if obj:
                    if pos == (obj.row, obj.col):
                        self._pickup_idx += 1
                        return Action.PICK_UP
                    path = bfs_path(env, pos, (obj.row, obj.col))
                    if len(path) > 1:
                        self._plan = [pos_to_move_action(path[i][0], path[i][1],
                                                          path[i+1][0], path[i+1][1])
                                      for i in range(len(path) - 1)]
                        return self._plan.pop(0)

        # 3) Exploration: BFS to nearest unvisited cell
        frontier = self._nearest_unvisited(env, state)
        if frontier:
            path = bfs_path(env, pos, frontier)
            if len(path) > 1:
                self._plan = [pos_to_move_action(path[i][0], path[i][1],
                                                  path[i+1][0], path[i+1][1])
                              for i in range(len(path) - 1)]
                return self._plan.pop(0)

        return Action.NOOP

    def _nearest_unvisited(
        self, env: GridWorld, state: GridState
    ) -> Optional[Tuple[int, int]]:
        """BFS from agent to find the nearest unvisited passable cell."""
        start = (state.agent_row, state.agent_col)
        visited_q = {start}
        queue: Deque = deque([(start, 0)])
        while queue:
            (r, c), dist = queue.popleft()
            if (r, c) not in state.explored:
                return (r, c)
            for nr, nc in env.passable_neighbors(r, c):
                if (nr, nc) not in visited_q:
                    visited_q.add((nr, nc))
                    queue.append(((nr, nc), dist + 1))
        return None

"""
GridWorld: 2D grid environment for embodied AI benchmarking.

Grid cells:
  ' ' = free space
  '#' = obstacle
  '@' = agent
  'o' = object (with color/name metadata)
  'T' = target cell
"""

from __future__ import annotations
import numpy as np
from typing import Optional, Dict, List, Tuple, Any
from dataclasses import dataclass, field
from enum import IntEnum
import copy


class Action(IntEnum):
    MOVE_NORTH = 0
    MOVE_SOUTH = 1
    MOVE_EAST = 2
    MOVE_WEST = 3
    TURN_LEFT = 4
    TURN_RIGHT = 5
    PICK_UP = 6
    NOOP = 7


# Facing directions: 0=N, 1=E, 2=S, 3=W
FACING_NAMES = ["north", "east", "south", "west"]
FACING_DELTAS = {0: (-1, 0), 1: (0, 1), 2: (1, 0), 3: (0, -1)}

# Action → direction delta (for MOVE_*)
ACTION_DELTAS = {
    Action.MOVE_NORTH: (-1, 0),
    Action.MOVE_SOUTH: (1, 0),
    Action.MOVE_EAST: (0, 1),
    Action.MOVE_WEST: (0, -1),
}


@dataclass
class GridObject:
    name: str
    color: str
    row: int
    col: int
    picked_up: bool = False

    def __repr__(self):
        return f"{self.color} {self.name} @ ({self.row},{self.col})"


@dataclass
class GridState:
    """Snapshot of the entire grid state."""
    rows: int
    cols: int
    obstacles: set  # set of (r, c)
    objects: List[GridObject]
    agent_row: int
    agent_col: int
    agent_facing: int  # 0=N,1=E,2=S,3=W
    target: Optional[Tuple[int, int]]
    step_count: int
    explored: set  # set of (r, c) visited by agent


class GridWorld:
    """
    Deterministic 2D grid world environment.

    Args:
        rows: grid height
        cols: grid width
        obstacles: list of (r, c) obstacle positions
        objects: list of GridObject instances
        start: (r, c) agent start position (default top-left)
        target: (r, c) target cell (optional)
        max_steps: episode step limit
        seed: random seed for procedural generation
    """

    def __init__(
        self,
        rows: int = 10,
        cols: int = 10,
        obstacles: Optional[List[Tuple[int, int]]] = None,
        objects: Optional[List[GridObject]] = None,
        start: Tuple[int, int] = (0, 0),
        target: Optional[Tuple[int, int]] = None,
        max_steps: int = 200,
    ):
        self.rows = rows
        self.cols = cols
        self._initial_obstacles = set(obstacles or [])
        self._initial_objects = copy.deepcopy(objects or [])
        self._start = start
        self._target = target
        self.max_steps = max_steps
        self.reset()

    # ------------------------------------------------------------------ #
    #  Public API
    # ------------------------------------------------------------------ #

    def reset(self) -> GridState:
        self._obstacles = set(self._initial_obstacles)
        self._objects: List[GridObject] = copy.deepcopy(self._initial_objects)
        self._agent_row, self._agent_col = self._start
        self._agent_facing: int = 0  # facing North
        self._step_count: int = 0
        self._explored: set = {self._start}
        self._inventory: List[GridObject] = []
        return self._get_state()

    def step(self, action: Action) -> Tuple[GridState, float, bool, Dict[str, Any]]:
        """
        Apply action. Returns (state, reward, done, info).
        reward = 0 by default (tasks define their own reward shaping).
        """
        if self._step_count >= self.max_steps:
            return self._get_state(), 0.0, True, {"reason": "max_steps"}

        info: Dict[str, Any] = {}
        reward = 0.0
        done = False

        if action in ACTION_DELTAS:
            dr, dc = ACTION_DELTAS[action]
            nr, nc = self._agent_row + dr, self._agent_col + dc
            if self._is_passable(nr, nc):
                self._agent_row, self._agent_col = nr, nc
                self._explored.add((nr, nc))
                # Update facing to match movement direction
                for f, (fdr, fdc) in FACING_DELTAS.items():
                    if (fdr, fdc) == (dr, dc):
                        self._agent_facing = f
                        break
            else:
                info["collision"] = True

        elif action == Action.TURN_LEFT:
            self._agent_facing = (self._agent_facing - 1) % 4

        elif action == Action.TURN_RIGHT:
            self._agent_facing = (self._agent_facing + 1) % 4

        elif action == Action.PICK_UP:
            obj = self._object_at(self._agent_row, self._agent_col)
            if obj and not obj.picked_up:
                obj.picked_up = True
                self._inventory.append(obj)
                info["picked_up"] = repr(obj)
            else:
                info["pick_failed"] = True

        elif action == Action.NOOP:
            pass

        self._step_count += 1
        return self._get_state(), reward, done, info

    def render(self, show_inventory: bool = True) -> str:
        """Return ASCII representation of current grid state."""
        grid = [["." for _ in range(self.cols)] for _ in range(self.rows)]

        # Obstacles
        for (r, c) in self._obstacles:
            if 0 <= r < self.rows and 0 <= c < self.cols:
                grid[r][c] = "#"

        # Target
        if self._target:
            tr, tc = self._target
            grid[tr][tc] = "T"

        # Objects
        for obj in self._objects:
            if not obj.picked_up:
                grid[obj.row][obj.col] = obj.color[0].upper()

        # Explored marker (subtle)
        for (r, c) in self._explored:
            if grid[r][c] == ".":
                grid[r][c] = "·"

        # Agent
        facing_chars = ["^", ">", "v", "<"]
        grid[self._agent_row][self._agent_col] = facing_chars[self._agent_facing]

        lines = []
        border = "+" + "-" * self.cols + "+"
        lines.append(border)
        for row in grid:
            lines.append("|" + "".join(row) + "|")
        lines.append(border)

        lines.append(
            f"Step {self._step_count}/{self.max_steps}  "
            f"Agent:({self._agent_row},{self._agent_col})  "
            f"Facing:{FACING_NAMES[self._agent_facing]}"
        )
        if show_inventory and self._inventory:
            lines.append(f"Inventory: {[repr(o) for o in self._inventory]}")

        return "\n".join(lines)

    # ------------------------------------------------------------------ #
    #  Properties
    # ------------------------------------------------------------------ #

    @property
    def agent_pos(self) -> Tuple[int, int]:
        return (self._agent_row, self._agent_col)

    @property
    def agent_facing(self) -> int:
        return self._agent_facing

    @property
    def target(self) -> Optional[Tuple[int, int]]:
        return self._target

    @property
    def objects(self) -> List[GridObject]:
        return self._objects

    @property
    def inventory(self) -> List[GridObject]:
        return self._inventory

    @property
    def step_count(self) -> int:
        return self._step_count

    @property
    def explored(self) -> set:
        return set(self._explored)

    @property
    def total_cells(self) -> int:
        return self.rows * self.cols - len(self._obstacles)

    # ------------------------------------------------------------------ #
    #  Internal helpers
    # ------------------------------------------------------------------ #

    def _is_passable(self, r: int, c: int) -> bool:
        return 0 <= r < self.rows and 0 <= c < self.cols and (r, c) not in self._obstacles

    def _object_at(self, r: int, c: int) -> Optional[GridObject]:
        for obj in self._objects:
            if not obj.picked_up and obj.row == r and obj.col == c:
                return obj
        return None

    def _get_state(self) -> GridState:
        return GridState(
            rows=self.rows,
            cols=self.cols,
            obstacles=set(self._obstacles),
            objects=copy.deepcopy(self._objects),
            agent_row=self._agent_row,
            agent_col=self._agent_col,
            agent_facing=self._agent_facing,
            target=self._target,
            step_count=self._step_count,
            explored=set(self._explored),
        )

    def passable_neighbors(self, r: int, c: int) -> List[Tuple[int, int]]:
        """BFS helper: return passable neighbors of (r, c)."""
        result = []
        for dr, dc in [(-1, 0), (1, 0), (0, 1), (0, -1)]:
            nr, nc = r + dr, c + dc
            if self._is_passable(nr, nc):
                result.append((nr, nc))
        return result

    def bfs_distance(self, start: Tuple[int, int], goal: Tuple[int, int]) -> Optional[int]:
        """Return BFS shortest-path distance, or None if unreachable."""
        if start == goal:
            return 0
        visited = {start}
        queue = [(start, 0)]
        while queue:
            (r, c), dist = queue.pop(0)
            for nr, nc in self.passable_neighbors(r, c):
                if (nr, nc) == goal:
                    return dist + 1
                if (nr, nc) not in visited:
                    visited.add((nr, nc))
                    queue.append(((nr, nc), dist + 1))
        return None

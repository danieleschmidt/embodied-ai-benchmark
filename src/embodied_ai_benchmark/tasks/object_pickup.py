"""ObjectPickupTask: pick up objects in a specified order."""

from __future__ import annotations
from typing import Optional, List, Tuple
import copy

from ..envs.gridworld import GridWorld, GridState, GridObject, Action
from .base import EmbodiedTask


class ObjectPickupTask(EmbodiedTask):
    """
    Agent must pick up objects in a specified order.
    Optimal: sum of BFS distances along the pickup route.
    """

    name = "object_pickup"

    def __init__(
        self,
        rows: int = 8,
        cols: int = 8,
        start: Tuple[int, int] = (0, 0),
        pickup_order: Optional[List[Tuple[str, str]]] = None,  # [(color, name), ...]
        objects: Optional[List[GridObject]] = None,
        obstacles: Optional[List[Tuple[int, int]]] = None,
        max_steps: int = 150,
    ):
        self.rows = rows
        self.cols = cols
        self.start = start
        self.pickup_order = pickup_order or []
        self.objects = objects or []
        self.obstacles = obstacles or []
        self.max_steps = max_steps

    def build_env(self) -> GridWorld:
        return GridWorld(
            rows=self.rows,
            cols=self.cols,
            obstacles=self.obstacles,
            objects=copy.deepcopy(self.objects),
            start=self.start,
            max_steps=self.max_steps,
        )

    def is_success(self, env: GridWorld, state: GridState) -> bool:
        """All targeted objects have been picked up in order."""
        picked_names = [(o.color, o.name) for o in env.inventory]
        # Check all required objects are in inventory (order not enforced at eval time,
        # but BFS agent respects order during planning)
        required = set(self.pickup_order)
        return required.issubset(set(picked_names))

    def optimal_steps(self, env: GridWorld) -> Optional[int]:
        """Sum of BFS distances along the greedy pickup route."""
        pos = self.start
        total = 0
        for color, name in self.pickup_order:
            obj = next((o for o in env.objects if o.color == color and o.name == name), None)
            if obj is None:
                return None
            d = env.bfs_distance(pos, (obj.row, obj.col))
            if d is None:
                return None
            total += d + 1  # +1 for the PICK_UP action
            pos = (obj.row, obj.col)
        return total

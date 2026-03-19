"""
InstructionAgent: executes parsed instruction sequences from InstructionFollowingTask.

Handles "go forward N" by resolving the agent's current facing direction,
"goto R C" via BFS, and "pick_up color name" by navigating then picking up.
"""

from __future__ import annotations
from typing import List, Optional, Tuple
from collections import deque

from ..envs.gridworld import GridWorld, GridState, Action, FACING_DELTAS
from .base import Agent
from .bfs_agent import bfs_path, pos_to_move_action


# Map facing index → forward MOVE action
FACING_TO_MOVE = {
    0: Action.MOVE_NORTH,
    1: Action.MOVE_EAST,
    2: Action.MOVE_SOUTH,
    3: Action.MOVE_WEST,
}


class InstructionAgent(Agent):
    """
    Follows pre-parsed instructions from InstructionFollowingTask.parsed_instructions.
    """

    name = "instruction"

    def __init__(self, instructions: list):
        """
        Args:
            instructions: output of parse_instructions() — a list of Action or
                          tuple ("pick_up", color, name) / ("goto", r, c).
        """
        self._instructions = list(instructions)
        self._plan: List[Action] = []
        self._instr_idx: int = 0
        self._completed: bool = False

    def reset(self) -> None:
        self._plan = []
        self._instr_idx = 0
        self._completed = False

    def act(self, env: GridWorld, state: GridState) -> Action:
        # Execute buffered plan first
        if self._plan:
            return self._plan.pop(0)

        # All instructions done
        if self._instr_idx >= len(self._instructions):
            if not self._completed:
                self._completed = True
                env._instructions_complete = True
            return Action.NOOP

        instr = self._instructions[self._instr_idx]
        self._instr_idx += 1

        # Simple action
        if isinstance(instr, Action):
            # "go forward" was stored as MOVE_NORTH placeholder — resolve to facing direction
            if instr == Action.MOVE_NORTH:
                return FACING_TO_MOVE[state.agent_facing]
            return instr

        # Compound instruction tuple
        if isinstance(instr, tuple):
            kind = instr[0]

            if kind == "pick_up":
                _, color, name = instr
                obj = next(
                    (o for o in env.objects if o.color == color and o.name == name and not o.picked_up),
                    None,
                )
                if obj is None:
                    return Action.NOOP  # object already picked up or not found

                pos = (state.agent_row, state.agent_col)
                if pos == (obj.row, obj.col):
                    return Action.PICK_UP

                path = bfs_path(env, pos, (obj.row, obj.col))
                if len(path) > 1:
                    self._plan = [
                        pos_to_move_action(path[i][0], path[i][1], path[i+1][0], path[i+1][1])
                        for i in range(len(path) - 1)
                    ]
                    self._plan.append(Action.PICK_UP)
                    # Re-process instruction next time (don't advance idx) isn't needed —
                    # we've already fully expanded this instruction into the plan.
                    if self._plan:
                        return self._plan.pop(0)

            elif kind == "goto":
                _, tr, tc = instr
                pos = (state.agent_row, state.agent_col)
                goal = (tr, tc)
                if pos == goal:
                    return Action.NOOP
                path = bfs_path(env, pos, goal)
                if len(path) > 1:
                    self._plan = [
                        pos_to_move_action(path[i][0], path[i][1], path[i+1][0], path[i+1][1])
                        for i in range(len(path) - 1)
                    ]
                    if self._plan:
                        return self._plan.pop(0)

        return Action.NOOP

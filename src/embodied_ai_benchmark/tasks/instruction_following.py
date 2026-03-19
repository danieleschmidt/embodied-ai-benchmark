"""
InstructionFollowingTask: follow a sequence of natural language instructions.

Supported instructions:
  - "turn left"
  - "turn right"
  - "go forward <N>"
  - "go north/south/east/west [<N>]"
  - "pick up <color> <object>"
  - "go to <row> <col>"

Instructions are parsed at task init time into action sequences.
Success = all instructions executed without error.
"""

from __future__ import annotations
import re
from typing import Optional, List, Tuple
import copy

from ..envs.gridworld import GridWorld, GridState, GridObject, Action, FACING_DELTAS, FACING_NAMES
from .base import EmbodiedTask


DIRECTION_MAP = {"north": 0, "east": 1, "south": 2, "west": 3}
MOVE_ACTIONS = {0: Action.MOVE_NORTH, 1: Action.MOVE_EAST, 2: Action.MOVE_SOUTH, 3: Action.MOVE_WEST}


def parse_instructions(instructions: List[str]) -> List[Action]:
    """
    Parse natural language instructions into a flat action sequence.
    Raises ValueError for unrecognized instructions.
    """
    actions = []
    for instr in instructions:
        instr = instr.strip().lower()

        if instr == "turn left":
            actions.append(Action.TURN_LEFT)

        elif instr == "turn right":
            actions.append(Action.TURN_RIGHT)

        elif m := re.fullmatch(r"go forward(?: (\d+))?", instr):
            n = int(m.group(1)) if m.group(1) else 1
            # MOVE_NORTH is the default "forward" direction in a straight-line sense.
            # At runtime, the agent's facing determines actual forward direction.
            # We encode as MOVE_NORTH here; InstructionAgent interprets dynamically.
            actions.extend([Action.MOVE_NORTH] * n)  # placeholder — agent resolves facing

        elif m := re.fullmatch(r"go (north|south|east|west)(?: (\d+))?", instr):
            direction = DIRECTION_MAP[m.group(1)]
            n = int(m.group(2)) if m.group(2) else 1
            move_action = MOVE_ACTIONS[direction]
            actions.extend([move_action] * n)

        elif m := re.fullmatch(r"pick up (\w+) (\w+)", instr):
            # Encoded as a special placeholder tuple — resolved at runtime by InstructionAgent
            actions.append(("pick_up", m.group(1), m.group(2)))  # type: ignore

        elif m := re.fullmatch(r"go to (\d+) (\d+)", instr):
            actions.append(("goto", int(m.group(1)), int(m.group(2))))  # type: ignore

        else:
            raise ValueError(f"Unrecognized instruction: '{instr}'")

    return actions


class InstructionFollowingTask(EmbodiedTask):
    """
    Agent follows a scripted instruction sequence.
    Optimal = the minimum number of primitive actions to execute all instructions.
    """

    name = "instruction_following"

    def __init__(
        self,
        rows: int = 8,
        cols: int = 8,
        start: Tuple[int, int] = (0, 0),
        start_facing: int = 0,
        instructions: Optional[List[str]] = None,
        objects: Optional[List[GridObject]] = None,
        obstacles: Optional[List[Tuple[int, int]]] = None,
        max_steps: int = 150,
    ):
        self.rows = rows
        self.cols = cols
        self.start = start
        self.start_facing = start_facing
        self.instructions = instructions or []
        self.objects = objects or []
        self.obstacles = obstacles or []
        self.max_steps = max_steps

        # Pre-parse to catch syntax errors early
        self._parsed = parse_instructions(self.instructions)

    def build_env(self) -> GridWorld:
        env = GridWorld(
            rows=self.rows,
            cols=self.cols,
            obstacles=self.obstacles,
            objects=copy.deepcopy(self.objects),
            start=self.start,
            max_steps=self.max_steps,
        )
        env._agent_facing = self.start_facing
        return env

    @property
    def parsed_instructions(self) -> list:
        return list(self._parsed)

    def is_success(self, env: GridWorld, state: GridState) -> bool:
        """Success is tracked externally by InstructionAgent completing all instructions."""
        # The agent sets a flag; we check via env attribute if set
        return getattr(env, "_instructions_complete", False)

    def optimal_steps(self, env: GridWorld) -> Optional[int]:
        """Count the minimum primitive actions to execute all instructions."""
        count = 0
        for item in self._parsed:
            if isinstance(item, tuple):
                if item[0] == "goto":
                    # BFS to target
                    pass  # can't know without agent position history; return None
                count += 1  # pick_up costs 1 action after moving
            else:
                count += 1
        return count

"""Embodied AI Benchmark — 2D grid world evaluation suite."""

from .envs.gridworld import GridWorld, GridObject, GridState, Action
from .tasks.base import EmbodiedTask, TaskResult
from .tasks.navigation import NavigationTask
from .tasks.object_pickup import ObjectPickupTask
from .tasks.exploration import RoomExplorationTask
from .tasks.instruction_following import InstructionFollowingTask
from .agents.base import Agent
from .agents.random_agent import RandomAgent
from .agents.bfs_agent import BFSAgent
from .agents.instruction_agent import InstructionAgent
from .evaluation.runner import BenchmarkRunner, AgentTaskMetrics, EpisodeResult

__version__ = "1.0.0"

__all__ = [
    "GridWorld", "GridObject", "GridState", "Action",
    "EmbodiedTask", "TaskResult",
    "NavigationTask", "ObjectPickupTask", "RoomExplorationTask", "InstructionFollowingTask",
    "Agent", "RandomAgent", "BFSAgent", "InstructionAgent",
    "BenchmarkRunner", "AgentTaskMetrics", "EpisodeResult",
]

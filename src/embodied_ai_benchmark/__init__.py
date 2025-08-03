"""
Embodied-AI Benchmark++

A comprehensive evaluation suite for embodied AI systems with multi-agent tasks,
LLM-guided curriculum learning, and cross-simulator compatibility.
"""

from .core.base_task import BaseTask
from .core.base_env import BaseEnv
from .core.base_agent import BaseAgent
from .core.base_metric import BaseMetric
from .evaluation.benchmark_suite import BenchmarkSuite
from .evaluation.evaluator import Evaluator
from .tasks.task_factory import make_env
from .multiagent.multi_agent_benchmark import MultiAgentBenchmark

__version__ = "1.0.0"
__all__ = [
    "BaseTask",
    "BaseEnv", 
    "BaseAgent",
    "BaseMetric",
    "BenchmarkSuite",
    "Evaluator",
    "make_env",
    "MultiAgentBenchmark",
]
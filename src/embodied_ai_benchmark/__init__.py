"""
Embodied-AI Benchmark++

A comprehensive evaluation suite for embodied AI systems with quantum-inspired task planning,
LLM-guided curriculum learning, advanced multi-agent coordination, and cross-simulator compatibility.

Features:
- Quantum-inspired task planning with superposition states
- LLM-guided adaptive curriculum learning
- Advanced multi-agent coordination protocols  
- Natural language task specification
- Realistic physics simulation with multiple engines
- Cross-platform deployment ready
- Global compliance (GDPR, CCPA, PDPA)
"""

# Core components
from .core.base_task import BaseTask
from .core.base_env import BaseEnv
from .core.base_agent import BaseAgent, RandomAgent, ScriptedAgent
from .core.base_metric import BaseMetric, SuccessMetric, EfficiencyMetric, SafetyMetric

# Evaluation and benchmarking
from .evaluation.benchmark_suite import BenchmarkSuite
from .evaluation.evaluator import Evaluator

# Task creation and management
from .tasks.task_factory import make_env, make_task, register_task, get_available_tasks

# Multi-agent systems
from .multiagent.multi_agent_benchmark import MultiAgentBenchmark
from .multiagent.coordination_protocols import (
    CommunicationProtocol, 
    DynamicRoleAssignment, 
    CoordinationOrchestrator,
    Message,
    MessageType,
    CoordinationTask
)

# Curriculum learning
from .curriculum.llm_curriculum import LLMCurriculum, CurriculumTrainer, PerformanceAnalysis

# Language interface
from .language.language_interface import LanguageTaskInterface, TaskParser, InstructionGenerator

# Physics simulation
from .physics.physics_config import PhysicsConfig, MaterialProperties, PhysicsEngine

# Utilities and optimization
from .utils.error_handling import ErrorHandler, ErrorRecoveryStrategy
from .utils.concurrent_execution import ConcurrentBenchmarkExecutor, AdvancedTaskManager, LoadBalancer
from .utils.benchmark_metrics import BenchmarkMetricsCollector
from .utils.caching import LRUCache, AdaptiveCache, PersistentCache, cache_result

__version__ = "1.0.0"
__all__ = [
    # Core
    "BaseTask",
    "BaseEnv", 
    "BaseAgent",
    "RandomAgent",
    "ScriptedAgent",
    "BaseMetric",
    "SuccessMetric", 
    "EfficiencyMetric",
    "SafetyMetric",
    
    # Evaluation
    "BenchmarkSuite",
    "Evaluator",
    
    # Tasks
    "make_env",
    "make_task",
    "register_task", 
    "get_available_tasks",
    
    # Multi-agent
    "MultiAgentBenchmark",
    "CommunicationProtocol",
    "DynamicRoleAssignment",
    "CoordinationOrchestrator",
    "Message",
    "MessageType", 
    "CoordinationTask",
    
    # Curriculum learning
    "LLMCurriculum",
    "CurriculumTrainer", 
    "PerformanceAnalysis",
    
    # Language interface
    "LanguageTaskInterface",
    "TaskParser",
    "InstructionGenerator",
    
    # Physics
    "PhysicsConfig",
    "MaterialProperties",
    "PhysicsEngine",
    
    # Utilities and optimization
    "ErrorHandler",
    "ErrorRecoveryStrategy",
    "ConcurrentBenchmarkExecutor",
    "AdvancedTaskManager",
    "LoadBalancer",
    "BenchmarkMetricsCollector",
    "LRUCache",
    "AdaptiveCache",
    "PersistentCache",
    "cache_result",
]
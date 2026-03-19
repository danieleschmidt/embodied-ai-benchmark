"""
BenchmarkRunner: evaluates agents on tasks, reports metrics.

Metrics per task:
  - success_rate: fraction of episodes where agent succeeded
  - mean_efficiency: mean(optimal/taken) over successful episodes
  - mean_steps: mean steps taken

Composite score = 0.6 * success_rate + 0.4 * mean_efficiency
"""

from __future__ import annotations
import time
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, field

from ..tasks.base import EmbodiedTask, TaskResult
from ..agents.base import Agent


@dataclass
class EpisodeResult:
    task_name: str
    agent_name: str
    success: bool
    steps_taken: int
    optimal_steps: Optional[int]
    efficiency: float
    wall_time_ms: float


@dataclass
class AgentTaskMetrics:
    task_name: str
    agent_name: str
    episodes: int
    success_rate: float
    mean_efficiency: float
    mean_steps: float
    composite_score: float
    results: List[EpisodeResult] = field(default_factory=list)

    def summary_line(self) -> str:
        return (
            f"  {self.agent_name:<20} | success={self.success_rate:.0%}  "
            f"efficiency={self.mean_efficiency:.2f}  "
            f"steps={self.mean_steps:.1f}  "
            f"composite={self.composite_score:.3f}"
        )


class BenchmarkRunner:
    """
    Evaluates one or more agents across one or more tasks.

    Usage:
        runner = BenchmarkRunner(episodes_per_task=5)
        runner.register_task(NavigationTask())
        runner.register_agent(BFSAgent())
        results = runner.run()
        runner.print_report(results)
    """

    def __init__(self, episodes_per_task: int = 1, verbose: bool = True):
        self.episodes_per_task = episodes_per_task
        self.verbose = verbose
        self._tasks: List[EmbodiedTask] = []
        self._agents: List[Agent] = []

    def register_task(self, task: EmbodiedTask) -> "BenchmarkRunner":
        self._tasks.append(task)
        return self

    def register_agent(self, agent: Agent) -> "BenchmarkRunner":
        self._agents.append(agent)
        return self

    def run(self) -> Dict[str, Dict[str, AgentTaskMetrics]]:
        """
        Returns nested dict: results[task_name][agent_name] = AgentTaskMetrics
        """
        all_results: Dict[str, Dict[str, AgentTaskMetrics]] = {}

        for task in self._tasks:
            all_results[task.name] = {}
            for agent in self._agents:
                metrics = self._run_agent_task(agent, task)
                all_results[task.name][agent.name] = metrics

        return all_results

    def _run_agent_task(self, agent: Agent, task: EmbodiedTask) -> AgentTaskMetrics:
        episodes: List[EpisodeResult] = []

        for ep in range(self.episodes_per_task):
            env = task.build_env()
            agent.reset()
            state = env.reset()
            done = False
            t0 = time.perf_counter()

            while not task.is_done(env, state):
                action = agent.act(env, state)
                state, _, env_done, info = env.step(action)
                if env_done:
                    break

            elapsed_ms = (time.perf_counter() - t0) * 1000
            success = task.is_success(env, state)
            opt = task.optimal_steps(env)

            if opt is not None and env.step_count > 0:
                efficiency = min(1.0, opt / env.step_count) if env.step_count > 0 else 1.0
            else:
                efficiency = 1.0 if success else 0.0

            episodes.append(EpisodeResult(
                task_name=task.name,
                agent_name=agent.name,
                success=success,
                steps_taken=env.step_count,
                optimal_steps=opt,
                efficiency=efficiency,
                wall_time_ms=elapsed_ms,
            ))

            if self.verbose:
                status = "✓" if success else "✗"
                print(f"  [{status}] {task.name}/{agent.name} ep{ep+1}: "
                      f"steps={env.step_count} opt={opt} eff={efficiency:.2f}")

        n = len(episodes)
        success_rate = sum(e.success for e in episodes) / n
        successful = [e for e in episodes if e.success]
        mean_eff = sum(e.efficiency for e in successful) / len(successful) if successful else 0.0
        mean_steps = sum(e.steps_taken for e in episodes) / n
        composite = 0.6 * success_rate + 0.4 * mean_eff

        return AgentTaskMetrics(
            task_name=task.name,
            agent_name=agent.name,
            episodes=n,
            success_rate=success_rate,
            mean_efficiency=mean_eff,
            mean_steps=mean_steps,
            composite_score=composite,
            results=episodes,
        )

    def print_report(self, results: Dict[str, Dict[str, AgentTaskMetrics]]) -> None:
        print("\n" + "=" * 70)
        print("  EMBODIED AI BENCHMARK RESULTS")
        print("=" * 70)

        all_composites: Dict[str, List[float]] = {}

        for task_name, agent_results in results.items():
            print(f"\n📋 Task: {task_name}")
            print("-" * 60)
            for agent_name, metrics in agent_results.items():
                print(metrics.summary_line())
                if agent_name not in all_composites:
                    all_composites[agent_name] = []
                all_composites[agent_name].append(metrics.composite_score)

        print("\n" + "=" * 70)
        print("  OVERALL COMPOSITE SCORES (mean across tasks)")
        print("=" * 70)
        sorted_agents = sorted(
            all_composites.items(), key=lambda x: -sum(x[1]) / len(x[1])
        )
        for rank, (agent_name, scores) in enumerate(sorted_agents, 1):
            mean_score = sum(scores) / len(scores)
            bar = "█" * int(mean_score * 20)
            print(f"  #{rank} {agent_name:<20} {mean_score:.3f}  {bar}")
        print()

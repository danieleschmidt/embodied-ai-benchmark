"""Integration test: BenchmarkRunner with multiple agents and tasks."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from embodied_ai_benchmark.evaluation.runner import BenchmarkRunner
from embodied_ai_benchmark.tasks.navigation import NavigationTask
from embodied_ai_benchmark.tasks.exploration import RoomExplorationTask
from embodied_ai_benchmark.agents.random_agent import RandomAgent
from embodied_ai_benchmark.agents.bfs_agent import BFSAgent


def test_runner_returns_results():
    runner = BenchmarkRunner(episodes_per_task=2, verbose=False)
    runner.register_task(NavigationTask(rows=5, cols=5, start=(0, 0), target=(4, 4)))
    runner.register_agent(RandomAgent(seed=42))
    runner.register_agent(BFSAgent())
    results = runner.run()

    assert "navigation" in results
    assert "random" in results["navigation"]
    assert "bfs" in results["navigation"]


def test_bfs_outperforms_random():
    runner = BenchmarkRunner(episodes_per_task=1, verbose=False)
    runner.register_task(NavigationTask(rows=6, cols=6, start=(0, 0), target=(5, 5), max_steps=200))
    runner.register_agent(RandomAgent(seed=0))
    runner.register_agent(BFSAgent())
    results = runner.run()

    bfs_score = results["navigation"]["bfs"].composite_score
    # BFS should get a perfect score (success=1.0, efficiency=1.0)
    assert bfs_score > 0.9


def test_composite_score_range():
    runner = BenchmarkRunner(episodes_per_task=1, verbose=False)
    runner.register_task(RoomExplorationTask(rows=4, cols=4, coverage_pct=0.5))
    runner.register_agent(BFSAgent())
    results = runner.run()

    metrics = results["room_exploration"]["bfs"]
    assert 0.0 <= metrics.composite_score <= 1.0
    assert 0.0 <= metrics.success_rate <= 1.0
    assert 0.0 <= metrics.mean_efficiency <= 1.0


def test_print_report_runs(capsys):
    runner = BenchmarkRunner(episodes_per_task=1, verbose=False)
    runner.register_task(NavigationTask(rows=4, cols=4, start=(0, 0), target=(3, 3)))
    runner.register_agent(BFSAgent())
    results = runner.run()
    runner.print_report(results)
    captured = capsys.readouterr()
    assert "navigation" in captured.out
    assert "bfs" in captured.out
    assert "COMPOSITE" in captured.out

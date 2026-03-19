#!/usr/bin/env python3
"""
Quick demo of the embodied-ai-benchmark suite.
Shows all 4 tasks, 3 agents, and the benchmark report.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from embodied_ai_benchmark import (
    GridWorld, GridObject, Action,
    NavigationTask, ObjectPickupTask, RoomExplorationTask, InstructionFollowingTask,
    RandomAgent, BFSAgent, InstructionAgent,
    BenchmarkRunner,
)
from embodied_ai_benchmark.tasks.instruction_following import parse_instructions


# ── GridWorld render demo ──────────────────────────────────────────────────────
print("=" * 50)
print("GridWorld Demo")
print("=" * 50)

env = GridWorld(
    rows=6, cols=6,
    obstacles=[(2, 1), (2, 2), (2, 3), (3, 3)],
    objects=[
        GridObject("ball", "red", 0, 5),
        GridObject("cube", "blue", 5, 0),
    ],
    start=(0, 0),
    target=(5, 5),
    max_steps=100,
)
env.reset()
print(env.render())

# ── BFS navigation demo ────────────────────────────────────────────────────────
print("\nBFS agent navigating (5,5):")
nav_task = NavigationTask(rows=6, cols=6, start=(0, 0), target=(5, 5),
                           obstacles=[(2, 1), (2, 2), (2, 3), (3, 3)])
env2 = nav_task.build_env()
agent = BFSAgent()
agent.reset()
state = env2.reset()
while not nav_task.is_done(env2, state):
    action = agent.act(env2, state)
    state, _, done, _ = env2.step(action)
    if done:
        break
print(env2.render())
print(f"Steps: {env2.step_count}, Optimal: {nav_task.optimal_steps(env2)}, "
      f"Success: {nav_task.is_success(env2, state)}")

# ── Instruction following demo ─────────────────────────────────────────────────
print("\n" + "=" * 50)
print("Instruction Following Demo")
print("=" * 50)

instrs = ["go east 3", "go south 2", "turn left", "turn right", "go east 1"]
print(f"Instructions: {instrs}")

if_task = InstructionFollowingTask(
    rows=6, cols=6, start=(0, 0), instructions=instrs
)
env3 = if_task.build_env()
agent3 = InstructionAgent(if_task.parsed_instructions)
agent3.reset()
state = env3.reset()
while not if_task.is_done(env3, state):
    action = agent3.act(env3, state)
    state, _, done, _ = env3.step(action)
    if done:
        break
print(env3.render())
print(f"Final pos: ({state.agent_row}, {state.agent_col}), steps: {env3.step_count}")

# ── Full benchmark ─────────────────────────────────────────────────────────────
print("\n" + "=" * 50)
print("Running Full Benchmark")
print("=" * 50)

objects = [
    GridObject("ball", "red", 2, 0),
    GridObject("cube", "blue", 4, 4),
]
pickup_order = [("red", "ball"), ("blue", "cube")]

# --- Navigation + Exploration + Pickup ---
runner1 = BenchmarkRunner(episodes_per_task=3, verbose=True)
runner1.register_task(NavigationTask(rows=6, cols=6, start=(0, 0), target=(5, 5),
                                     obstacles=[(2, 1), (2, 2), (2, 3)]))
runner1.register_task(ObjectPickupTask(rows=6, cols=6, start=(0, 0),
                                        pickup_order=pickup_order,
                                        objects=objects))
runner1.register_task(RoomExplorationTask(rows=6, cols=6, coverage_pct=0.7))
runner1.register_agent(RandomAgent(seed=42))
runner1.register_agent(BFSAgent(pickup_order=pickup_order))
results1 = runner1.run()
runner1.print_report(results1)

# --- Instruction following (InstructionAgent vs Random) ---
print("\n--- Instruction Following Task ---")
if_task2 = InstructionFollowingTask(
    rows=6, cols=6, start=(0, 0),
    instructions=["go south 2", "pick up red ball", "go south 2", "go east 4", "pick up blue cube"],
    objects=objects,
)

# Build InstructionAgent from this task's instructions
instr_agent = InstructionAgent(if_task2.parsed_instructions)
instr_agent.name = "instruction"

runner2 = BenchmarkRunner(episodes_per_task=3, verbose=True)
runner2.register_task(if_task2)
runner2.register_agent(RandomAgent(seed=42))
runner2.register_agent(instr_agent)
results2 = runner2.run()
runner2.print_report(results2)

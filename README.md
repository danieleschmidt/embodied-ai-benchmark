# Embodied AI Benchmark

A lightweight benchmark suite for evaluating embodied AI agents on core capabilities: **navigation**, **object manipulation**, **spatial exploration**, and **instruction following** — all in a reproducible 2D grid world. No simulators, no heavy dependencies, pure Python + numpy.

## Why

Most embodied AI benchmarks require simulators (Habitat, ManiSkill, iGibson) that are heavy, hard to set up, and environment-specific. This suite lets you:

- Develop and debug agent logic against clean, deterministic environments
- Measure efficiency vs. theoretical optimal (BFS lower bound)
- Compare baselines (random, BFS-optimal, instruction-following) on equal footing
- Run in CI in milliseconds

## Tasks

| Task | Goal | Optimal Measure |
|------|------|-----------------|
| `NavigationTask` | Reach a target cell | BFS shortest path |
| `ObjectPickupTask` | Pick up objects in specified order | Sum of BFS distances along pickup route |
| `RoomExplorationTask` | Visit ≥N% of all passable cells | Spanning-tree lower bound |
| `InstructionFollowingTask` | Execute natural language instruction sequence | Parsed action count |

### Instruction Language

The `InstructionFollowingTask` parses commands:

```
turn left / turn right
go forward [N]
go north/south/east/west [N]
pick up <color> <object>
go to <row> <col>
```

## Agents

| Agent | Description |
|-------|-------------|
| `RandomAgent` | Uniform random action baseline |
| `BFSAgent` | Optimal planner for navigation, pickup, exploration |
| `InstructionAgent` | Executes instruction sequences from `InstructionFollowingTask` |

## Metrics

Per task/agent:
- **Success rate** — fraction of episodes completed
- **Mean efficiency** — `optimal_steps / steps_taken`, clamped to [0, 1]
- **Mean steps** — raw step count
- **Composite score** — `0.6 × success_rate + 0.4 × mean_efficiency`

## Quickstart

```python
from embodied_ai_benchmark import (
    NavigationTask, RoomExplorationTask,
    BFSAgent, RandomAgent,
    BenchmarkRunner,
)

runner = BenchmarkRunner(episodes_per_task=5)
runner.register_task(NavigationTask(rows=8, cols=8, start=(0,0), target=(7,7)))
runner.register_task(RoomExplorationTask(rows=8, cols=8, coverage_pct=0.8))
runner.register_agent(RandomAgent(seed=42))
runner.register_agent(BFSAgent())

results = runner.run()
runner.print_report(results)
```

## GridWorld

The environment is a 2D grid with:
- **Free cells** (`.`) — passable
- **Obstacles** (`#`) — impassable
- **Agent** (`^>v<`) — facing direction shown
- **Target** (`T`) — destination marker
- **Objects** (`R`, `B`, etc.) — colored, pickable
- **Explored** (`·`) — visited cells

```
+----------+
|^.........| 
|..........| 
|.##.......| 
|..........| 
|.......T..| 
+----------+
Step 0/100  Agent:(0,0)  Facing:north
```

## Install

```bash
pip install -e .
```

Or just add `src/` to your Python path — no install required for development.

## Tests

```bash
python -m pytest tests/ -v
```

51 tests, all passing, ~0.1s.

## Structure

```
src/embodied_ai_benchmark/
  envs/
    gridworld.py          # GridWorld environment, GridObject, Action enum
  tasks/
    base.py               # EmbodiedTask ABC, TaskResult
    navigation.py         # NavigationTask
    object_pickup.py      # ObjectPickupTask
    exploration.py        # RoomExplorationTask
    instruction_following.py  # InstructionFollowingTask + parser
  agents/
    base.py               # Agent ABC
    random_agent.py       # RandomAgent
    bfs_agent.py          # BFSAgent
    instruction_agent.py  # InstructionAgent
  evaluation/
    runner.py             # BenchmarkRunner, metrics, report
demo.py                   # runnable demo
```

## Extending

### New task

```python
from embodied_ai_benchmark.tasks.base import EmbodiedTask

class MyTask(EmbodiedTask):
    name = "my_task"

    def build_env(self):
        return GridWorld(rows=8, cols=8, ...)

    def is_success(self, env, state):
        return ...  # your success condition

    def optimal_steps(self, env):
        return ...  # theoretical minimum, or None
```

### New agent

```python
from embodied_ai_benchmark.agents.base import Agent

class MyAgent(Agent):
    name = "my_agent"

    def act(self, env, state):
        return Action.MOVE_SOUTH  # your policy
```

## License

MIT

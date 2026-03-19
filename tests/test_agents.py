"""Tests for agent implementations."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from embodied_ai_benchmark.envs.gridworld import GridWorld, GridObject, Action
from embodied_ai_benchmark.agents.random_agent import RandomAgent
from embodied_ai_benchmark.agents.bfs_agent import BFSAgent, bfs_path
from embodied_ai_benchmark.agents.instruction_agent import InstructionAgent
from embodied_ai_benchmark.tasks.instruction_following import parse_instructions


class TestRandomAgent:
    def test_returns_action(self):
        agent = RandomAgent(seed=42)
        env = GridWorld(rows=5, cols=5, start=(0, 0))
        state = env.reset()
        action = agent.act(env, state)
        assert isinstance(action, Action)

    def test_seeded_reproducible(self):
        env = GridWorld(rows=5, cols=5, start=(0, 0))
        state = env.reset()
        a1 = RandomAgent(seed=1)
        a2 = RandomAgent(seed=1)
        actions1 = [a1.act(env, state) for _ in range(10)]
        actions2 = [a2.act(env, state) for _ in range(10)]
        assert actions1 == actions2


class TestBFSPath:
    def test_trivial(self):
        env = GridWorld(rows=5, cols=5, start=(0, 0))
        env.reset()
        path = bfs_path(env, (0, 0), (0, 0))
        assert path == [(0, 0)]

    def test_straight_path(self):
        env = GridWorld(rows=5, cols=5, start=(0, 0))
        env.reset()
        path = bfs_path(env, (0, 0), (0, 3))
        assert path[0] == (0, 0)
        assert path[-1] == (0, 3)
        assert len(path) == 4

    def test_around_obstacle(self):
        obs = [(0, 1), (1, 1), (2, 1)]
        env = GridWorld(rows=5, cols=5, obstacles=obs, start=(0, 0))
        env.reset()
        path = bfs_path(env, (0, 0), (0, 2))
        # Path must avoid obstacles — should go around
        assert path[0] == (0, 0)
        assert path[-1] == (0, 2)
        for pos in path:
            assert pos not in obs


class TestBFSAgent:
    def test_navigates_to_target(self):
        from embodied_ai_benchmark.tasks.navigation import NavigationTask
        task = NavigationTask(rows=5, cols=5, start=(0, 0), target=(4, 4), max_steps=50)
        env = task.build_env()
        agent = BFSAgent()
        agent.reset()
        state = env.reset()
        while not task.is_done(env, state):
            action = agent.act(env, state)
            state, _, done, _ = env.step(action)
            if done:
                break
        assert task.is_success(env, state)
        # BFS should be optimal
        assert env.step_count == task.optimal_steps(env)

    def test_pickup_task(self):
        from embodied_ai_benchmark.tasks.object_pickup import ObjectPickupTask
        objects = [GridObject("ball", "red", 2, 0)]
        task = ObjectPickupTask(
            rows=5, cols=5, start=(0, 0),
            pickup_order=[("red", "ball")],
            objects=objects,
            max_steps=20,
        )
        env = task.build_env()
        agent = BFSAgent(pickup_order=[("red", "ball")])
        agent.reset()
        state = env.reset()
        while not task.is_done(env, state):
            action = agent.act(env, state)
            state, _, done, _ = env.step(action)
            if done:
                break
        assert task.is_success(env, state)

    def test_explores_grid(self):
        from embodied_ai_benchmark.tasks.exploration import RoomExplorationTask
        task = RoomExplorationTask(rows=5, cols=5, coverage_pct=0.6, max_steps=100)
        env = task.build_env()
        agent = BFSAgent()
        agent.reset()
        state = env.reset()
        while not task.is_done(env, state):
            action = agent.act(env, state)
            state, _, done, _ = env.step(action)
            if done:
                break
        assert task.is_success(env, state)


class TestInstructionAgent:
    def test_turn_left(self):
        instructions = parse_instructions(["turn left"])
        agent = InstructionAgent(instructions)
        env = GridWorld(rows=5, cols=5, start=(0, 0))
        state = env.reset()
        action = agent.act(env, state)
        assert action == Action.TURN_LEFT

    def test_go_east_sequence(self):
        instructions = parse_instructions(["go east 3"])
        agent = InstructionAgent(instructions)
        env = GridWorld(rows=5, cols=5, start=(0, 0), max_steps=20)
        state = env.reset()
        agent.reset()
        for _ in range(3):
            action = agent.act(env, state)
            state, _, _, _ = env.step(action)
        assert state.agent_col == 3

    def test_pick_up_instruction(self):
        obj = GridObject("ball", "red", 0, 2)
        instructions = parse_instructions(["pick up red ball"])
        agent = InstructionAgent(instructions)
        env = GridWorld(rows=5, cols=5, objects=[obj], start=(0, 0), max_steps=20)
        state = env.reset()
        agent.reset()
        # Run until done or max steps
        for _ in range(15):
            action = agent.act(env, state)
            state, _, _, _ = env.step(action)
            if env.inventory:
                break
        assert len(env.inventory) == 1
        assert env.inventory[0].name == "ball"

    def test_go_forward_uses_facing(self):
        """go forward should move in the agent's current facing direction."""
        instructions = parse_instructions(["turn right", "go forward 2"])
        agent = InstructionAgent(instructions)
        env = GridWorld(rows=5, cols=5, start=(0, 0), max_steps=20)
        state = env.reset()
        agent.reset()
        # After turn right: facing East
        action1 = agent.act(env, state)
        assert action1 == Action.TURN_RIGHT
        state, _, _, _ = env.step(action1)
        # go forward 2 — should produce MOVE_EAST
        action2 = agent.act(env, state)
        assert action2 == Action.MOVE_EAST

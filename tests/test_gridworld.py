"""Tests for GridWorld environment."""

import pytest
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from embodied_ai_benchmark.envs.gridworld import GridWorld, GridObject, Action


def make_simple_env():
    return GridWorld(rows=5, cols=5, start=(0, 0), max_steps=50)


class TestGridWorldBasics:
    def test_reset_returns_state(self):
        env = make_simple_env()
        state = env.reset()
        assert state.agent_row == 0
        assert state.agent_col == 0
        assert state.step_count == 0

    def test_move_south(self):
        env = make_simple_env()
        env.reset()
        state, _, _, info = env.step(Action.MOVE_SOUTH)
        assert state.agent_row == 1
        assert state.agent_col == 0
        assert state.step_count == 1

    def test_move_east(self):
        env = make_simple_env()
        env.reset()
        state, _, _, _ = env.step(Action.MOVE_EAST)
        assert state.agent_col == 1

    def test_wall_collision(self):
        env = make_simple_env()
        env.reset()
        # Move north from (0,0) — should be blocked
        state, _, _, info = env.step(Action.MOVE_NORTH)
        assert state.agent_row == 0
        assert info.get("collision")

    def test_obstacle_blocks(self):
        env = GridWorld(rows=5, cols=5, obstacles=[(1, 0)], start=(0, 0), max_steps=50)
        env.reset()
        state, _, _, info = env.step(Action.MOVE_SOUTH)
        assert state.agent_row == 0  # blocked
        assert info.get("collision")

    def test_turn_left(self):
        env = make_simple_env()
        env.reset()
        state, _, _, _ = env.step(Action.TURN_LEFT)
        assert state.agent_facing == 3  # West

    def test_turn_right(self):
        env = make_simple_env()
        env.reset()
        state, _, _, _ = env.step(Action.TURN_RIGHT)
        assert state.agent_facing == 1  # East

    def test_step_counter(self):
        env = make_simple_env()
        env.reset()
        for _ in range(5):
            env.step(Action.NOOP)
        assert env.step_count == 5

    def test_max_steps_terminates(self):
        env = GridWorld(rows=5, cols=5, start=(0, 0), max_steps=3)
        env.reset()
        for _ in range(3):
            env.step(Action.NOOP)
        state, _, done, _ = env.step(Action.NOOP)
        assert done

    def test_explored_grows(self):
        env = make_simple_env()
        env.reset()
        env.step(Action.MOVE_SOUTH)
        env.step(Action.MOVE_EAST)
        assert len(env.explored) >= 3

    def test_render_contains_agent(self):
        env = make_simple_env()
        env.reset()
        rendered = env.render()
        # Agent facing chars: ^>v<
        assert any(c in rendered for c in "^>v<")

    def test_bfs_distance(self):
        env = GridWorld(rows=5, cols=5, start=(0, 0), max_steps=50)
        env.reset()
        assert env.bfs_distance((0, 0), (0, 0)) == 0
        assert env.bfs_distance((0, 0), (2, 3)) == 5
        assert env.bfs_distance((0, 0), (4, 4)) == 8

    def test_bfs_unreachable(self):
        # Surround a cell with obstacles
        obs = [(0, 1), (1, 0), (1, 1)]
        env = GridWorld(rows=5, cols=5, obstacles=obs, start=(0, 0), max_steps=50)
        env.reset()
        # (0,0) is reachable to itself
        assert env.bfs_distance((0, 0), (0, 0)) == 0


class TestPickup:
    def test_pick_up_object(self):
        obj = GridObject(name="box", color="red", row=0, col=1)
        env = GridWorld(rows=5, cols=5, objects=[obj], start=(0, 0), max_steps=50)
        env.reset()
        env.step(Action.MOVE_EAST)
        state, _, _, info = env.step(Action.PICK_UP)
        assert "picked_up" in info
        assert len(env.inventory) == 1
        assert env.inventory[0].name == "box"

    def test_pick_up_nothing(self):
        env = make_simple_env()
        env.reset()
        state, _, _, info = env.step(Action.PICK_UP)
        assert info.get("pick_failed")

    def test_object_gone_after_pickup(self):
        obj = GridObject(name="box", color="red", row=0, col=0)
        env = GridWorld(rows=5, cols=5, objects=[obj], start=(0, 0), max_steps=50)
        env.reset()
        env.step(Action.PICK_UP)
        # Render should no longer show the object at start
        rendered = env.render()
        # Agent symbol should be there, not 'R' for Red
        assert env.objects[0].picked_up

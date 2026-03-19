"""Tests for all four task types."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from embodied_ai_benchmark.envs.gridworld import GridObject
from embodied_ai_benchmark.tasks.navigation import NavigationTask
from embodied_ai_benchmark.tasks.object_pickup import ObjectPickupTask
from embodied_ai_benchmark.tasks.exploration import RoomExplorationTask
from embodied_ai_benchmark.tasks.instruction_following import InstructionFollowingTask


class TestNavigationTask:
    def test_build_env(self):
        task = NavigationTask(rows=5, cols=5, start=(0, 0), target=(4, 4))
        env = task.build_env()
        assert env.rows == 5
        assert env.target == (4, 4)

    def test_optimal_steps(self):
        task = NavigationTask(rows=5, cols=5, start=(0, 0), target=(3, 3))
        env = task.build_env()
        assert task.optimal_steps(env) == 6

    def test_success_at_target(self):
        task = NavigationTask(rows=5, cols=5, start=(0, 0), target=(1, 0))
        env = task.build_env()
        state = env.reset()
        from embodied_ai_benchmark.envs.gridworld import Action
        env.step(Action.MOVE_SOUTH)
        state = env._get_state()
        assert task.is_success(env, state)

    def test_not_success_not_at_target(self):
        task = NavigationTask(rows=5, cols=5, start=(0, 0), target=(4, 4))
        env = task.build_env()
        state = env._get_state()
        assert not task.is_success(env, state)

    def test_is_done_on_timeout(self):
        task = NavigationTask(rows=5, cols=5, start=(0, 0), target=(4, 4), max_steps=2)
        env = task.build_env()
        state = env.reset()
        from embodied_ai_benchmark.envs.gridworld import Action
        env.step(Action.NOOP)
        env.step(Action.NOOP)
        state = env._get_state()
        assert task.is_done(env, state)


class TestObjectPickupTask:
    def _make_task(self):
        objects = [
            GridObject(name="ball", color="red", row=1, col=1),
            GridObject(name="cube", color="blue", row=3, col=3),
        ]
        return ObjectPickupTask(
            rows=5, cols=5, start=(0, 0),
            pickup_order=[("red", "ball"), ("blue", "cube")],
            objects=objects,
        )

    def test_build_env(self):
        task = self._make_task()
        env = task.build_env()
        assert len(env.objects) == 2

    def test_optimal_steps(self):
        task = self._make_task()
        env = task.build_env()
        # start→(1,1): 2 steps + 1 pickup + (1,1)→(3,3): 4 steps + 1 pickup = 8
        assert task.optimal_steps(env) == 8

    def test_success_after_pickup(self):
        from embodied_ai_benchmark.envs.gridworld import Action
        task = self._make_task()
        env = task.build_env()
        state = env.reset()
        # Navigate to (1,1) and pick up
        env.step(Action.MOVE_SOUTH)
        env.step(Action.MOVE_EAST)
        env.step(Action.PICK_UP)
        # Navigate to (3,3) and pick up
        env.step(Action.MOVE_SOUTH)
        env.step(Action.MOVE_SOUTH)
        env.step(Action.MOVE_EAST)
        env.step(Action.MOVE_EAST)
        env.step(Action.PICK_UP)
        state = env._get_state()
        assert task.is_success(env, state)


class TestRoomExplorationTask:
    def test_build_env(self):
        task = RoomExplorationTask(rows=5, cols=5, coverage_pct=0.8)
        env = task.build_env()
        assert env.rows == 5

    def test_optimal_steps(self):
        task = RoomExplorationTask(rows=5, cols=5, coverage_pct=0.8)
        env = task.build_env()
        # 25 cells * 0.8 = 20, optimal = 19
        assert task.optimal_steps(env) == 19

    def test_not_success_at_start(self):
        task = RoomExplorationTask(rows=5, cols=5, coverage_pct=0.8)
        env = task.build_env()
        state = env._get_state()
        assert not task.is_success(env, state)  # only 1 cell explored

    def test_success_after_full_coverage(self):
        from embodied_ai_benchmark.envs.gridworld import Action
        task = RoomExplorationTask(rows=3, cols=3, coverage_pct=1.0, max_steps=50)
        env = task.build_env()
        env.reset()
        # Manually mark all cells as explored
        for r in range(3):
            for c in range(3):
                env._explored.add((r, c))
        state = env._get_state()
        assert task.is_success(env, state)


class TestInstructionFollowingTask:
    def test_parse_turn_left(self):
        task = InstructionFollowingTask(instructions=["turn left"])
        from embodied_ai_benchmark.envs.gridworld import Action
        assert task.parsed_instructions == [Action.TURN_LEFT]

    def test_parse_turn_right(self):
        task = InstructionFollowingTask(instructions=["turn right"])
        from embodied_ai_benchmark.envs.gridworld import Action
        assert task.parsed_instructions == [Action.TURN_RIGHT]

    def test_parse_go_north_3(self):
        task = InstructionFollowingTask(instructions=["go north 3"])
        from embodied_ai_benchmark.envs.gridworld import Action
        assert task.parsed_instructions == [Action.MOVE_NORTH] * 3

    def test_parse_go_east(self):
        task = InstructionFollowingTask(instructions=["go east"])
        from embodied_ai_benchmark.envs.gridworld import Action
        assert task.parsed_instructions == [Action.MOVE_EAST]

    def test_parse_pick_up(self):
        task = InstructionFollowingTask(instructions=["pick up red ball"])
        parsed = task.parsed_instructions
        assert len(parsed) == 1
        assert parsed[0] == ("pick_up", "red", "ball")

    def test_invalid_instruction_raises(self):
        import pytest
        with pytest.raises(ValueError, match="Unrecognized"):
            InstructionFollowingTask(instructions=["fly to moon"])

    def test_build_env(self):
        task = InstructionFollowingTask(
            rows=5, cols=5, start=(0, 0),
            instructions=["go east", "go south"]
        )
        env = task.build_env()
        assert env.rows == 5

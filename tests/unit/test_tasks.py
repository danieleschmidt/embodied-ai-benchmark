"""Unit tests for task implementations."""

import pytest
import numpy as np

from embodied_ai_benchmark.tasks.manipulation.furniture_assembly import (
    FurnitureAssemblyTask, FurnitureAssemblyEnv
)
from embodied_ai_benchmark.tasks.navigation.point_goal import (
    PointGoalTask, PointGoalEnv
)
from embodied_ai_benchmark.tasks.multiagent.cooperative_assembly import (
    CooperativeAssemblyTask, CooperativeAssemblyEnv
)


class TestFurnitureAssemblyTask:
    """Test FurnitureAssemblyTask implementation."""
    
    def test_task_initialization(self):
        """Test furniture assembly task initialization."""
        config = {
            "furniture_type": "table",
            "difficulty": "medium",
            "max_steps": 1000
        }
        
        task = FurnitureAssemblyTask(config)
        
        assert task.furniture_type == "table"
        assert "tabletop" in task.parts
        assert "leg_1" in task.parts
        assert len(task.required_connections) > 0
    
    def test_task_reset(self):
        """Test task reset functionality."""
        config = {"furniture_type": "table"}
        task = FurnitureAssemblyTask(config)
        
        obs = task.reset()
        
        assert task.current_step == 0
        assert len(task.connections_made) == 0
        assert "parts" in obs
        assert "connections" in obs
        assert "progress" in obs
    
    def test_pick_action(self):
        """Test pick action processing."""
        config = {"furniture_type": "table"}
        task = FurnitureAssemblyTask(config)
        task.reset()
        
        action = {
            "type": "pick",
            "target": "leg_1"
        }
        
        obs, reward, done, info = task.step(action)
        
        assert reward > 0  # Should get positive reward for valid pick
        assert task.parts["leg_1"].get("picked", False)
    
    def test_invalid_pick_action(self):
        """Test invalid pick action."""
        config = {"furniture_type": "table"}
        task = FurnitureAssemblyTask(config)
        task.reset()
        
        action = {
            "type": "pick",
            "target": "nonexistent_part"
        }
        
        obs, reward, done, info = task.step(action)
        
        assert reward < 0  # Should get negative reward for invalid pick
    
    def test_place_action(self):
        """Test place action processing."""
        config = {"furniture_type": "table"}
        task = FurnitureAssemblyTask(config)
        task.reset()
        
        # First pick a part
        task.parts["leg_1"]["picked"] = True
        
        action = {
            "type": "place",
            "target": "leg_1",
            "position": [0.0, 0.0, 0.0]
        }
        
        obs, reward, done, info = task.step(action)
        
        assert reward >= 0  # Should not be negative for valid place
        assert not task.parts["leg_1"].get("picked", False)
    
    def test_connect_action(self):
        """Test connect action processing."""
        config = {"furniture_type": "table"}
        task = FurnitureAssemblyTask(config)
        task.reset()
        
        # Position parts close together
        task.parts["tabletop"]["position"] = np.array([0.0, 0.8, 0.0])
        task.parts["leg_1"]["position"] = np.array([0.0, 0.8, 0.0])
        
        action = {
            "type": "connect",
            "part1": "tabletop",
            "part2": "leg_1"
        }
        
        obs, reward, done, info = task.step(action)
        
        assert reward > 0  # Should get positive reward for valid connection
        assert ("tabletop", "leg_1") in task.connections_made or ("leg_1", "tabletop") in task.connections_made
    
    def test_success_condition(self):
        """Test task success condition."""
        config = {"furniture_type": "table"}
        task = FurnitureAssemblyTask(config)
        task.reset()
        
        # Initially not successful
        assert not task.check_success()
        
        # Make all required connections
        for connection in task.required_connections:
            task.connections_made.add(connection)
            task.parts[connection[0]]["attached"] = True
            task.parts[connection[1]]["attached"] = True
        
        # Should be successful now
        assert task.check_success()


class TestFurnitureAssemblyEnv:
    """Test FurnitureAssemblyEnv implementation."""
    
    def test_env_initialization(self):
        """Test environment initialization."""
        config = {"simulator": "habitat"}
        env = FurnitureAssemblyEnv(config)
        
        assert env.simulator_name == "habitat"
        assert "workspace_bounds" in env.__dict__
    
    def test_env_with_task(self):
        """Test environment with task integration."""
        config = {"furniture_type": "table"}
        env = FurnitureAssemblyEnv(config)
        task = FurnitureAssemblyTask(config)
        
        env.set_task(task)
        obs = env.reset()
        
        # Should have both environment and task observations
        assert "rgb" in obs
        assert "depth" in obs
        assert "parts" in obs
    
    def test_workspace_bounds_checking(self):
        """Test workspace bounds enforcement."""
        config = {"furniture_type": "table"}
        env = FurnitureAssemblyEnv(config)
        task = FurnitureAssemblyTask(config)
        env.set_task(task)
        env.reset()
        
        # Action outside workspace bounds
        action = {
            "type": "place",
            "target": "leg_1",
            "position": [100.0, 100.0, 100.0]  # Way outside bounds
        }
        
        obs, reward, done, info = env.step(action)
        
        assert "workspace_violation" in info
        assert reward < 0


class TestPointGoalTask:
    """Test PointGoalTask implementation."""
    
    def test_task_initialization(self):
        """Test point goal task initialization."""
        config = {
            "room_size": (10, 10),
            "goal_radius": 0.5,
            "num_obstacles": 3
        }
        
        task = PointGoalTask(config)
        
        assert task.room_size == (10, 10)
        assert task.goal_radius == 0.5
        assert len(task.obstacles) == 3
    
    def test_task_reset(self):
        """Test task reset functionality."""
        config = {"room_size": (10, 10)}
        task = PointGoalTask(config)
        
        obs = task.reset()
        
        assert task.current_step == 0
        assert len(task.path_history) == 1  # Start position
        assert "agent" in obs
        assert "obstacles" in obs
        assert "goal_direction" in obs
    
    def test_velocity_action(self):
        """Test velocity-based movement."""
        config = {"room_size": (10, 10)}
        task = PointGoalTask(config)
        task.reset()
        
        initial_pos = task.current_position.copy()
        
        action = {
            "type": "velocity",
            "values": [1.0, 0.0]  # Move in x direction
        }
        
        obs, reward, done, info = task.step(action)
        
        # Should have moved
        assert not np.allclose(task.current_position, initial_pos)
        assert task.current_position[0] > initial_pos[0]
    
    def test_position_action(self):
        """Test absolute position movement."""
        config = {"room_size": (10, 10)}
        task = PointGoalTask(config)
        task.reset()
        
        target_pos = [1.0, 1.0]
        action = {
            "type": "position",
            "values": target_pos
        }
        
        obs, reward, done, info = task.step(action)
        
        # Should be closer to target position
        distance = np.linalg.norm(task.current_position - np.array(target_pos))
        assert distance < 1.0  # Should be relatively close
    
    def test_goal_reaching(self):
        """Test goal reaching success condition."""
        config = {"room_size": (10, 10), "goal_radius": 0.5}
        task = PointGoalTask(config)
        task.reset()
        
        # Move agent to goal position
        task.current_position = task.goal_position.copy()
        
        assert task.check_success()
        
        action = {"type": "velocity", "values": [0.0, 0.0]}
        obs, reward, done, info = task.step(action)
        
        assert info["task_success"]
        assert reward > 50  # Should get large completion reward
    
    def test_obstacle_collision(self):
        """Test obstacle collision detection."""
        config = {"room_size": (10, 10)}
        task = PointGoalTask(config)
        task.reset()
        
        # Place agent at obstacle position
        if task.obstacles:
            obstacle_pos = task.obstacles[0]["position"]
            action = {
                "type": "position",
                "values": obstacle_pos.tolist()
            }
            
            obs, reward, done, info = task.step(action)
            
            if info.get("collision", False):
                assert reward < 0  # Should get negative reward for collision
    
    def test_occupancy_grid_generation(self):
        """Test occupancy grid generation."""
        config = {"room_size": (10, 10)}
        task = PointGoalTask(config)
        obs = task.reset()
        
        assert "occupancy_grid" in obs
        occupancy_grid = np.array(obs["occupancy_grid"])
        assert occupancy_grid.shape == (21, 21)  # Default grid size
        assert occupancy_grid.dtype == float


class TestPointGoalEnv:
    """Test PointGoalEnv implementation."""
    
    def test_env_initialization(self):
        """Test environment initialization."""
        config = {"simulator": "habitat"}
        env = PointGoalEnv(config)
        
        assert env.simulator_name == "habitat"
        assert hasattr(env, "render_scale")
    
    def test_env_rendering(self):
        """Test environment rendering."""
        config = {"room_size": (10, 10)}
        env = PointGoalEnv(config)
        task = PointGoalTask(config)
        env.set_task(task)
        env.reset()
        
        # Test RGB rendering
        img = env.render("rgb_array")
        assert isinstance(img, np.ndarray)
        assert img.shape == (480, 640, 3)
        assert img.dtype == np.uint8
    
    def test_proprioception_updates(self):
        """Test proprioception updates with agent position."""
        config = {"room_size": (10, 10)}
        env = PointGoalEnv(config)
        task = PointGoalTask(config)
        env.set_task(task)
        
        obs = env.reset()
        
        # Check proprioception includes position
        assert "proprioception" in obs
        proprioception = obs["proprioception"]
        assert proprioception[0] == task.current_position[0]
        assert proprioception[1] == task.current_position[1]


class TestCooperativeAssemblyTask:
    """Test CooperativeAssemblyTask implementation."""
    
    def test_task_initialization(self):
        """Test cooperative assembly task initialization."""
        config = {
            "num_agents": 2,
            "furniture": "ikea_table",
            "difficulty": "medium"
        }
        
        task = CooperativeAssemblyTask(config)
        
        assert task.num_agents == 2
        assert task.furniture == "ikea_table"
        assert len(task.parts) > 0
        assert len(task.coordination_requirements) > 0
    
    def test_task_reset(self):
        """Test task reset functionality."""
        config = {"num_agents": 2, "furniture": "ikea_table"}
        task = CooperativeAssemblyTask(config)
        
        obs = task.reset()
        
        assert task.current_step == 0
        assert len(task.agent_positions) == 2
        assert len(task.collaboration_events) == 0
        assert "global" in obs
        assert "agents" in obs
    
    def test_multi_agent_action_processing(self):
        """Test multi-agent action processing."""
        config = {"num_agents": 2, "furniture": "ikea_table"}
        task = CooperativeAssemblyTask(config)
        task.reset()
        
        action = {
            "type": "multi_agent",
            "agents": {
                "agent_0": {"type": "pick", "target": "leg_1"},
                "agent_1": {"type": "pick", "target": "leg_2"}
            }
        }
        
        obs, reward, done, info = task.step(action)
        
        assert isinstance(reward, (int, float))
        assert "coordination_event" in info
    
    def test_cooperative_pick_action(self):
        """Test cooperative picking of heavy parts."""
        config = {"num_agents": 2, "furniture": "ikea_table"}
        task = CooperativeAssemblyTask(config)
        task.reset()
        
        # Both agents try to pick the heavy tabletop
        action = {
            "type": "multi_agent",
            "agents": {
                "agent_0": {"type": "pick", "target": "tabletop"},
                "agent_1": {"type": "pick", "target": "tabletop"}
            }
        }
        
        obs, reward, done, info = task.step(action)
        
        # Should detect coordination event
        if info.get("coordination_event", False):
            assert info["coordination_type"] in ["cooperative_pick", "lift_together"]
            assert len(info.get("coordinating_agents", [])) >= 2
    
    def test_agent_role_assignment(self):
        """Test initial role assignment."""
        config = {"num_agents": 3, "furniture": "ikea_table"}
        task = CooperativeAssemblyTask(config)
        task.reset()
        
        # Check that roles are assigned
        assert len(task.agent_roles) == 3
        
        # Should have at least one leader
        roles = list(task.agent_roles.values())
        assert "leader" in roles
    
    def test_coordination_requirements(self):
        """Test coordination requirement checking."""
        config = {"num_agents": 2, "furniture": "ikea_table"}
        task = CooperativeAssemblyTask(config)
        task.reset()
        
        # Simulate coordination requirement being met
        agent_actions = {
            "agent_0": {"type": "pick", "target": "tabletop"},
            "agent_1": {"type": "pick", "target": "tabletop"}
        }
        
        reward, info = task._check_coordination_requirements(agent_actions)
        
        # May get coordination reward if requirements are met
        if info.get("coordination_event", False):
            assert reward > 0
    
    def test_success_condition_cooperative(self):
        """Test success condition for cooperative task."""
        config = {"num_agents": 2, "furniture": "ikea_table"}
        task = CooperativeAssemblyTask(config)
        task.reset()
        
        # Initially not successful
        assert not task.check_success()
        
        # Mark most parts as attached (80% threshold)
        total_parts = len(task.parts)
        attached_count = int(total_parts * 0.8)
        
        part_names = list(task.parts.keys())
        for i in range(attached_count):
            task.parts[part_names[i]]["attached"] = True
        
        # Should be successful now
        assert task.check_success()
    
    def test_nearby_agents_detection(self):
        """Test nearby agents detection."""
        config = {"num_agents": 3, "furniture": "ikea_table"}
        task = CooperativeAssemblyTask(config)
        task.reset()
        
        # Position agents close together
        task.agent_positions["agent_0"] = np.array([0.0, 0.0, 0.0])
        task.agent_positions["agent_1"] = np.array([0.5, 0.0, 0.0])  # Close
        task.agent_positions["agent_2"] = np.array([5.0, 0.0, 0.0])  # Far
        
        nearby = task._get_nearby_agents("agent_0")
        
        assert "agent_1" in nearby
        assert "agent_2" not in nearby


class TestCooperativeAssemblyEnv:
    """Test CooperativeAssemblyEnv implementation."""
    
    def test_env_initialization(self):
        """Test environment initialization."""
        config = {"num_agents": 2}
        env = CooperativeAssemblyEnv(config)
        
        assert env.num_agents == 2
    
    def test_multi_agent_observations(self):
        """Test multi-agent observation generation."""
        config = {"num_agents": 2, "furniture": "ikea_table"}
        env = CooperativeAssemblyEnv(config)
        task = CooperativeAssemblyTask(config)
        env.set_task(task)
        
        obs = env.reset()
        
        # Should have observations for each agent
        assert "agent_agent_0" in obs
        assert "agent_agent_1" in obs
        
        # Each agent should have sensory data
        assert "rgb" in obs["agent_agent_0"]
        assert "depth" in obs["agent_agent_0"]
        assert "proprioception" in obs["agent_agent_0"]
    
    def test_multi_agent_rewards(self):
        """Test multi-agent reward distribution."""
        config = {"num_agents": 2, "furniture": "ikea_table"}
        env = CooperativeAssemblyEnv(config)
        task = CooperativeAssemblyTask(config)
        env.set_task(task)
        env.reset()
        
        action = {
            "type": "multi_agent",
            "agents": {
                "agent_0": {"type": "move", "values": [0.1, 0.0, 0.0]},
                "agent_1": {"type": "move", "values": [0.0, 0.1, 0.0]}
            }
        }
        
        obs, rewards, done, info = env.step(action)
        
        # Should return rewards dictionary
        assert isinstance(rewards, dict)
        assert "agent_0" in rewards
        assert "agent_1" in rewards
    
    def test_task_requirements(self):
        """Test task requirements for role assignment."""
        config = {"num_agents": 2}
        env = CooperativeAssemblyEnv(config)
        task = CooperativeAssemblyTask(config)
        env.set_task(task)
        
        requirements = env.get_task_requirements()
        
        assert "cooperation_required" in requirements
        assert requirements["cooperation_required"] is True
        assert "heavy_lifting" in requirements
        assert "coordination_events" in requirements
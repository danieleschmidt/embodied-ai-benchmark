"""Cooperative multi-agent furniture assembly task."""

import numpy as np
from typing import Any, Dict, List, Tuple, Optional

from ...core.base_task import BaseTask
from ...core.base_env import BaseEnv


class CooperativeAssemblyTask(BaseTask):
    """Multi-agent cooperative furniture assembly task."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.num_agents = config.get("num_agents", 2)
        self.furniture = config.get("furniture", "ikea_table")
        self.agent_roles = {}
        self.parts = self._generate_furniture_parts()
        self.agent_assignments = {}
        self.shared_workspace = self._define_workspace()
        self.coordination_requirements = self._get_coordination_requirements()
        self.agent_positions = {}
        self.collaboration_events = []
        
    def _generate_furniture_parts(self) -> Dict[str, Dict[str, Any]]:
        """Generate furniture parts requiring cooperation."""
        if self.furniture == "ikea_table":
            return {
                "tabletop": {
                    "position": np.array([0.0, 0.8, 0.0]),
                    "size": np.array([1.2, 0.05, 0.8]),
                    "mass": 8.0,  # Heavy - requires 2 agents
                    "requires_cooperation": True,
                    "min_agents": 2,
                    "attachment_points": ["leg_1", "leg_2", "leg_3", "leg_4"]
                },
                "leg_1": {
                    "position": np.array([0.5, 0.4, 0.35]),
                    "size": np.array([0.05, 0.8, 0.05]),
                    "mass": 1.2,
                    "requires_cooperation": False,
                    "min_agents": 1
                },
                "leg_2": {
                    "position": np.array([-0.5, 0.4, 0.35]),
                    "size": np.array([0.05, 0.8, 0.05]),
                    "mass": 1.2,
                    "requires_cooperation": False,
                    "min_agents": 1
                },
                "leg_3": {
                    "position": np.array([0.5, 0.4, -0.35]),
                    "size": np.array([0.05, 0.8, 0.05]),
                    "mass": 1.2,
                    "requires_cooperation": False,
                    "min_agents": 1
                },
                "leg_4": {
                    "position": np.array([-0.5, 0.4, -0.35]),
                    "size": np.array([0.05, 0.8, 0.05]),
                    "mass": 1.2,
                    "requires_cooperation": False,
                    "min_agents": 1
                },
                "support_beam": {
                    "position": np.array([0.0, 0.3, 0.0]),
                    "size": np.array([1.0, 0.05, 0.05]),
                    "mass": 3.0,
                    "requires_cooperation": True,
                    "min_agents": 2
                }
            }
        else:
            # Default cooperative task
            return {
                "heavy_base": {
                    "position": np.array([0.0, 0.1, 0.0]),
                    "size": np.array([1.0, 0.2, 1.0]),
                    "mass": 10.0,
                    "requires_cooperation": True,
                    "min_agents": 2
                },
                "top_piece": {
                    "position": np.array([0.0, 0.6, 0.0]),
                    "size": np.array([0.8, 0.1, 0.8]),
                    "mass": 2.0,
                    "requires_cooperation": False,
                    "min_agents": 1
                }
            }
    
    def _define_workspace(self) -> Dict[str, Any]:
        """Define shared workspace for cooperation."""
        return {
            "bounds": {
                "x": [-2.0, 2.0],
                "y": [0.0, 2.5],
                "z": [-2.0, 2.0]
            },
            "collaboration_zones": [
                {
                    "name": "assembly_area",
                    "center": np.array([0.0, 0.5, 0.0]),
                    "radius": 1.5
                },
                {
                    "name": "parts_storage",
                    "center": np.array([2.0, 0.2, 0.0]),
                    "radius": 0.8
                }
            ]
        }
    
    def _get_coordination_requirements(self) -> List[Dict[str, Any]]:
        """Define required coordination events."""
        return [
            {
                "type": "lift_together",
                "parts": ["tabletop"],
                "required_agents": 2,
                "synchronization_tolerance": 0.1  # seconds
            },
            {
                "type": "position_hold",
                "parts": ["tabletop"],
                "required_agents": 1,
                "while_other_attaches": True
            },
            {
                "type": "simultaneous_attachment",
                "parts": ["leg_1", "leg_2"],
                "required_agents": 2,
                "synchronization_tolerance": 0.2
            }
        ]
    
    def reset(self) -> Dict[str, Any]:
        super().reset()
        
        # Reset agent assignments and positions
        self.agent_assignments = {}
        self.agent_positions = {}
        self.collaboration_events = []
        
        # Initialize agent positions around workspace
        for i in range(self.num_agents):
            agent_id = f"agent_{i}"
            
            # Position agents around the assembly area
            angle = (2 * np.pi * i) / self.num_agents
            radius = 1.8
            x = radius * np.cos(angle)
            z = radius * np.sin(angle)
            
            self.agent_positions[agent_id] = np.array([x, 0.0, z])
            self.agent_roles[agent_id] = "unassigned"
        
        # Assign initial roles
        self._assign_initial_roles()
        
        # Randomize part positions
        for part_name, part_info in self.parts.items():
            # Add some random offset but keep within workspace
            offset = np.random.uniform(-0.2, 0.2, 3)
            offset[1] = max(0, offset[1])  # Keep above ground
            part_info["position"] = part_info["position"] + offset
            part_info["held_by"] = []
            part_info["attached"] = False
        
        return self._get_observation()
    
    def _assign_initial_roles(self):
        """Assign initial roles to agents."""
        agent_ids = list(self.agent_positions.keys())
        
        if len(agent_ids) >= 2:
            self.agent_roles[agent_ids[0]] = "leader"
            for i in range(1, len(agent_ids)):
                self.agent_roles[agent_ids[i]] = "follower"
    
    def step(self, action: Dict[str, Any]) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        self.current_step += 1
        
        # Process multi-agent joint action
        reward = 0.0
        info = {
            "task_success": False,
            "coordination_event": False,
            "coordinating_agents": [],
            "coordination_type": None
        }
        
        if action.get("type") == "multi_agent":
            agent_actions = action.get("agents", {})
            reward, info = self._process_multi_agent_actions(agent_actions, info)
        else:
            reward = -1.0  # Invalid action format
        
        # Check for required coordination
        coordination_reward, coord_info = self._check_coordination_requirements(agent_actions)
        reward += coordination_reward
        info.update(coord_info)
        
        # Check task completion
        success = self.check_success()
        info["task_success"] = success
        
        if success:
            reward += 200.0  # Large cooperation completion bonus
        
        # Cooperation bonus for simultaneous actions
        if self._check_simultaneous_actions(agent_actions):
            reward += 5.0
            info["coordination_event"] = True
            info["coordination_type"] = "simultaneous_action"
        
        # Time penalty
        reward -= 0.1
        
        done = self.is_done()
        observation = self._get_observation()
        
        # Record step data
        self._episode_data.append({
            "step": self.current_step,
            "agent_actions": agent_actions,
            "reward": reward,
            "cooperation_events": len(self.collaboration_events),
            "success": success
        })
        
        return observation, reward, done, info
    
    def _process_multi_agent_actions(self, 
                                   agent_actions: Dict[str, Dict[str, Any]], 
                                   info: Dict[str, Any]) -> Tuple[float, Dict[str, Any]]:
        """Process actions from multiple agents."""
        total_reward = 0.0
        
        for agent_id, action in agent_actions.items():
            agent_reward = 0.0
            action_type = action.get("type", "move")
            
            if action_type == "pick":
                agent_reward += self._process_pick_action(agent_id, action, info)
            elif action_type == "place":
                agent_reward += self._process_place_action(agent_id, action, info)
            elif action_type == "hold":
                agent_reward += self._process_hold_action(agent_id, action, info)
            elif action_type == "connect":
                agent_reward += self._process_connect_action(agent_id, action, info)
            elif action_type == "move":
                agent_reward += self._process_move_action(agent_id, action, info)
            elif action_type == "communicate":
                agent_reward += self._process_communication_action(agent_id, action, info)
            
            total_reward += agent_reward
        
        return total_reward, info
    
    def _process_pick_action(self, 
                           agent_id: str, 
                           action: Dict[str, Any], 
                           info: Dict[str, Any]) -> float:
        """Process pick action for specific agent."""
        part_name = action.get("target", "")
        
        if part_name not in self.parts:
            return -1.0
        
        part = self.parts[part_name]
        
        # Check if part requires cooperation
        if part.get("requires_cooperation", False):
            # Count how many agents are attempting to pick this part
            agents_picking = [part_name]  # Simplified - would need to track across agents
            
            if len(agents_picking) < part.get("min_agents", 1):
                return -0.5  # Not enough agents
            
            # Cooperative pick successful
            part["held_by"].append(agent_id)
            
            if len(part["held_by"]) >= part["min_agents"]:
                info["coordination_event"] = True
                info["coordination_type"] = "cooperative_pick"
                info["coordinating_agents"] = part["held_by"]
                return 5.0  # Cooperation bonus
        else:
            # Single agent pick
            if len(part.get("held_by", [])) == 0:
                part["held_by"] = [agent_id]
                return 1.0
        
        return 0.0
    
    def _process_place_action(self, 
                            agent_id: str, 
                            action: Dict[str, Any], 
                            info: Dict[str, Any]) -> float:
        """Process place action for specific agent."""
        part_name = action.get("target", "")
        target_position = np.array(action.get("position", [0, 0, 0]))
        
        if part_name not in self.parts:
            return -1.0
        
        part = self.parts[part_name]
        
        if agent_id not in part.get("held_by", []):
            return -0.5  # Agent not holding this part
        
        # Check if cooperative placement is required
        if part.get("requires_cooperation", False):
            holding_agents = part.get("held_by", [])
            
            if len(holding_agents) >= part["min_agents"]:
                # All required agents are holding - place together
                part["position"] = target_position
                part["held_by"] = []
                
                info["coordination_event"] = True
                info["coordination_type"] = "cooperative_place"
                info["coordinating_agents"] = holding_agents
                
                return 8.0  # Large cooperation reward
        else:
            # Single agent placement
            part["position"] = target_position
            part["held_by"] = []
            return 1.0
        
        return 0.0
    
    def _process_hold_action(self, 
                           agent_id: str, 
                           action: Dict[str, Any], 
                           info: Dict[str, Any]) -> float:
        """Process hold action (maintaining position while other agent works)."""
        part_name = action.get("target", "")
        
        if part_name not in self.parts:
            return -1.0
        
        part = self.parts[part_name]
        
        if agent_id in part.get("held_by", []):
            # Agent is helping by holding the part steady
            return 0.5  # Small continuous reward for cooperation
        
        return 0.0
    
    def _process_connect_action(self, 
                              agent_id: str, 
                              action: Dict[str, Any], 
                              info: Dict[str, Any]) -> float:
        """Process connection action between parts."""
        part1 = action.get("part1", "")
        part2 = action.get("part2", "")
        
        if part1 not in self.parts or part2 not in self.parts:
            return -1.0
        
        # Check if connection is valid and parts are close enough
        pos1 = self.parts[part1]["position"]
        pos2 = self.parts[part2]["position"]
        distance = np.linalg.norm(pos1 - pos2)
        
        if distance < 0.1:  # Close enough to connect
            # Successful connection
            self.parts[part1]["attached"] = True
            self.parts[part2]["attached"] = True
            
            return 10.0  # Large reward for successful connection
        
        return -0.3  # Parts not aligned properly
    
    def _process_move_action(self, 
                           agent_id: str, 
                           action: Dict[str, Any], 
                           info: Dict[str, Any]) -> float:
        """Process agent movement action."""
        target_position = np.array(action.get("values", [0, 0, 0]))
        
        if agent_id in self.agent_positions:
            # Update agent position
            self.agent_positions[agent_id] = target_position[:3]  # x, y, z only
            
            # Small reward for moving toward collaboration zones
            for zone in self.shared_workspace["collaboration_zones"]:
                distance_to_zone = np.linalg.norm(target_position[:3] - zone["center"])
                if distance_to_zone <= zone["radius"]:
                    return 0.1  # Small reward for being in collaboration area
        
        return -0.01  # Small movement penalty
    
    def _process_communication_action(self, 
                                    agent_id: str, 
                                    action: Dict[str, Any], 
                                    info: Dict[str, Any]) -> float:
        """Process communication between agents."""
        message_type = action.get("message_type", "coordinate")
        recipient = action.get("recipient", "broadcast")
        
        if message_type in ["coordinate", "request_help", "confirm_ready"]:
            return 0.2  # Small reward for meaningful communication
        
        return 0.0
    
    def _check_coordination_requirements(self, 
                                       agent_actions: Dict[str, Dict[str, Any]]) -> Tuple[float, Dict[str, Any]]:
        """Check if coordination requirements are being met."""
        reward = 0.0
        info = {}
        
        for requirement in self.coordination_requirements:
            req_type = requirement["type"]
            required_parts = requirement["parts"]
            required_agents = requirement["required_agents"]
            
            if req_type == "lift_together":
                # Check if multiple agents are picking the same heavy part
                for part_name in required_parts:
                    if part_name in self.parts:
                        picking_agents = [
                            agent_id for agent_id, action in agent_actions.items()
                            if (action.get("type") == "pick" and 
                                action.get("target") == part_name)
                        ]
                        
                        if len(picking_agents) >= required_agents:
                            reward += 10.0
                            info["coordination_event"] = True
                            info["coordination_type"] = "lift_together"
                            info["coordinating_agents"] = picking_agents
                            
                            # Record collaboration event
                            self.collaboration_events.append({
                                "step": self.current_step,
                                "type": "lift_together",
                                "agents": picking_agents,
                                "part": part_name
                            })
        
        return reward, info
    
    def _check_simultaneous_actions(self, agent_actions: Dict[str, Dict[str, Any]]) -> bool:
        """Check if agents are performing actions simultaneously."""
        action_types = [action.get("type", "") for action in agent_actions.values()]
        
        # Check for coordinated actions
        if action_types.count("pick") >= 2:
            return True
        if action_types.count("place") >= 2:
            return True
        if "hold" in action_types and "connect" in action_types:
            return True
        
        return False
    
    def check_success(self) -> bool:
        """Check if cooperative assembly is complete."""
        # All parts must be attached
        attached_parts = sum(1 for part in self.parts.values() if part.get("attached", False))
        total_parts = len(self.parts)
        
        # Require most parts to be attached (allowing for some flexibility)
        return attached_parts >= (total_parts * 0.8)
    
    def compute_reward(self) -> float:
        """Compute reward based on cooperation and assembly progress."""
        # Assembly progress
        attached_parts = sum(1 for part in self.parts.values() if part.get("attached", False))
        assembly_progress = attached_parts / len(self.parts)
        
        # Cooperation events
        cooperation_bonus = len(self.collaboration_events) * 2.0
        
        # Efficiency penalty
        efficiency_penalty = self.current_step / self.max_steps
        
        return assembly_progress * 20 + cooperation_bonus - efficiency_penalty
    
    def _get_observation(self) -> Dict[str, Any]:
        """Get multi-agent task observation."""
        # Global state information
        global_state = {
            "parts": {},
            "workspace": self.shared_workspace,
            "cooperation_requirements": self.coordination_requirements
        }
        
        # Part states
        for name, part in self.parts.items():
            global_state["parts"][name] = {
                "position": part["position"].tolist(),
                "size": part["size"].tolist(),
                "held_by": part.get("held_by", []),
                "attached": part.get("attached", False),
                "requires_cooperation": part.get("requires_cooperation", False)
            }
        
        # Agent-specific observations
        agent_observations = {}
        for agent_id, position in self.agent_positions.items():
            agent_observations[f"agent_{agent_id}"] = {
                "position": position.tolist(),
                "role": self.agent_roles.get(agent_id, "unassigned"),
                "nearby_agents": self._get_nearby_agents(agent_id),
                "visible_parts": self._get_visible_parts(agent_id),
                "assigned_task": self._get_agent_task(agent_id)
            }
        
        return {
            "global": global_state,
            "agents": agent_observations,
            "cooperation_events": len(self.collaboration_events),
            "step": self.current_step
        }
    
    def _get_nearby_agents(self, agent_id: str) -> List[str]:
        """Get list of nearby agents for coordination."""
        nearby = []
        agent_pos = self.agent_positions.get(agent_id, np.zeros(3))
        
        for other_id, other_pos in self.agent_positions.items():
            if other_id != agent_id:
                distance = np.linalg.norm(agent_pos - other_pos)
                if distance <= 2.0:  # Within 2 meters
                    nearby.append(other_id)
        
        return nearby
    
    def _get_visible_parts(self, agent_id: str) -> List[str]:
        """Get list of parts visible to agent."""
        visible = []
        agent_pos = self.agent_positions.get(agent_id, np.zeros(3))
        
        for part_name, part in self.parts.items():
            distance = np.linalg.norm(agent_pos - part["position"])
            if distance <= 3.0:  # Within 3 meters
                visible.append(part_name)
        
        return visible
    
    def _get_agent_task(self, agent_id: str) -> Dict[str, Any]:
        """Get current task assignment for agent."""
        role = self.agent_roles.get(agent_id, "unassigned")
        
        if role == "leader":
            return {
                "type": "coordinate",
                "description": "Coordinate team actions and manage heavy parts"
            }
        elif role == "follower":
            return {
                "type": "assist",
                "description": "Assist leader and handle individual tasks"
            }
        else:
            return {
                "type": "explore",
                "description": "Find optimal position and await assignment"
            }


class CooperativeAssemblyEnv(BaseEnv):
    """Environment for cooperative multi-agent assembly tasks."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.task = None
        self.num_agents = config.get("num_agents", 2)
        
    def set_task(self, task: CooperativeAssemblyTask):
        """Set the cooperative assembly task."""
        self.task = task
        
    def get_task_requirements(self) -> Dict[str, Any]:
        """Get task requirements for role assignment."""
        if self.task:
            return {
                "cooperation_required": True,
                "heavy_lifting": True,
                "coordination_events": len(self.task.coordination_requirements)
            }
        return {}
    
    def reset(self, seed: Optional[int] = None) -> Dict[str, Any]:
        super().reset(seed)
        if self.task:
            task_obs = self.task.reset()
        else:
            task_obs = {}
        
        # Environment observations for each agent
        env_obs = {}
        for i in range(self.num_agents):
            agent_id = f"agent_{i}"
            env_obs[f"agent_{agent_id}"] = {
                "rgb": np.random.randint(0, 255, (240, 320, 3), dtype=np.uint8),
                "depth": np.random.uniform(0.1, 5.0, (240, 320)).astype(np.float32),
                "proprioception": np.random.uniform(-1, 1, 12).astype(np.float32)
            }
        
        # Merge observations
        observation = {**env_obs, **task_obs}
        return observation
    
    def step(self, action: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, float], bool, Dict[str, Any]]:
        """Step with multi-agent action and return multi-agent rewards."""
        if not self.task:
            return self._get_observation(), {}, True, {"error": "No task set"}
        
        # Delegate to task
        task_obs, total_reward, done, info = self.task.step(action)
        
        # Distribute reward among agents
        rewards = {}
        if action.get("type") == "multi_agent":
            agent_actions = action.get("agents", {})
            num_active_agents = len(agent_actions)
            
            for agent_id in agent_actions.keys():
                # Base reward distribution
                rewards[agent_id] = total_reward / max(1, num_active_agents)
                
                # Cooperation bonus for agents involved in coordination
                if info.get("coordination_event", False):
                    coordinating_agents = info.get("coordinating_agents", [])
                    if agent_id in coordinating_agents:
                        rewards[agent_id] += 2.0  # Cooperation bonus
        
        # Add environment observations
        env_obs = {}
        for i in range(self.num_agents):
            agent_id = f"agent_{i}"
            env_obs[f"agent_{agent_id}"] = {
                "rgb": np.random.randint(0, 255, (240, 320, 3), dtype=np.uint8),
                "depth": np.random.uniform(0.1, 5.0, (240, 320)).astype(np.float32),
                "proprioception": np.random.uniform(-1, 1, 12).astype(np.float32)
            }
        
        observation = {**env_obs, **task_obs}
        return observation, rewards, done, info
    
    def render(self, mode: str = "rgb_array") -> Optional[np.ndarray]:
        """Render the cooperative assembly environment."""
        if mode == "rgb_array":
            # Return synthetic multi-agent view
            return np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        elif mode == "human":
            if self.task:
                print(f"Cooperative Assembly - Step: {self.task.current_step}")
                print(f"Agents: {self.num_agents}")
                print(f"Cooperation events: {len(self.task.collaboration_events)}")
        return None
    
    def close(self):
        """Clean up environment."""
        pass
    
    def _get_observation(self) -> Dict[str, Any]:
        """Get current environment observation."""
        env_obs = {}
        for i in range(self.num_agents):
            agent_id = f"agent_{i}"
            env_obs[f"agent_{agent_id}"] = {
                "rgb": np.random.randint(0, 255, (240, 320, 3), dtype=np.uint8),
                "depth": np.random.uniform(0.1, 5.0, (240, 320)).astype(np.float32),
                "proprioception": np.zeros(12, dtype=np.float32)
            }
        
        if self.task:
            task_obs = self.task._get_observation()
            return {**env_obs, **task_obs}
        
        return env_obs
"""Furniture assembly task implementation."""

import numpy as np
from typing import Any, Dict, List, Tuple, Optional

from ...core.base_task import BaseTask
from ...core.base_env import BaseEnv


class FurnitureAssemblyTask(BaseTask):
    """Furniture assembly task for single agents."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.furniture_type = config.get("furniture_type", "table")
        self.parts = self._generate_furniture_parts()
        self.target_assembly = self._generate_target_assembly()
        self.current_assembly = {}
        self.connections_made = set()
        self.required_connections = self._get_required_connections()
        
    def _generate_furniture_parts(self) -> Dict[str, Dict[str, Any]]:
        """Generate furniture parts based on type."""
        if self.furniture_type == "table":
            return {
                "tabletop": {
                    "position": np.array([0.0, 0.8, 0.0]),
                    "size": np.array([1.2, 0.05, 0.8]),
                    "mass": 5.0,
                    "attachment_points": ["leg_1", "leg_2", "leg_3", "leg_4"]
                },
                "leg_1": {
                    "position": np.array([0.5, 0.4, 0.35]),
                    "size": np.array([0.05, 0.8, 0.05]),
                    "mass": 1.0,
                    "attachment_points": ["tabletop"]
                },
                "leg_2": {
                    "position": np.array([-0.5, 0.4, 0.35]),
                    "size": np.array([0.05, 0.8, 0.05]),
                    "mass": 1.0,
                    "attachment_points": ["tabletop"]
                },
                "leg_3": {
                    "position": np.array([0.5, 0.4, -0.35]),
                    "size": np.array([0.05, 0.8, 0.05]),
                    "mass": 1.0,
                    "attachment_points": ["tabletop"]
                },
                "leg_4": {
                    "position": np.array([-0.5, 0.4, -0.35]),
                    "size": np.array([0.05, 0.8, 0.05]),
                    "mass": 1.0,
                    "attachment_points": ["tabletop"]
                }
            }
        elif self.furniture_type == "chair":
            return {
                "seat": {
                    "position": np.array([0.0, 0.45, 0.0]),
                    "size": np.array([0.4, 0.05, 0.4]),
                    "mass": 3.0,
                    "attachment_points": ["leg_1", "leg_2", "leg_3", "leg_4", "backrest"]
                },
                "backrest": {
                    "position": np.array([0.0, 0.7, -0.175]),
                    "size": np.array([0.4, 0.5, 0.05]),
                    "mass": 2.0,
                    "attachment_points": ["seat"]
                },
                "leg_1": {"position": np.array([0.175, 0.225, 0.175]), "size": np.array([0.05, 0.45, 0.05]), "mass": 0.8},
                "leg_2": {"position": np.array([-0.175, 0.225, 0.175]), "size": np.array([0.05, 0.45, 0.05]), "mass": 0.8},
                "leg_3": {"position": np.array([0.175, 0.225, -0.175]), "size": np.array([0.05, 0.45, 0.05]), "mass": 0.8},
                "leg_4": {"position": np.array([-0.175, 0.225, -0.175]), "size": np.array([0.05, 0.45, 0.05]), "mass": 0.8}
            }
        else:
            # Default simple furniture
            return {
                "base": {
                    "position": np.array([0.0, 0.0, 0.0]),
                    "size": np.array([0.5, 0.1, 0.5]),
                    "mass": 2.0,
                    "attachment_points": ["top"]
                },
                "top": {
                    "position": np.array([0.0, 0.5, 0.0]),
                    "size": np.array([0.4, 0.1, 0.4]),
                    "mass": 1.5,
                    "attachment_points": ["base"]
                }
            }
    
    def _generate_target_assembly(self) -> Dict[str, Any]:
        """Generate target assembly configuration."""
        return {
            "connections": self._get_required_connections(),
            "stability_threshold": 0.9,
            "alignment_tolerance": 0.05  # 5cm tolerance
        }
    
    def _get_required_connections(self) -> List[Tuple[str, str]]:
        """Get required connections for successful assembly."""
        if self.furniture_type == "table":
            return [
                ("tabletop", "leg_1"),
                ("tabletop", "leg_2"), 
                ("tabletop", "leg_3"),
                ("tabletop", "leg_4")
            ]
        elif self.furniture_type == "chair":
            return [
                ("seat", "leg_1"),
                ("seat", "leg_2"),
                ("seat", "leg_3"), 
                ("seat", "leg_4"),
                ("seat", "backrest")
            ]
        else:
            return [("base", "top")]
    
    def reset(self) -> Dict[str, Any]:
        super().reset()
        self.current_assembly = {}
        self.connections_made = set()
        
        # Randomize initial part positions
        for part_name, part_info in self.parts.items():
            offset = np.random.uniform(-0.3, 0.3, 3)
            offset[1] = max(0, offset[1])  # Keep parts above ground
            part_info["position"] = part_info["position"] + offset
            part_info["attached"] = False
            part_info["stability"] = 0.0
        
        return self._get_observation()
    
    def step(self, action: Dict[str, Any]) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        self.current_step += 1
        
        # Process action (manipulation command)
        reward = 0.0
        info = {"task_success": False, "collision": False}
        
        action_type = action.get("type", "move")
        
        if action_type == "pick":
            reward += self._process_pick_action(action, info)
        elif action_type == "place":
            reward += self._process_place_action(action, info)
        elif action_type == "connect":
            reward += self._process_connect_action(action, info)
        elif action_type == "move":
            reward += self._process_move_action(action, info)
        
        # Check task completion
        success = self.check_success()
        info["task_success"] = success
        
        if success:
            reward += 100.0  # Completion bonus
        
        # Time penalty
        reward -= 0.1
        
        # Step penalty for inefficiency
        if self.current_step > self.max_steps * 0.8:
            reward -= 0.5
        
        done = self.is_done()
        observation = self._get_observation()
        
        # Record step data
        self._episode_data.append({
            "step": self.current_step,
            "action": action,
            "reward": reward,
            "success": success,
            "connections_made": len(self.connections_made),
            "total_connections": len(self.required_connections)
        })
        
        return observation, reward, done, info
    
    def _process_pick_action(self, action: Dict[str, Any], info: Dict[str, Any]) -> float:
        """Process pick action."""
        part_name = action.get("target", "")
        
        if part_name not in self.parts:
            return -1.0  # Invalid part
        
        part = self.parts[part_name]
        
        # Check if part is pickable (not attached)
        if part.get("attached", False):
            return -0.5  # Cannot pick attached part
        
        # Success - part picked
        part["picked"] = True
        return 1.0
    
    def _process_place_action(self, action: Dict[str, Any], info: Dict[str, Any]) -> float:
        """Process place action."""
        part_name = action.get("target", "")
        target_position = np.array(action.get("position", [0, 0, 0]))
        
        if part_name not in self.parts:
            return -1.0
        
        part = self.parts[part_name]
        
        if not part.get("picked", False):
            return -0.5  # Part not picked
        
        # Update part position
        part["position"] = target_position
        part["picked"] = False
        
        # Check for collisions
        collision = self._check_collision(part_name, target_position)
        if collision:
            info["collision"] = True
            return -2.0
        
        return 0.5
    
    def _process_connect_action(self, action: Dict[str, Any], info: Dict[str, Any]) -> float:
        """Process connection action."""
        part1 = action.get("part1", "")
        part2 = action.get("part2", "")
        
        if part1 not in self.parts or part2 not in self.parts:
            return -1.0
        
        # Check if this is a valid connection
        connection = (part1, part2) if (part1, part2) in self.required_connections else (part2, part1)
        
        if connection not in self.required_connections:
            return -0.5  # Invalid connection
        
        if connection in self.connections_made:
            return -0.2  # Already connected
        
        # Check proximity and alignment
        pos1 = self.parts[part1]["position"]
        pos2 = self.parts[part2]["position"]
        distance = np.linalg.norm(pos1 - pos2)
        
        if distance > self.target_assembly["alignment_tolerance"] * 3:
            return -0.3  # Too far apart
        
        # Successful connection
        self.connections_made.add(connection)
        self.parts[part1]["attached"] = True
        self.parts[part2]["attached"] = True
        
        return 10.0  # Large reward for successful connection
    
    def _process_move_action(self, action: Dict[str, Any], info: Dict[str, Any]) -> float:
        """Process general movement action."""
        # Simple movement with small reward/penalty
        target_pos = np.array(action.get("values", [0, 0, 0]))
        
        # Penalty for large movements (encourage efficiency)
        movement_magnitude = np.linalg.norm(target_pos)
        return -0.01 * movement_magnitude
    
    def _check_collision(self, part_name: str, position: np.ndarray) -> bool:
        """Check if placing part at position causes collision."""
        part_size = self.parts[part_name]["size"]
        
        for other_name, other_part in self.parts.items():
            if other_name == part_name:
                continue
            
            other_pos = other_part["position"]
            other_size = other_part["size"]
            
            # Simple AABB collision detection
            if (abs(position[0] - other_pos[0]) < (part_size[0] + other_size[0]) / 2 and
                abs(position[1] - other_pos[1]) < (part_size[1] + other_size[1]) / 2 and
                abs(position[2] - other_pos[2]) < (part_size[2] + other_size[2]) / 2):
                return True
        
        return False
    
    def check_success(self) -> bool:
        """Check if assembly is complete and stable."""
        # All required connections must be made
        if len(self.connections_made) < len(self.required_connections):
            return False
        
        # Check stability (simplified)
        total_stability = 0.0
        for part in self.parts.values():
            if part.get("attached", False):
                total_stability += 1.0
        
        stability_ratio = total_stability / len(self.parts)
        return stability_ratio >= self.target_assembly["stability_threshold"]
    
    def compute_reward(self) -> float:
        """Compute reward based on current assembly state."""
        # Connection progress reward
        connection_progress = len(self.connections_made) / len(self.required_connections)
        
        # Stability reward
        attached_parts = sum(1 for part in self.parts.values() if part.get("attached", False))
        stability_reward = attached_parts / len(self.parts)
        
        # Efficiency penalty
        efficiency_penalty = self.current_step / self.max_steps
        
        return connection_progress * 10 + stability_reward * 5 - efficiency_penalty
    
    def _get_observation(self) -> Dict[str, Any]:
        """Get current task observation."""
        # Part positions and states
        part_states = {}
        for name, part in self.parts.items():
            part_states[name] = {
                "position": part["position"].tolist(),
                "size": part["size"].tolist(),
                "attached": part.get("attached", False),
                "picked": part.get("picked", False)
            }
        
        # Connection state
        connection_matrix = np.zeros((len(self.parts), len(self.parts)))
        part_names = list(self.parts.keys())
        
        for connection in self.connections_made:
            i = part_names.index(connection[0])
            j = part_names.index(connection[1])
            connection_matrix[i, j] = 1
            connection_matrix[j, i] = 1
        
        return {
            "parts": part_states,
            "connections": connection_matrix.tolist(),
            "progress": len(self.connections_made) / len(self.required_connections),
            "task_step": self.current_step,
            "furniture_type": self.furniture_type
        }


class FurnitureAssemblyEnv(BaseEnv):
    """Environment for furniture assembly tasks."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.task = None
        self.workspace_bounds = config.get("workspace_bounds", {
            "x": [-2.0, 2.0],
            "y": [0.0, 2.0], 
            "z": [-2.0, 2.0]
        })
        
    def set_task(self, task: FurnitureAssemblyTask):
        """Set the assembly task."""
        self.task = task
    
    def reset(self, seed: Optional[int] = None) -> Dict[str, Any]:
        super().reset(seed)
        if self.task:
            task_obs = self.task.reset()
        else:
            task_obs = {}
        
        # Add environment-specific observations
        env_obs = {
            "rgb": np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8),
            "depth": np.random.uniform(0.1, 5.0, (480, 640)).astype(np.float32),
            "proprioception": np.zeros(12, dtype=np.float32),
            "workspace_bounds": self.workspace_bounds
        }
        
        # Merge observations
        observation = {**env_obs, **task_obs}
        return observation
    
    def step(self, action: Dict[str, Any]) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        if not self.task:
            return self._get_observation(), 0.0, True, {"error": "No task set"}
        
        # Check workspace bounds
        if "position" in action:
            pos = np.array(action["position"])
            if not self._in_workspace(pos):
                info = {"workspace_violation": True}
                return self._get_observation(), -5.0, False, info
        
        # Delegate to task
        task_obs, reward, done, info = self.task.step(action)
        
        # Add environment observations
        env_obs = {
            "rgb": np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8),
            "depth": np.random.uniform(0.1, 5.0, (480, 640)).astype(np.float32),
            "proprioception": np.random.uniform(-1, 1, 12).astype(np.float32)
        }
        
        observation = {**env_obs, **task_obs}
        return observation, reward, done, info
    
    def _in_workspace(self, position: np.ndarray) -> bool:
        """Check if position is within workspace bounds."""
        bounds = self.workspace_bounds
        return (bounds["x"][0] <= position[0] <= bounds["x"][1] and
                bounds["y"][0] <= position[1] <= bounds["y"][1] and
                bounds["z"][0] <= position[2] <= bounds["z"][1])
    
    def render(self, mode: str = "rgb_array") -> Optional[np.ndarray]:
        """Render the environment."""
        if mode == "rgb_array":
            # Return synthetic image for now
            return np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        elif mode == "human":
            print(f"Furniture Assembly Environment - Step: {self.task.current_step if self.task else 0}")
        return None
    
    def close(self):
        """Clean up environment."""
        pass
    
    def _get_observation(self) -> Dict[str, Any]:
        """Get current environment observation."""
        env_obs = {
            "rgb": np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8),
            "depth": np.random.uniform(0.1, 5.0, (480, 640)).astype(np.float32),
            "proprioception": np.zeros(12, dtype=np.float32)
        }
        
        if self.task:
            task_obs = self.task._get_observation()
            return {**env_obs, **task_obs}
        
        return env_obs
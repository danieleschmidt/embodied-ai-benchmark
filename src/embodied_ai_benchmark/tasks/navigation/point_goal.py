"""Point goal navigation task implementation."""

import numpy as np
from typing import Any, Dict, List, Tuple, Optional

from ...core.base_task import BaseTask
from ...core.base_env import BaseEnv


class PointGoalTask(BaseTask):
    """Point goal navigation task."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.room_size = config.get("room_size", (10, 10))
        self.goal_radius = config.get("goal_radius", 0.5)
        self.start_position = np.array([0.0, 0.0])
        self.goal_position = np.array([0.0, 0.0])
        self.current_position = np.array([0.0, 0.0])
        self.obstacles = self._generate_obstacles(config.get("num_obstacles", 5))
        self.path_history = []
        self.min_goal_distance = float('inf')
        
    def _generate_obstacles(self, num_obstacles: int) -> List[Dict[str, Any]]:
        """Generate random obstacles in the room."""
        obstacles = []
        
        for i in range(num_obstacles):
            # Random position within room bounds
            pos_x = np.random.uniform(-self.room_size[0]/2 + 1, self.room_size[0]/2 - 1)
            pos_y = np.random.uniform(-self.room_size[1]/2 + 1, self.room_size[1]/2 - 1)
            
            obstacle = {
                "id": f"obstacle_{i}",
                "position": np.array([pos_x, pos_y]),
                "radius": np.random.uniform(0.3, 0.8),
                "type": "circular"
            }
            obstacles.append(obstacle)
        
        return obstacles
    
    def reset(self) -> Dict[str, Any]:
        super().reset()
        
        # Reset positions
        self.start_position = self._sample_free_position()
        self.goal_position = self._sample_free_position()
        
        # Ensure goal is not too close to start
        while np.linalg.norm(self.goal_position - self.start_position) < 2.0:
            self.goal_position = self._sample_free_position()
        
        self.current_position = self.start_position.copy()
        self.path_history = [self.current_position.copy()]
        self.min_goal_distance = np.linalg.norm(self.goal_position - self.current_position)
        
        return self._get_observation()
    
    def _sample_free_position(self) -> np.ndarray:
        """Sample a position that doesn't collide with obstacles."""
        max_attempts = 100
        
        for _ in range(max_attempts):
            x = np.random.uniform(-self.room_size[0]/2 + 0.5, self.room_size[0]/2 - 0.5)
            y = np.random.uniform(-self.room_size[1]/2 + 0.5, self.room_size[1]/2 - 0.5)
            position = np.array([x, y])
            
            # Check if position is free
            if self._is_position_free(position):
                return position
        
        # Fallback to origin if no free position found
        return np.array([0.0, 0.0])
    
    def _is_position_free(self, position: np.ndarray, agent_radius: float = 0.3) -> bool:
        """Check if position is free from obstacles."""
        for obstacle in self.obstacles:
            distance = np.linalg.norm(position - obstacle["position"])
            if distance < (obstacle["radius"] + agent_radius):
                return False
        return True
    
    def step(self, action: Dict[str, Any]) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        self.current_step += 1
        
        # Process movement action
        action_type = action.get("type", "velocity")
        reward = 0.0
        info = {"task_success": False, "collision": False}
        
        if action_type == "velocity":
            reward += self._process_velocity_action(action, info)
        elif action_type == "position":
            reward += self._process_position_action(action, info)
        else:
            reward -= 0.1  # Invalid action penalty
        
        # Check goal reached
        goal_distance = np.linalg.norm(self.current_position - self.goal_position)
        
        if goal_distance <= self.goal_radius:
            info["task_success"] = True
            reward += 100.0  # Large completion reward
        else:
            # Distance-based reward shaping
            if goal_distance < self.min_goal_distance:
                reward += 1.0  # Progress reward
                self.min_goal_distance = goal_distance
            
            # Distance penalty (encourage getting closer)
            reward -= goal_distance * 0.1
        
        # Time penalty
        reward -= 0.1
        
        # Efficiency penalty for long paths
        if len(self.path_history) > 1:
            path_length = sum(
                np.linalg.norm(self.path_history[i] - self.path_history[i-1])
                for i in range(1, len(self.path_history))
            )
            straight_distance = np.linalg.norm(self.goal_position - self.start_position)
            if path_length > straight_distance * 1.5:
                reward -= 0.05
        
        done = self.is_done()
        observation = self._get_observation()
        
        # Record step data
        self._episode_data.append({
            "step": self.current_step,
            "position": self.current_position.copy(),
            "goal_distance": goal_distance,
            "action": action,
            "reward": reward,
            "collision": info["collision"]
        })
        
        return observation, reward, done, info
    
    def _process_velocity_action(self, action: Dict[str, Any], info: Dict[str, Any]) -> float:
        """Process velocity-based movement action."""
        velocity = np.array(action.get("values", [0.0, 0.0]))
        
        # Limit velocity magnitude
        max_velocity = 1.0
        if np.linalg.norm(velocity) > max_velocity:
            velocity = velocity / np.linalg.norm(velocity) * max_velocity
        
        # Compute new position
        dt = 0.1  # Time step
        new_position = self.current_position + velocity * dt
        
        # Check room bounds
        new_position[0] = np.clip(new_position[0], -self.room_size[0]/2, self.room_size[0]/2)
        new_position[1] = np.clip(new_position[1], -self.room_size[1]/2, self.room_size[1]/2)
        
        # Check collision with obstacles
        if not self._is_position_free(new_position):
            info["collision"] = True
            return -5.0  # Collision penalty
        
        # Update position
        self.current_position = new_position
        self.path_history.append(self.current_position.copy())
        
        # Small reward for valid movement
        return 0.1
    
    def _process_position_action(self, action: Dict[str, Any], info: Dict[str, Any]) -> float:
        """Process absolute position action."""
        target_position = np.array(action.get("values", self.current_position))
        
        # Check if movement is reasonable (not teleporting)
        movement_distance = np.linalg.norm(target_position - self.current_position)
        max_movement = 0.5
        
        if movement_distance > max_movement:
            # Clip movement to maximum allowed
            direction = (target_position - self.current_position) / movement_distance
            target_position = self.current_position + direction * max_movement
        
        # Check room bounds
        target_position[0] = np.clip(target_position[0], -self.room_size[0]/2, self.room_size[0]/2)
        target_position[1] = np.clip(target_position[1], -self.room_size[1]/2, self.room_size[1]/2)
        
        # Check collision
        if not self._is_position_free(target_position):
            info["collision"] = True
            return -5.0
        
        # Update position
        self.current_position = target_position
        self.path_history.append(self.current_position.copy())
        
        return 0.1
    
    def check_success(self) -> bool:
        """Check if goal has been reached."""
        distance = np.linalg.norm(self.current_position - self.goal_position)
        return distance <= self.goal_radius
    
    def compute_reward(self) -> float:
        """Compute reward based on current state."""
        goal_distance = np.linalg.norm(self.current_position - self.goal_position)
        
        if self.check_success():
            return 100.0  # Success reward
        
        # Distance-based reward (closer is better)
        max_distance = np.linalg.norm(self.room_size)
        distance_reward = (max_distance - goal_distance) / max_distance * 10
        
        # Efficiency reward (shorter path is better)
        if len(self.path_history) > 1:
            path_length = sum(
                np.linalg.norm(self.path_history[i] - self.path_history[i-1])
                for i in range(1, len(self.path_history))
            )
            straight_distance = np.linalg.norm(self.goal_position - self.start_position)
            efficiency = max(0, 1 - (path_length / (straight_distance + 1e-6)))
            efficiency_reward = efficiency * 5
        else:
            efficiency_reward = 0
        
        return distance_reward + efficiency_reward
    
    def _get_observation(self) -> Dict[str, Any]:
        """Get current navigation observation."""
        # Agent state
        agent_state = {
            "position": self.current_position.tolist(),
            "goal_position": self.goal_position.tolist(),
            "goal_distance": float(np.linalg.norm(self.current_position - self.goal_position))
        }
        
        # Environment state
        obstacle_positions = [obs["position"].tolist() for obs in self.obstacles]
        obstacle_radii = [obs["radius"] for obs in self.obstacles]
        
        # Local occupancy grid around agent (for navigation)
        grid_resolution = 0.1
        grid_size = 21  # 21x21 grid (2.1m x 2.1m around agent)
        occupancy_grid = np.zeros((grid_size, grid_size))
        
        center = grid_size // 2
        for i in range(grid_size):
            for j in range(grid_size):
                # World position for this grid cell
                world_x = self.current_position[0] + (i - center) * grid_resolution
                world_y = self.current_position[1] + (j - center) * grid_resolution
                world_pos = np.array([world_x, world_y])
                
                # Check if occupied by obstacle
                for obstacle in self.obstacles:
                    if np.linalg.norm(world_pos - obstacle["position"]) <= obstacle["radius"]:
                        occupancy_grid[i, j] = 1.0
                        break
        
        # Goal direction (relative to agent)
        goal_direction = self.goal_position - self.current_position
        goal_direction_norm = goal_direction / (np.linalg.norm(goal_direction) + 1e-6)
        
        return {
            "agent": agent_state,
            "obstacles": {
                "positions": obstacle_positions,
                "radii": obstacle_radii
            },
            "occupancy_grid": occupancy_grid.tolist(),
            "goal_direction": goal_direction_norm.tolist(),
            "room_bounds": self.room_size,
            "step": self.current_step
        }


class PointGoalEnv(BaseEnv):
    """Environment for point goal navigation."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.task = None
        self.render_scale = 20  # pixels per meter
        
    def set_task(self, task: PointGoalTask):
        """Set the navigation task."""
        self.task = task
    
    def reset(self, seed: Optional[int] = None) -> Dict[str, Any]:
        super().reset(seed)
        if self.task:
            task_obs = self.task.reset()
        else:
            task_obs = {}
        
        # Environment observations
        env_obs = {
            "rgb": self._render_top_down(),
            "depth": np.random.uniform(0.1, 10.0, (480, 640)).astype(np.float32),
            "proprioception": np.array([
                self.task.current_position[0] if self.task else 0,
                self.task.current_position[1] if self.task else 0,
                0, 0, 0, 0,  # velocity, acceleration
                0, 0, 0, 0, 0, 0   # IMU data
            ], dtype=np.float32)
        }
        
        return {**env_obs, **task_obs}
    
    def step(self, action: Dict[str, Any]) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        if not self.task:
            return self._get_observation(), 0.0, True, {"error": "No task set"}
        
        # Delegate to task
        task_obs, reward, done, info = self.task.step(action)
        
        # Update environment observations
        env_obs = {
            "rgb": self._render_top_down(),
            "depth": np.random.uniform(0.1, 10.0, (480, 640)).astype(np.float32),
            "proprioception": np.array([
                self.task.current_position[0],
                self.task.current_position[1],
                0, 0, 0, 0,  # velocity, acceleration
                0, 0, 0, 0, 0, 0   # IMU data
            ], dtype=np.float32)
        }
        
        observation = {**env_obs, **task_obs}
        return observation, reward, done, info
    
    def _render_top_down(self) -> np.ndarray:
        """Render top-down view of the navigation environment."""
        if not self.task:
            return np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Create image
        img_height, img_width = 480, 640
        image = np.ones((img_height, img_width, 3), dtype=np.uint8) * 255  # White background
        
        # Scale and center
        room_to_img_scale = min(img_width / self.task.room_size[0], img_height / self.task.room_size[1]) * 0.8
        center_x, center_y = img_width // 2, img_height // 2
        
        def world_to_img(world_pos):
            img_x = int(center_x + world_pos[0] * room_to_img_scale)
            img_y = int(center_y - world_pos[1] * room_to_img_scale)  # Flip Y axis
            return img_x, img_y
        
        # Draw room bounds
        room_corners = [
            [-self.task.room_size[0]/2, -self.task.room_size[1]/2],
            [self.task.room_size[0]/2, -self.task.room_size[1]/2],
            [self.task.room_size[0]/2, self.task.room_size[1]/2],
            [-self.task.room_size[0]/2, self.task.room_size[1]/2]
        ]
        
        # Draw obstacles
        for obstacle in self.task.obstacles:
            center = world_to_img(obstacle["position"])
            radius = int(obstacle["radius"] * room_to_img_scale)
            
            # Draw filled circle for obstacle
            y, x = np.ogrid[:img_height, :img_width]
            mask = (x - center[0])**2 + (y - center[1])**2 <= radius**2
            image[mask] = [100, 100, 100]  # Gray obstacles
        
        # Draw agent
        agent_pos = world_to_img(self.task.current_position)
        agent_radius = int(0.3 * room_to_img_scale)
        y, x = np.ogrid[:img_height, :img_width]
        mask = (x - agent_pos[0])**2 + (y - agent_pos[1])**2 <= agent_radius**2
        image[mask] = [0, 0, 255]  # Blue agent
        
        # Draw goal
        goal_pos = world_to_img(self.task.goal_position)
        goal_radius = int(self.task.goal_radius * room_to_img_scale)
        y, x = np.ogrid[:img_height, :img_width]
        mask = (x - goal_pos[0])**2 + (y - goal_pos[1])**2 <= goal_radius**2
        image[mask] = [0, 255, 0]  # Green goal
        
        # Draw path history
        if len(self.task.path_history) > 1:
            for i in range(1, len(self.task.path_history)):
                start = world_to_img(self.task.path_history[i-1])
                end = world_to_img(self.task.path_history[i])
                # Simple line drawing (could use cv2.line for better quality)
                # For now, just mark path points
                if 0 <= end[0] < img_width and 0 <= end[1] < img_height:
                    image[end[1], end[0]] = [255, 0, 0]  # Red path
        
        return image
    
    def render(self, mode: str = "rgb_array") -> Optional[np.ndarray]:
        """Render the environment."""
        if mode == "rgb_array":
            return self._render_top_down()
        elif mode == "human":
            if self.task:
                print(f"Navigation Environment - Step: {self.task.current_step}")
                print(f"Position: {self.task.current_position}")
                print(f"Goal: {self.task.goal_position}")
                print(f"Distance to goal: {np.linalg.norm(self.task.current_position - self.task.goal_position):.2f}")
        return None
    
    def close(self):
        """Clean up environment."""
        pass
    
    def _get_observation(self) -> Dict[str, Any]:
        """Get current environment observation."""
        env_obs = {
            "rgb": self._render_top_down(),
            "depth": np.random.uniform(0.1, 10.0, (480, 640)).astype(np.float32),
            "proprioception": np.zeros(12, dtype=np.float32)
        }
        
        if self.task:
            task_obs = self.task._get_observation()
            return {**env_obs, **task_obs}
        
        return env_obs
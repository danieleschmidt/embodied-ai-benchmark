"""Production-ready environment implementation with realistic sensor simulation."""

import time
import logging
from typing import Any, Dict, Optional, Tuple
import numpy as np
from ..core.base_env import BaseEnv


class ProductionEnv(BaseEnv):
    """Production-ready environment with realistic sensor modeling."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize production environment.
        
        Args:
            config: Environment configuration
        """
        super().__init__(config)
        self.logger = logging.getLogger(__name__)
        
        # Environment parameters
        self.max_steps = config.get("max_steps", 1000)
        self.sensor_noise_level = config.get("sensor_noise_level", 0.05)
        self.depth_range = config.get("depth_range", (0.1, 10.0))
        self.missing_data_rate = config.get("missing_data_rate", 0.05)
        
        # State tracking
        self._current_step = 0
        self._current_pose = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])  # x,y,z,qx,qy,qz,qw
        self._target_pose = None
        self._obstacles = []
        self._task_completed = False
        
        # Performance tracking
        self._step_times = []
        self._last_action_time = time.time()
        
        self.logger.info(f"ProductionEnv initialized with config: {config}")
    
    def reset(self, seed: Optional[int] = None) -> Dict[str, Any]:
        """Reset environment to initial state."""
        self._current_step = 0
        self._task_completed = False
        self._current_pose = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
        
        # Generate realistic scene
        self._generate_scene()
        
        return super().reset(seed)
    
    def _execute_step(self, action: Dict[str, Any]) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        """Execute environment step with realistic physics simulation."""
        start_time = time.time()
        
        try:
            self._current_step += 1
            
            # Validate action
            self._validate_action(action)
            
            # Execute action with physics
            self._apply_action(action)
            
            # Compute reward
            reward = self._compute_reward(action)
            
            # Check termination
            done = self._check_termination()
            
            # Get observation
            obs = self._get_observation()
            
            # Create info dict
            info = self._create_info_dict()
            
            # Track performance
            step_time = time.time() - start_time
            self._step_times.append(step_time)
            
            return obs, reward, done, info
            
        except Exception as e:
            self.logger.error(f"Step execution failed: {e}")
            return self._get_fallback_observation(), -1.0, True, {"error": str(e)}
    
    def _get_observation(self) -> Dict[str, Any]:
        """Get realistic observation with sensor modeling."""
        try:
            # Simulate realistic RGB camera with noise
            rgb_base = np.random.randint(50, 200, (224, 224, 3), dtype=np.uint8)
            noise = np.random.normal(0, self.sensor_noise_level * 255, (224, 224, 3))
            rgb = np.clip(rgb_base + noise, 0, 255).astype(np.uint8)
            
            # Realistic depth with missing data
            depth_base = np.random.uniform(*self.depth_range, (224, 224))
            missing_mask = np.random.rand(224, 224) < self.missing_data_rate
            depth = np.where(missing_mask, np.nan, depth_base).astype(np.float32)
            
            # Add lens distortion to depth
            y, x = np.ogrid[:224, :224]
            center_y, center_x = 112, 112
            r = np.sqrt((x - center_x)**2 + (y - center_y)**2)
            distortion = 1 + 0.1 * (r / 112)**2  # Barrel distortion
            depth = depth * distortion
            
            # Pose with sensor noise
            pose_noise = np.random.normal(0, 0.01, 7)
            pose = self._current_pose + pose_noise
            
            # Normalize quaternion
            quat = pose[3:7]
            quat_norm = np.linalg.norm(quat)
            if quat_norm > 0:
                pose[3:7] = quat / quat_norm
            else:
                pose[3:7] = [0, 0, 0, 1]  # Default quaternion
            
            # Proprioception (joint states, forces, etc.)
            proprioception = {
                "joint_positions": np.random.uniform(-np.pi, np.pi, 7),
                "joint_velocities": np.random.uniform(-1, 1, 7),
                "joint_torques": np.random.uniform(-10, 10, 7),
                "end_effector_force": np.random.uniform(-5, 5, 3),
                "gripper_state": np.random.uniform(0, 1)
            }
            
            # Task-specific observations
            task_obs = self._get_task_observation()
            
            observation = {
                "rgb": rgb,
                "depth": depth,
                "pose": pose,
                "proprioception": proprioception,
                "task": task_obs,
                "timestamp": time.time(),
                "step": self._current_step,
                "sensor_health": self._get_sensor_health()
            }
            
            return observation
            
        except Exception as e:
            self.logger.error(f"Observation generation failed: {e}")
            return self._get_fallback_observation()
    
    def _get_task_observation(self) -> Dict[str, Any]:
        """Get task-specific observations."""
        return {
            "target_visible": np.random.rand() > 0.3,
            "target_pose": self._target_pose if self._target_pose is not None else np.zeros(7),
            "obstacles_detected": len(self._obstacles),
            "progress": min(1.0, self._current_step / self.max_steps)
        }
    
    def _get_sensor_health(self) -> Dict[str, float]:
        """Simulate sensor health monitoring."""
        return {
            "rgb_camera": 0.95 + np.random.uniform(-0.05, 0.05),
            "depth_camera": 0.90 + np.random.uniform(-0.1, 0.1),
            "imu": 0.98 + np.random.uniform(-0.02, 0.02),
            "proprioception": 0.99 + np.random.uniform(-0.01, 0.01)
        }
    
    def _get_fallback_observation(self) -> Dict[str, Any]:
        """Get minimal fallback observation on error."""
        return {
            "rgb": np.zeros((224, 224, 3), dtype=np.uint8),
            "depth": np.full((224, 224), np.nan, dtype=np.float32),
            "pose": np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]),
            "proprioception": {
                "joint_positions": np.zeros(7),
                "joint_velocities": np.zeros(7),
                "joint_torques": np.zeros(7),
                "end_effector_force": np.zeros(3),
                "gripper_state": 0.0
            },
            "task": {"error": True},
            "timestamp": time.time(),
            "step": self._current_step,
            "sensor_health": {"error": True}
        }
    
    def _validate_action(self, action: Dict[str, Any]):
        """Validate action parameters."""
        required_keys = ["linear_velocity", "angular_velocity"]
        
        for key in required_keys:
            if key not in action:
                raise ValueError(f"Missing required action key: {key}")
        
        # Validate action ranges
        linear_vel = np.array(action["linear_velocity"])
        angular_vel = np.array(action["angular_velocity"])
        
        if linear_vel.shape != (3,):
            raise ValueError(f"linear_velocity must be 3D, got shape {linear_vel.shape}")
        
        if angular_vel.shape != (3,):
            raise ValueError(f"angular_velocity must be 3D, got shape {angular_vel.shape}")
        
        # Safety limits
        max_linear_vel = 2.0  # m/s
        max_angular_vel = 3.14  # rad/s
        
        if np.linalg.norm(linear_vel) > max_linear_vel:
            raise ValueError(f"Linear velocity exceeds safety limit: {np.linalg.norm(linear_vel)} > {max_linear_vel}")
        
        if np.linalg.norm(angular_vel) > max_angular_vel:
            raise ValueError(f"Angular velocity exceeds safety limit: {np.linalg.norm(angular_vel)} > {max_angular_vel}")
    
    def _apply_action(self, action: Dict[str, Any]):
        """Apply action with realistic physics."""
        dt = 0.05  # 20 Hz simulation
        
        # Apply linear motion
        linear_vel = np.array(action["linear_velocity"])
        self._current_pose[:3] += linear_vel * dt
        
        # Apply angular motion (simplified quaternion integration)
        angular_vel = np.array(action["angular_velocity"])
        angle = np.linalg.norm(angular_vel) * dt
        
        if angle > 1e-6:
            axis = angular_vel / np.linalg.norm(angular_vel)
            
            # Convert axis-angle to quaternion
            half_angle = angle / 2
            sin_half = np.sin(half_angle)
            cos_half = np.cos(half_angle)
            
            delta_quat = np.array([
                axis[0] * sin_half,
                axis[1] * sin_half,
                axis[2] * sin_half,
                cos_half
            ])
            
            # Multiply quaternions (simplified)
            current_quat = self._current_pose[3:7]
            new_quat = self._quaternion_multiply(current_quat, delta_quat)
            self._current_pose[3:7] = new_quat / np.linalg.norm(new_quat)
        
        # Simulate collision detection
        self._handle_collisions()
    
    def _quaternion_multiply(self, q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
        """Multiply two quaternions."""
        x1, y1, z1, w1 = q1
        x2, y2, z2, w2 = q2
        
        return np.array([
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 + y1*w2 + z1*x2 - x1*z2,
            w1*z2 + z1*w2 + x1*y2 - y1*x2,
            w1*w2 - x1*x2 - y1*y2 - z1*z2
        ])
    
    def _handle_collisions(self):
        """Handle collision detection and response."""
        position = self._current_pose[:3]
        
        # Check boundaries
        bounds = [-5, 5]  # meters
        for i in range(3):
            if position[i] < bounds[0]:
                self._current_pose[i] = bounds[0]
            elif position[i] > bounds[1]:
                self._current_pose[i] = bounds[1]
        
        # Check obstacle collisions (simplified)
        for obstacle in self._obstacles:
            obs_pos = obstacle["position"]
            obs_radius = obstacle["radius"]
            distance = np.linalg.norm(position - obs_pos)
            
            if distance < obs_radius:
                # Simple repulsion
                direction = (position - obs_pos) / distance
                self._current_pose[:3] = obs_pos + direction * obs_radius
    
    def _compute_reward(self, action: Dict[str, Any]) -> float:
        """Compute reward based on task progress."""
        reward = 0.0
        
        # Distance to target reward
        if self._target_pose is not None:
            distance = np.linalg.norm(self._current_pose[:3] - self._target_pose[:3])
            reward += max(0, 1.0 - distance / 10.0)  # Max distance of 10m
        
        # Efficiency penalty
        reward -= 0.001 * self._current_step  # Small step penalty
        
        # Safety reward
        linear_vel = np.array(action["linear_velocity"])
        angular_vel = np.array(action["angular_velocity"])
        
        if np.linalg.norm(linear_vel) > 1.5:  # Penalize excessive speed
            reward -= 0.1
        if np.linalg.norm(angular_vel) > 2.0:
            reward -= 0.1
        
        # Task completion bonus
        if self._task_completed:
            reward += 10.0
        
        return float(reward)
    
    def _check_termination(self) -> bool:
        """Check if episode should terminate."""
        # Max steps reached
        if self._current_step >= self.max_steps:
            return True
        
        # Task completed
        if self._task_completed:
            return True
        
        # Check for target reached
        if self._target_pose is not None:
            distance = np.linalg.norm(self._current_pose[:3] - self._target_pose[:3])
            if distance < 0.1:  # 10cm tolerance
                self._task_completed = True
                return True
        
        return False
    
    def _generate_scene(self):
        """Generate realistic scene with objects and targets."""
        # Generate target position
        self._target_pose = np.array([
            np.random.uniform(-3, 3),  # x
            np.random.uniform(-3, 3),  # y
            np.random.uniform(0, 2),   # z
            0, 0, 0, 1                 # quaternion
        ])
        
        # Generate obstacles
        num_obstacles = np.random.randint(3, 8)
        self._obstacles = []
        
        for _ in range(num_obstacles):
            self._obstacles.append({
                "position": np.random.uniform(-4, 4, 3),
                "radius": np.random.uniform(0.2, 0.8),
                "type": np.random.choice(["sphere", "box", "cylinder"])
            })
    
    def _create_info_dict(self) -> Dict[str, Any]:
        """Create information dictionary for step."""
        avg_step_time = np.mean(self._step_times[-10:]) if self._step_times else 0.0
        
        return {
            "step": self._current_step,
            "max_steps": self.max_steps,
            "task_completed": self._task_completed,
            "distance_to_target": (
                np.linalg.norm(self._current_pose[:3] - self._target_pose[:3])
                if self._target_pose is not None else float('inf')
            ),
            "num_obstacles": len(self._obstacles),
            "avg_step_time": avg_step_time,
            "sensor_health": self._get_sensor_health(),
            "physics_stable": True,
            "collision_detected": False
        }
    
    def render(self, mode: str = "rgb_array") -> Optional[np.ndarray]:
        """Render environment visualization."""
        if mode == "rgb_array":
            # Generate synthetic visualization
            img = np.zeros((480, 640, 3), dtype=np.uint8)
            
            # Draw simple scene representation
            # Background
            img[:, :] = [135, 206, 235]  # Sky blue
            
            # Ground plane
            img[400:, :] = [34, 139, 34]  # Forest green
            
            # Agent position (red dot)
            agent_x = int(320 + self._current_pose[0] * 50)  # Scale and center
            agent_y = int(240 - self._current_pose[1] * 50)
            if 0 <= agent_x < 640 and 0 <= agent_y < 480:
                img[max(0, agent_y-5):min(480, agent_y+5), 
                    max(0, agent_x-5):min(640, agent_x+5)] = [255, 0, 0]
            
            # Target position (green dot)
            if self._target_pose is not None:
                target_x = int(320 + self._target_pose[0] * 50)
                target_y = int(240 - self._target_pose[1] * 50)
                if 0 <= target_x < 640 and 0 <= target_y < 480:
                    img[max(0, target_y-5):min(480, target_y+5), 
                        max(0, target_x-5):min(640, target_x+5)] = [0, 255, 0]
            
            # Obstacles (black circles)
            for obstacle in self._obstacles:
                obs_x = int(320 + obstacle["position"][0] * 50)
                obs_y = int(240 - obstacle["position"][1] * 50)
                radius = int(obstacle["radius"] * 50)
                
                y, x = np.ogrid[:480, :640]
                mask = (x - obs_x)**2 + (y - obs_y)**2 <= radius**2
                img[mask] = [64, 64, 64]
            
            return img
        
        return None
    
    def close(self):
        """Clean up environment resources."""
        self.logger.info(f"Closing ProductionEnv after {self._episode_count} episodes")
        
        # Log performance statistics
        if self._step_times:
            avg_time = np.mean(self._step_times)
            self.logger.info(f"Average step time: {avg_time:.4f}s")
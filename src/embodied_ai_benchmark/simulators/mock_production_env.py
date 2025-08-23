"""Mock production environment for validation without external dependencies."""

import time
import random
import logging
from typing import Any, Dict, Optional, Tuple
from ..core.base_env import BaseEnv


class MockProductionEnv(BaseEnv):
    """Mock production environment for validation."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize mock production environment."""
        super().__init__(config)
        self.logger = logging.getLogger(__name__)
        
        # Environment parameters
        self.max_steps = config.get("max_steps", 1000)
        self.sensor_noise_level = config.get("sensor_noise_level", 0.05)
        
        # State tracking
        self._current_step = 0
        self._current_pose = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]  # x,y,z,qx,qy,qz,qw
        self._target_pose = None
        self._task_completed = False
        
        self.logger.info(f"MockProductionEnv initialized with config: {config}")
    
    def reset(self, seed: Optional[int] = None) -> Dict[str, Any]:
        """Reset environment to initial state."""
        if seed is not None:
            random.seed(seed)
            
        self._current_step = 0
        self._task_completed = False
        self._current_pose = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
        
        # Generate target
        self._target_pose = [
            random.uniform(-3, 3),  # x
            random.uniform(-3, 3),  # y
            random.uniform(0, 2),   # z
            0, 0, 0, 1             # quaternion
        ]
        
        return super().reset(seed)
    
    def _execute_step(self, action: Dict[str, Any]) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        """Execute environment step."""
        self._current_step += 1
        
        # Validate action
        self._validate_action(action)
        
        # Apply action
        self._apply_action(action)
        
        # Compute reward
        reward = self._compute_reward(action)
        
        # Check termination
        done = self._check_termination()
        
        # Get observation
        obs = self._get_observation()
        
        # Create info
        info = {
            "step": self._current_step,
            "max_steps": self.max_steps,
            "task_completed": self._task_completed,
            "distance_to_target": self._get_distance_to_target(),
        }
        
        return obs, reward, done, info
    
    def _validate_action(self, action: Dict[str, Any]):
        """Validate action parameters."""
        required_keys = ["linear_velocity", "angular_velocity"]
        
        for key in required_keys:
            if key not in action:
                raise ValueError(f"Missing required action key: {key}")
        
        linear_vel = action["linear_velocity"]
        angular_vel = action["angular_velocity"]
        
        if not isinstance(linear_vel, (list, tuple)) or len(linear_vel) != 3:
            raise ValueError(f"linear_velocity must be 3D array, got {linear_vel}")
        
        if not isinstance(angular_vel, (list, tuple)) or len(angular_vel) != 3:
            raise ValueError(f"angular_velocity must be 3D array, got {angular_vel}")
        
        # Safety limits
        max_linear_vel = 2.0
        max_angular_vel = 3.14
        
        linear_norm = sum(x*x for x in linear_vel)**0.5
        angular_norm = sum(x*x for x in angular_vel)**0.5
        
        if linear_norm > max_linear_vel:
            raise ValueError(f"Linear velocity exceeds safety limit: {linear_norm}")
        
        if angular_norm > max_angular_vel:
            raise ValueError(f"Angular velocity exceeds safety limit: {angular_norm}")
    
    def _apply_action(self, action: Dict[str, Any]):
        """Apply action with simple physics."""
        dt = 0.05  # 20 Hz simulation
        
        # Apply linear motion
        linear_vel = action["linear_velocity"]
        for i in range(3):
            self._current_pose[i] += linear_vel[i] * dt
        
        # Simple bounds checking
        for i in range(3):
            self._current_pose[i] = max(-5, min(5, self._current_pose[i]))
    
    def _compute_reward(self, action: Dict[str, Any]) -> float:
        """Compute reward based on task progress."""
        reward = 0.0
        
        # Distance to target reward
        if self._target_pose is not None:
            distance = self._get_distance_to_target()
            reward += max(0, 1.0 - distance / 10.0)
        
        # Step penalty
        reward -= 0.001
        
        # Task completion bonus
        if self._task_completed:
            reward += 10.0
        
        return reward
    
    def _get_distance_to_target(self) -> float:
        """Get distance to target."""
        if self._target_pose is None:
            return float('inf')
        
        dx = self._current_pose[0] - self._target_pose[0]
        dy = self._current_pose[1] - self._target_pose[1]
        dz = self._current_pose[2] - self._target_pose[2]
        
        return (dx*dx + dy*dy + dz*dz)**0.5
    
    def _check_termination(self) -> bool:
        """Check if episode should terminate."""
        # Max steps
        if self._current_step >= self.max_steps:
            return True
        
        # Task completed
        if self._task_completed:
            return True
        
        # Target reached
        if self._get_distance_to_target() < 0.1:
            self._task_completed = True
            return True
        
        return False
    
    def _get_observation(self) -> Dict[str, Any]:
        """Get observation."""
        # Mock RGB image (224x224x3)
        rgb = [[[random.randint(50, 200) for _ in range(3)] for _ in range(224)] for _ in range(224)]
        
        # Mock depth image (224x224)  
        depth = [[random.uniform(0.1, 10.0) for _ in range(224)] for _ in range(224)]
        
        # Add some missing data
        for i in range(224):
            for j in range(224):
                if random.random() < 0.05:  # 5% missing data
                    depth[i][j] = float('nan')
        
        # Proprioception
        proprioception = {
            "joint_positions": [random.uniform(-3.14, 3.14) for _ in range(7)],
            "joint_velocities": [random.uniform(-1, 1) for _ in range(7)],
            "joint_torques": [random.uniform(-10, 10) for _ in range(7)],
            "end_effector_force": [random.uniform(-5, 5) for _ in range(3)],
            "gripper_state": random.uniform(0, 1)
        }
        
        # Task observation
        task_obs = {
            "target_visible": random.random() > 0.3,
            "target_pose": self._target_pose if self._target_pose else [0]*7,
            "progress": min(1.0, self._current_step / self.max_steps)
        }
        
        return {
            "rgb": rgb,
            "depth": depth,
            "pose": self._current_pose.copy(),
            "proprioception": proprioception,
            "task": task_obs,
            "timestamp": time.time(),
            "step": self._current_step
        }
    
    def render(self, mode: str = "rgb_array") -> Optional[list]:
        """Render environment."""
        if mode == "rgb_array":
            # Return mock image (480x640x3)
            return [[[random.randint(0, 255) for _ in range(3)] for _ in range(640)] for _ in range(480)]
        return None
    
    def close(self):
        """Close environment."""
        self.logger.info(f"MockProductionEnv closed after {self._current_step} steps")
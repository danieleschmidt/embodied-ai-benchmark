"""Core evaluator for running benchmark episodes."""

import time
from typing import Any, Dict, List, Optional
import numpy as np

from ..core.base_env import BaseEnv
from ..core.base_agent import BaseAgent


class Evaluator:
    """Core evaluator for running benchmark episodes."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize evaluator with configuration.
        
        Args:
            config: Evaluator configuration dictionary
        """
        self.config = config
        self.timeout = config.get("timeout", 300)  # seconds
        self.safety_checks = config.get("safety_checks", True)
        self.record_trajectories = config.get("record_trajectories", True)
        self.max_force_threshold = config.get("max_force_threshold", 100.0)
        
    def run_episode(self, 
                   env: BaseEnv,
                   agent: BaseAgent,
                   max_steps: int = 1000,
                   episode_id: int = 0,
                   seed: Optional[int] = None) -> Dict[str, Any]:
        """Run a single evaluation episode.
        
        Args:
            env: Environment instance
            agent: Agent instance
            max_steps: Maximum steps in episode
            episode_id: Episode identifier
            seed: Random seed for episode
            
        Returns:
            Episode results dictionary
        """
        start_time = time.time()
        
        # Reset environment and agent
        if seed is not None:
            env.seed(seed + episode_id)
        
        observation = env.reset(seed=seed)
        agent.reset()
        
        # Episode tracking
        step_data = []
        total_reward = 0.0
        total_steps = 0
        success = False
        safety_violations = []
        
        try:
            for step in range(max_steps):
                # Check timeout
                elapsed_time = time.time() - start_time
                if elapsed_time > self.timeout:
                    break
                
                # Agent selects action
                action = agent.act(observation)
                
                # Environment step
                next_observation, reward, done, info = env.step(action)
                
                # Safety checks
                if self.safety_checks:
                    violations = self._check_safety(action, info)
                    safety_violations.extend(violations)
                
                # Update agent
                agent.update(observation, action, reward, next_observation, done)
                
                # Record step data
                if self.record_trajectories:
                    step_info = {
                        "step": step,
                        "observation": self._serialize_observation(observation),
                        "action": action,
                        "reward": reward,
                        "next_observation": self._serialize_observation(next_observation),
                        "done": done,
                        "info": info,
                        "timestamp": time.time() - start_time
                    }
                    step_data.append(step_info)
                
                # Update tracking variables
                total_reward += reward
                total_steps += 1
                
                # Check for task completion
                if done:
                    success = info.get("task_success", False)
                    break
                
                observation = next_observation
            
        except Exception as e:
            # Handle any errors gracefully
            info = {"error": str(e), "step": total_steps}
            step_data.append({
                "step": total_steps,
                "error": str(e),
                "timestamp": time.time() - start_time
            })
        
        total_time = time.time() - start_time
        
        # Compile episode results
        episode_result = {
            "episode_id": episode_id,
            "total_steps": total_steps,
            "total_reward": total_reward,
            "success": success,
            "total_time": total_time,
            "avg_reward_per_step": total_reward / max(1, total_steps),
            "steps_per_second": total_steps / max(0.001, total_time),
            "safety_violations": safety_violations,
            "steps": step_data if self.record_trajectories else [],
            "agent_performance": agent.get_performance_metrics(),
            "env_state": env.get_state()
        }
        
        return episode_result
    
    def _check_safety(self, action: Dict[str, Any], info: Dict[str, Any]) -> List[str]:
        """Check for safety violations.
        
        Args:
            action: Action taken by agent
            info: Environment info dictionary
            
        Returns:
            List of safety violation descriptions
        """
        violations = []
        
        # Check for collisions
        if info.get("collision", False):
            violations.append("collision_detected")
        
        # Check for excessive forces
        applied_force = info.get("applied_force", 0.0)
        if applied_force > self.max_force_threshold:
            violations.append(f"excessive_force_{applied_force:.1f}N")
        
        # Check for out-of-bounds actions
        action_values = action.get("values", [])
        if isinstance(action_values, (list, np.ndarray)):
            if any(abs(val) > 10.0 for val in action_values):
                violations.append("out_of_bounds_action")
        
        # Check for unstable objects
        if info.get("object_fell", False):
            violations.append("object_instability")
        
        # Check for workspace violations
        if info.get("workspace_violation", False):
            violations.append("workspace_boundary_violation")
        
        return violations
    
    def _serialize_observation(self, observation: Dict[str, Any]) -> Dict[str, Any]:
        """Serialize observation for storage.
        
        Args:
            observation: Raw observation dictionary
            
        Returns:
            Serialized observation dictionary
        """
        serialized = {}
        
        for key, value in observation.items():
            if isinstance(value, np.ndarray):
                # Store shape and type info for large arrays
                if value.size > 1000:  # Large arrays (like images)
                    serialized[key] = {
                        "type": "large_array",
                        "shape": value.shape,
                        "dtype": str(value.dtype),
                        "mean": float(np.mean(value)),
                        "std": float(np.std(value)),
                        "min": float(np.min(value)),
                        "max": float(np.max(value))
                    }
                else:
                    serialized[key] = value.tolist()
            else:
                serialized[key] = value
        
        return serialized
    
    def run_batch_episodes(self,
                          env: BaseEnv,
                          agents: List[BaseAgent],
                          num_episodes_per_agent: int = 10,
                          max_steps: int = 1000) -> Dict[str, List[Dict[str, Any]]]:
        """Run batch evaluation for multiple agents.
        
        Args:
            env: Environment instance
            agents: List of agents to evaluate
            num_episodes_per_agent: Episodes per agent
            max_steps: Maximum steps per episode
            
        Returns:
            Dictionary mapping agent IDs to episode results
        """
        batch_results = {}
        
        for agent in agents:
            agent_id = getattr(agent, 'agent_id', f"agent_{id(agent)}")
            agent_results = []
            
            for episode in range(num_episodes_per_agent):
                episode_result = self.run_episode(
                    env, agent, max_steps, 
                    episode_id=episode,
                    seed=episode * 42  # Deterministic seeding
                )
                agent_results.append(episode_result)
            
            batch_results[agent_id] = agent_results
        
        return batch_results
    
    def compare_agents(self,
                      env: BaseEnv,
                      agents: List[BaseAgent],
                      num_episodes: int = 100,
                      max_steps: int = 1000) -> Dict[str, Any]:
        """Compare multiple agents on the same environment.
        
        Args:
            env: Environment instance
            agents: List of agents to compare
            num_episodes: Number of episodes per agent
            max_steps: Maximum steps per episode
            
        Returns:
            Comparison results dictionary
        """
        comparison_results = {}
        agent_summaries = {}
        
        # Run evaluation for each agent
        for agent in agents:
            agent_id = getattr(agent, 'agent_id', f"agent_{id(agent)}")
            
            agent_episodes = []
            for episode in range(num_episodes):
                episode_result = self.run_episode(
                    env, agent, max_steps, episode_id=episode
                )
                agent_episodes.append(episode_result)
            
            # Compute agent summary statistics
            success_rate = np.mean([ep["success"] for ep in agent_episodes])
            avg_reward = np.mean([ep["total_reward"] for ep in agent_episodes])
            avg_steps = np.mean([ep["total_steps"] for ep in agent_episodes])
            avg_time = np.mean([ep["total_time"] for ep in agent_episodes])
            safety_violation_rate = np.mean([
                len(ep["safety_violations"]) > 0 for ep in agent_episodes
            ])
            
            agent_summaries[agent_id] = {
                "success_rate": success_rate,
                "avg_reward": avg_reward,
                "avg_steps": avg_steps,
                "avg_time": avg_time,
                "safety_violation_rate": safety_violation_rate,
                "episodes": agent_episodes
            }
        
        # Rank agents by performance
        agent_rankings = sorted(
            agent_summaries.items(),
            key=lambda x: (x[1]["success_rate"], x[1]["avg_reward"]),
            reverse=True
        )
        
        comparison_results = {
            "agent_summaries": agent_summaries,
            "rankings": [{"agent_id": aid, "rank": i+1, "metrics": metrics} 
                        for i, (aid, metrics) in enumerate(agent_rankings)],
            "num_episodes_per_agent": num_episodes,
            "total_episodes": num_episodes * len(agents)
        }
        
        return comparison_results
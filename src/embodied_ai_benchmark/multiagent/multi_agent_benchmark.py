"""Multi-agent benchmark coordination system."""

import time
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed

from ..core.base_env import BaseEnv
from ..core.base_agent import BaseAgent
from ..core.base_metric import BaseMetric, CollaborationMetric
from ..evaluation.evaluator import Evaluator


class MultiAgentBenchmark:
    """Benchmark system for multi-agent cooperative tasks."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize multi-agent benchmark.
        
        Args:
            config: Benchmark configuration dictionary
        """
        self.config = config or {}
        self.evaluator = Evaluator(self.config.get("evaluator", {}))
        self.communication_protocol = CommunicationProtocol(
            self.config.get("communication", {})
        )
        self.role_manager = RoleManager(self.config.get("roles", {}))
        self._results_history = []
        
    def evaluate(self,
                 env: BaseEnv,
                 agents: Dict[str, BaseAgent],
                 num_episodes: int = 50,
                 metrics: List[str] = None,
                 seed: Optional[int] = None) -> Dict[str, Any]:
        """Evaluate multi-agent system.
        
        Args:
            env: Multi-agent environment
            agents: Dictionary of agent_id to agent instances
            num_episodes: Number of evaluation episodes
            metrics: List of metrics to compute
            seed: Random seed for reproducibility
            
        Returns:
            Multi-agent evaluation results
        """
        if seed is not None:
            np.random.seed(seed)
        
        if metrics is None:
            metrics = ['success', 'coordination', 'efficiency', 'communication']
        
        start_time = time.time()
        episode_results = []
        
        for episode in range(num_episodes):
            episode_result = self._run_multi_agent_episode(
                env, agents, episode_id=episode, metrics=metrics
            )
            episode_results.append(episode_result)
        
        total_time = time.time() - start_time
        
        # Aggregate results across episodes
        aggregated_results = self._aggregate_multi_agent_results(
            episode_results, total_time, metrics
        )
        
        self._results_history.append(aggregated_results)
        return aggregated_results
    
    def _run_multi_agent_episode(self,
                                env: BaseEnv,
                                agents: Dict[str, BaseAgent],
                                episode_id: int,
                                metrics: List[str],
                                max_steps: int = 1000) -> Dict[str, Any]:
        """Run single multi-agent episode.
        
        Args:
            env: Environment instance
            agents: Dictionary of agents
            episode_id: Episode identifier
            metrics: Metrics to track
            max_steps: Maximum episode steps
            
        Returns:
            Episode results dictionary
        """
        start_time = time.time()
        
        # Reset environment and agents
        observation = env.reset(seed=episode_id)
        for agent in agents.values():
            agent.reset()
        
        # Initialize role assignments
        self.role_manager.assign_roles(agents, env.get_task_requirements())
        
        # Episode tracking
        step_data = []
        total_reward = {agent_id: 0.0 for agent_id in agents.keys()}
        collaboration_events = []
        communication_log = []
        
        for step in range(max_steps):
            step_start = time.time()
            
            # Get actions from all agents
            actions = {}
            for agent_id, agent in agents.items():
                agent_obs = self._extract_agent_observation(observation, agent_id)
                action = agent.act(agent_obs)
                actions[agent_id] = action
            
            # Process communication
            messages = self.communication_protocol.process_communication(
                agents, actions, step
            )
            communication_log.extend(messages)
            
            # Execute joint action in environment
            joint_action = self._combine_actions(actions)
            next_observation, rewards, done, info = env.step(joint_action)
            
            # Update agents with experience
            for agent_id, agent in agents.items():
                agent_obs = self._extract_agent_observation(observation, agent_id)
                agent_next_obs = self._extract_agent_observation(next_observation, agent_id)
                agent_reward = rewards.get(agent_id, 0.0)
                
                agent.update(
                    agent_obs, actions[agent_id], agent_reward, 
                    agent_next_obs, done
                )
                
                total_reward[agent_id] += agent_reward
            
            # Track collaboration events
            if info.get("coordination_event", False):
                collaboration_events.append({
                    "step": step,
                    "type": info.get("coordination_type", "unknown"),
                    "agents": info.get("coordinating_agents", list(agents.keys()))
                })
            
            # Record step data
            step_data.append({
                "step": step,
                "actions": actions,
                "rewards": rewards,
                "joint_reward": sum(rewards.values()),
                "collaboration": info.get("coordination_event", False),
                "messages": len(messages),
                "step_time": time.time() - step_start
            })
            
            if done:
                break
            
            observation = next_observation
        
        total_time = time.time() - start_time
        
        # Compute episode metrics
        episode_metrics = self._compute_episode_metrics(
            step_data, collaboration_events, communication_log, metrics
        )
        
        return {
            "episode_id": episode_id,
            "total_steps": len(step_data),
            "total_time": total_time,
            "total_rewards": total_reward,
            "joint_reward": sum(total_reward.values()),
            "success": info.get("task_success", False),
            "collaboration_events": collaboration_events,
            "communication_log": communication_log,
            "metrics": episode_metrics,
            "steps": step_data
        }
    
    def _extract_agent_observation(self, 
                                  observation: Dict[str, Any], 
                                  agent_id: str) -> Dict[str, Any]:
        """Extract agent-specific observation from joint observation.
        
        Args:
            observation: Joint observation dictionary
            agent_id: Agent identifier
            
        Returns:
            Agent-specific observation
        """
        if f"agent_{agent_id}" in observation:
            return observation[f"agent_{agent_id}"]
        
        # Default: give each agent the full observation
        agent_obs = observation.copy()
        agent_obs["agent_id"] = agent_id
        return agent_obs
    
    def _combine_actions(self, actions: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Combine individual agent actions into joint action.
        
        Args:
            actions: Dictionary of agent_id to action
            
        Returns:
            Combined joint action dictionary
        """
        return {
            "type": "multi_agent",
            "agents": actions
        }
    
    def _compute_episode_metrics(self,
                               step_data: List[Dict[str, Any]],
                               collaboration_events: List[Dict[str, Any]],
                               communication_log: List[Dict[str, Any]],
                               metrics: List[str]) -> Dict[str, float]:
        """Compute metrics for multi-agent episode.
        
        Args:
            step_data: List of step data
            collaboration_events: List of collaboration events
            communication_log: List of communication messages
            metrics: Metrics to compute
            
        Returns:
            Dictionary of computed metrics
        """
        computed_metrics = {}
        
        if 'success' in metrics:
            # Success rate based on task completion
            success_steps = sum(1 for step in step_data if step.get("success", False))
            computed_metrics['success'] = float(success_steps > 0)
        
        if 'coordination' in metrics:
            # Coordination quality based on collaboration events
            total_steps = len(step_data)
            coordination_steps = len(collaboration_events)
            computed_metrics['coordination'] = coordination_steps / max(1, total_steps)
        
        if 'efficiency' in metrics:
            # Efficiency based on reward per step
            total_steps = len(step_data)
            total_joint_reward = sum(step["joint_reward"] for step in step_data)
            computed_metrics['efficiency'] = total_joint_reward / max(1, total_steps)
        
        if 'communication' in metrics:
            # Communication efficiency
            total_messages = len(communication_log)
            successful_messages = sum(
                1 for msg in communication_log 
                if msg.get("successful", True)
            )
            if total_messages > 0:
                computed_metrics['communication'] = successful_messages / total_messages
            else:
                computed_metrics['communication'] = 1.0
        
        return computed_metrics
    
    def _aggregate_multi_agent_results(self,
                                     episode_results: List[Dict[str, Any]],
                                     total_time: float,
                                     metrics: List[str]) -> Dict[str, Any]:
        """Aggregate results across multiple episodes.
        
        Args:
            episode_results: List of episode results
            total_time: Total evaluation time
            metrics: Computed metrics
            
        Returns:
            Aggregated results dictionary
        """
        if not episode_results:
            return {"error": "No episode results to aggregate"}
        
        # Aggregate basic statistics
        num_episodes = len(episode_results)
        success_rate = sum(ep["success"] for ep in episode_results) / num_episodes
        avg_steps = np.mean([ep["total_steps"] for ep in episode_results])
        avg_joint_reward = np.mean([ep["joint_reward"] for ep in episode_results])
        
        # Aggregate metrics
        aggregated_metrics = {}
        for metric in metrics:
            metric_values = [ep["metrics"][metric] for ep in episode_results]
            aggregated_metrics[metric] = {
                "mean": np.mean(metric_values),
                "std": np.std(metric_values),
                "min": np.min(metric_values),
                "max": np.max(metric_values)
            }
        
        # Collaboration analysis
        total_collaboration_events = sum(
            len(ep["collaboration_events"]) for ep in episode_results
        )
        avg_collaboration_per_episode = total_collaboration_events / num_episodes
        
        # Communication analysis
        total_messages = sum(
            len(ep["communication_log"]) for ep in episode_results
        )
        avg_messages_per_episode = total_messages / num_episodes
        
        return {
            "num_episodes": num_episodes,
            "total_time": total_time,
            "success_rate": success_rate,
            "avg_steps": avg_steps,
            "avg_joint_reward": avg_joint_reward,
            "avg_collaboration_events": avg_collaboration_per_episode,
            "avg_messages": avg_messages_per_episode,
            "metrics": aggregated_metrics,
            "episodes": episode_results
        }
    
    def analyze_cooperation(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze cooperation quality from benchmark results.
        
        Args:
            results: Benchmark results dictionary
            
        Returns:
            Cooperation analysis results
        """
        episodes = results.get("episodes", [])
        if not episodes:
            return {"error": "No episode data for analysis"}
        
        # Analyze communication patterns
        communication_analysis = self._analyze_communication_patterns(episodes)
        
        # Analyze role dynamics
        role_analysis = self._analyze_role_dynamics(episodes)
        
        # Analyze coordination efficiency
        coordination_analysis = self._analyze_coordination_efficiency(episodes)
        
        return {
            "communication": communication_analysis,
            "roles": role_analysis,
            "coordination": coordination_analysis,
            "overall_cooperation_score": self._compute_cooperation_score(
                communication_analysis, role_analysis, coordination_analysis
            )
        }
    
    def _analyze_communication_patterns(self, episodes: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze communication patterns across episodes."""
        total_messages = 0
        message_types = {}
        successful_communications = 0
        
        for episode in episodes:
            comm_log = episode.get("communication_log", [])
            total_messages += len(comm_log)
            
            for message in comm_log:
                msg_type = message.get("type", "unknown")
                message_types[msg_type] = message_types.get(msg_type, 0) + 1
                
                if message.get("successful", True):
                    successful_communications += 1
        
        communication_efficiency = (
            successful_communications / total_messages if total_messages > 0 else 1.0
        )
        
        return {
            "total_messages": total_messages,
            "message_types": message_types,
            "communication_efficiency": communication_efficiency,
            "avg_messages_per_episode": total_messages / len(episodes)
        }
    
    def _analyze_role_dynamics(self, episodes: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze role assignment and switching dynamics."""
        # Simplified role analysis
        role_switches = 0
        role_efficiency = []
        
        for episode in episodes:
            # Estimate role efficiency based on collaboration events
            collab_events = episode.get("collaboration_events", [])
            steps = episode.get("total_steps", 1)
            efficiency = len(collab_events) / steps
            role_efficiency.append(efficiency)
        
        return {
            "role_switches": role_switches,
            "avg_role_efficiency": np.mean(role_efficiency) if role_efficiency else 0.0,
            "role_stability": 1.0 - (role_switches / sum(ep["total_steps"] for ep in episodes))
        }
    
    def _analyze_coordination_efficiency(self, episodes: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze coordination efficiency across episodes."""
        coordination_scores = []
        
        for episode in episodes:
            collab_events = len(episode.get("collaboration_events", []))
            total_steps = episode.get("total_steps", 1)
            joint_reward = episode.get("joint_reward", 0)
            
            # Coordination score based on collaboration frequency and reward
            coord_score = (collab_events / total_steps) * (joint_reward / 100)
            coordination_scores.append(coord_score)
        
        return {
            "avg_coordination_score": np.mean(coordination_scores),
            "coordination_consistency": 1.0 - np.std(coordination_scores),
            "coordination_scores": coordination_scores
        }
    
    def _compute_cooperation_score(self,
                                 communication_analysis: Dict[str, Any],
                                 role_analysis: Dict[str, Any], 
                                 coordination_analysis: Dict[str, Any]) -> float:
        """Compute overall cooperation score."""
        comm_score = communication_analysis.get("communication_efficiency", 0.0)
        role_score = role_analysis.get("avg_role_efficiency", 0.0)
        coord_score = coordination_analysis.get("avg_coordination_score", 0.0)
        
        # Weighted combination
        cooperation_score = (0.4 * comm_score + 0.3 * role_score + 0.3 * coord_score)
        return float(np.clip(cooperation_score, 0.0, 1.0))


class CommunicationProtocol:
    """Handles multi-agent communication."""
    
    def __init__(self, config: Dict[str, Any]):
        self.bandwidth_limit = config.get("bandwidth_limit", 100)
        self.latency = config.get("latency", 0.1)
        self.packet_loss = config.get("packet_loss", 0.01)
        self.message_queue = []
        
    def process_communication(self,
                            agents: Dict[str, BaseAgent],
                            actions: Dict[str, Dict[str, Any]],
                            step: int) -> List[Dict[str, Any]]:
        """Process communication between agents."""
        messages = []
        
        # Extract communication actions
        for agent_id, action in actions.items():
            if action.get("type") == "communicate":
                message = {
                    "sender": agent_id,
                    "recipient": action.get("recipient", "broadcast"),
                    "content": action.get("message", {}),
                    "step": step,
                    "successful": np.random.random() > self.packet_loss
                }
                messages.append(message)
                
                # Deliver message to recipient(s)
                if message["successful"]:
                    self._deliver_message(agents, message)
        
        return messages
    
    def _deliver_message(self, agents: Dict[str, BaseAgent], message: Dict[str, Any]):
        """Deliver message to recipient agent(s)."""
        recipient = message["recipient"]
        
        if recipient == "broadcast":
            # Broadcast to all other agents
            for agent_id, agent in agents.items():
                if agent_id != message["sender"]:
                    agent.receive_message(message["sender"], message["content"])
        elif recipient in agents:
            # Direct message
            agents[recipient].receive_message(message["sender"], message["content"])


class RoleManager:
    """Manages dynamic role assignment for multi-agent teams."""
    
    def __init__(self, config: Dict[str, Any]):
        self.roles = config.get("available_roles", ["leader", "follower", "scout"])
        self.assignment_method = config.get("assignment_method", "capability_based")
        self.current_assignments = {}
        
    def assign_roles(self,
                    agents: Dict[str, BaseAgent],
                    task_requirements: Optional[Dict[str, Any]] = None) -> Dict[str, str]:
        """Assign roles to agents."""
        if self.assignment_method == "capability_based":
            return self._assign_by_capabilities(agents, task_requirements)
        elif self.assignment_method == "random":
            return self._assign_randomly(agents)
        else:
            return self._assign_fixed(agents)
    
    def _assign_by_capabilities(self,
                              agents: Dict[str, BaseAgent],
                              task_requirements: Optional[Dict[str, Any]]) -> Dict[str, str]:
        """Assign roles based on agent capabilities."""
        assignments = {}
        
        # Simple heuristic: first agent is leader, others are followers
        agent_list = list(agents.keys())
        
        for i, agent_id in enumerate(agent_list):
            if i == 0:
                assignments[agent_id] = "leader"
            elif i < len(agent_list) // 2:
                assignments[agent_id] = "scout"
            else:
                assignments[agent_id] = "follower"
        
        self.current_assignments = assignments
        return assignments
    
    def _assign_randomly(self, agents: Dict[str, BaseAgent]) -> Dict[str, str]:
        """Assign roles randomly."""
        assignments = {}
        
        for agent_id in agents.keys():
            role = np.random.choice(self.roles)
            assignments[agent_id] = role
        
        self.current_assignments = assignments
        return assignments
    
    def _assign_fixed(self, agents: Dict[str, BaseAgent]) -> Dict[str, str]:
        """Assign fixed roles."""
        assignments = {}
        
        for i, agent_id in enumerate(agents.keys()):
            role_index = i % len(self.roles)
            assignments[agent_id] = self.roles[role_index]
        
        self.current_assignments = assignments
        return assignments
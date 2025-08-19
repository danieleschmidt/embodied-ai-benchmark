"""Emergent Swarm Coordination for Multi-Agent Embodied AI."""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass, field
import time
import networkx as nx
from concurrent.futures import ThreadPoolExecutor
import math

from ..utils.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class SwarmConfig:
    """Configuration for emergent swarm coordination."""
    max_agents: int = 20
    communication_range: float = 5.0
    coordination_dim: int = 64
    emergence_layers: int = 4
    consensus_threshold: float = 0.8
    adaptation_rate: float = 0.01
    topology_update_freq: int = 10
    pheromone_decay: float = 0.1
    local_influence_radius: float = 2.0
    global_influence_weight: float = 0.3
    emergence_temperature: float = 1.0
    coordination_memory: int = 50
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class AgentState:
    """State representation for individual agent."""
    agent_id: int
    position: np.ndarray
    velocity: np.ndarray
    local_observation: torch.Tensor
    coordination_state: torch.Tensor
    communication_history: List[Dict[str, Any]] = field(default_factory=list)
    behavioral_signature: Optional[torch.Tensor] = None
    role_assignment: Optional[str] = None
    trust_scores: Dict[int, float] = field(default_factory=dict)


class EmergentCommunicationProtocol:
    """Self-organizing communication protocol for swarm coordination."""
    
    def __init__(self, config: SwarmConfig):
        self.config = config
        self.message_vocabulary = {}
        self.message_semantics = {}
        self.usage_statistics = {}
        self.emergence_history = []
        
        # Initialize basic communication primitives
        self._initialize_primitives()
        
    def _initialize_primitives(self):
        """Initialize basic communication primitives."""
        self.message_vocabulary = {
            'position_update': 0,
            'goal_sharing': 1,
            'task_request': 2,
            'resource_available': 3,
            'danger_warning': 4,
            'coordination_sync': 5,
            'role_negotiation': 6,
            'consensus_vote': 7
        }
        
        # Initialize semantic embeddings
        vocab_size = len(self.message_vocabulary)
        self.message_semantics = torch.randn(vocab_size, self.config.coordination_dim)
        
    def encode_message(self, message_type: str, content: Dict[str, Any]) -> torch.Tensor:
        """Encode message into emergent communication format."""
        if message_type not in self.message_vocabulary:
            # Emergent new message type
            message_type = self._create_emergent_message_type(content)
        
        # Get base semantic embedding
        msg_id = self.message_vocabulary[message_type]
        base_embedding = self.message_semantics[msg_id].clone()
        
        # Modulate embedding based on content
        content_modulation = self._encode_content(content)
        
        # Combine base semantics with content
        message_embedding = base_embedding + 0.1 * content_modulation
        
        # Track usage
        self._update_usage_statistics(message_type, content)
        
        return message_embedding
    
    def decode_message(self, message_embedding: torch.Tensor) -> Tuple[str, Dict[str, Any]]:
        """Decode emergent communication message."""
        # Find closest semantic match
        similarities = F.cosine_similarity(
            message_embedding.unsqueeze(0), 
            self.message_semantics, 
            dim=1
        )
        
        best_match_idx = torch.argmax(similarities).item()
        confidence = similarities[best_match_idx].item()
        
        # Get message type
        message_type = list(self.message_vocabulary.keys())[best_match_idx]
        
        # Extract content (simplified)
        content = self._decode_content(message_embedding, message_type)
        content['confidence'] = confidence
        
        return message_type, content
    
    def _create_emergent_message_type(self, content: Dict[str, Any]) -> str:
        """Create new emergent message type based on content patterns."""
        # Analyze content to determine new message type
        content_keys = set(content.keys())
        
        # Check for common patterns
        if 'position' in content_keys and 'velocity' in content_keys:
            new_type = 'movement_coordination'
        elif 'task_id' in content_keys and 'priority' in content_keys:
            new_type = 'task_coordination'
        elif 'resource_type' in content_keys and 'quantity' in content_keys:
            new_type = 'resource_sharing'
        else:
            new_type = f'emergent_type_{len(self.message_vocabulary)}'
        
        # Add to vocabulary
        new_id = len(self.message_vocabulary)
        self.message_vocabulary[new_type] = new_id
        
        # Create semantic embedding
        new_embedding = torch.randn(self.config.coordination_dim)
        self.message_semantics = torch.cat([
            self.message_semantics,
            new_embedding.unsqueeze(0)
        ], dim=0)
        
        logger.info(f"Emergent message type created: {new_type}")
        
        return new_type
    
    def _encode_content(self, content: Dict[str, Any]) -> torch.Tensor:
        """Encode message content into vector representation."""
        content_vector = torch.zeros(self.config.coordination_dim)
        
        # Simple content encoding (can be made more sophisticated)
        for i, (key, value) in enumerate(content.items()):
            if isinstance(value, (int, float)):
                content_vector[i % self.config.coordination_dim] = float(value)
            elif isinstance(value, str):
                # Hash string to number
                content_vector[i % self.config.coordination_dim] = hash(value) % 1000 / 1000.0
        
        return content_vector
    
    def _decode_content(self, message_embedding: torch.Tensor, message_type: str) -> Dict[str, Any]:
        """Decode content from message embedding."""
        # Simplified content extraction
        content = {
            'message_type': message_type,
            'embedding_norm': torch.norm(message_embedding).item(),
            'dominant_features': torch.topk(torch.abs(message_embedding), 3).indices.tolist()
        }
        
        return content
    
    def _update_usage_statistics(self, message_type: str, content: Dict[str, Any]):
        """Update usage statistics for message adaptation."""
        if message_type not in self.usage_statistics:
            self.usage_statistics[message_type] = {
                'count': 0,
                'success_rate': 0.5,
                'content_patterns': []
            }
        
        self.usage_statistics[message_type]['count'] += 1
        self.usage_statistics[message_type]['content_patterns'].append(list(content.keys()))
    
    def adapt_vocabulary(self, feedback: Dict[str, float]):
        """Adapt vocabulary based on communication success feedback."""
        for message_type, success_rate in feedback.items():
            if message_type in self.usage_statistics:
                stats = self.usage_statistics[message_type]
                # Update success rate with exponential moving average
                alpha = 0.1
                stats['success_rate'] = (
                    alpha * success_rate + 
                    (1 - alpha) * stats['success_rate']
                )
                
                # Adapt semantic embedding based on success
                if message_type in self.message_vocabulary:
                    msg_id = self.message_vocabulary[message_type]
                    adaptation = torch.randn_like(self.message_semantics[msg_id]) * 0.01
                    
                    if success_rate > 0.7:
                        # Reinforce successful patterns
                        self.message_semantics[msg_id] += adaptation
                    elif success_rate < 0.3:
                        # Explore new patterns for unsuccessful messages
                        self.message_semantics[msg_id] += adaptation * 3
    
    def get_emergence_metrics(self) -> Dict[str, Any]:
        """Get metrics on communication emergence."""
        return {
            'vocabulary_size': len(self.message_vocabulary),
            'avg_success_rate': np.mean([
                stats['success_rate'] for stats in self.usage_statistics.values()
            ]) if self.usage_statistics else 0,
            'message_diversity': len(set([
                tuple(pattern) for stats in self.usage_statistics.values()
                for pattern in stats['content_patterns']
            ])),
            'emergence_rate': len(self.emergence_history)
        }


class TopologyManager:
    """Manages dynamic swarm topology and network structure."""
    
    def __init__(self, config: SwarmConfig):
        self.config = config
        self.communication_graph = nx.Graph()
        self.topology_history = []
        self.centrality_cache = {}
        self.last_update = 0
        
    def update_topology(self, agent_states: List[AgentState], 
                       performance_feedback: Optional[Dict[int, float]] = None) -> nx.Graph:
        """Update swarm topology based on agent positions and performance."""
        self.last_update += 1
        
        # Clear existing edges
        self.communication_graph.clear()
        
        # Add all agents as nodes
        for agent in agent_states:
            self.communication_graph.add_node(
                agent.agent_id,
                position=agent.position,
                role=agent.role_assignment,
                performance=performance_feedback.get(agent.agent_id, 0.5) if performance_feedback else 0.5
            )
        
        # Create edges based on communication range and strategic value
        for i, agent_a in enumerate(agent_states):
            for j, agent_b in enumerate(agent_states[i+1:], i+1):
                distance = np.linalg.norm(agent_a.position - agent_b.position)
                
                if distance <= self.config.communication_range:
                    # Base connectivity
                    edge_weight = 1.0 - (distance / self.config.communication_range)
                    
                    # Enhance connectivity for complementary roles
                    if self._roles_complementary(agent_a.role_assignment, agent_b.role_assignment):
                        edge_weight *= 1.5
                    
                    # Consider trust scores
                    trust_ab = agent_a.trust_scores.get(agent_b.agent_id, 0.5)
                    trust_ba = agent_b.trust_scores.get(agent_a.agent_id, 0.5)
                    edge_weight *= (trust_ab + trust_ba) / 2
                    
                    if edge_weight > 0.3:  # Threshold for connection
                        self.communication_graph.add_edge(
                            agent_a.agent_id, 
                            agent_b.agent_id, 
                            weight=edge_weight,
                            distance=distance
                        )
        
        # Ensure connectivity through strategic bridges
        self._ensure_connectivity()
        
        # Cache centrality measures
        self._update_centrality_cache()
        
        # Store topology snapshot
        self.topology_history.append({
            'timestep': self.last_update,
            'num_edges': self.communication_graph.number_of_edges(),
            'avg_degree': np.mean([d for n, d in self.communication_graph.degree()]) if self.communication_graph.nodes() else 0,
            'clustering_coefficient': nx.average_clustering(self.communication_graph),
            'diameter': nx.diameter(self.communication_graph) if nx.is_connected(self.communication_graph) else float('inf')
        })
        
        return self.communication_graph
    
    def _roles_complementary(self, role_a: Optional[str], role_b: Optional[str]) -> bool:
        """Check if two roles are complementary."""
        if not role_a or not role_b:
            return False
        
        complementary_pairs = [
            ('leader', 'follower'),
            ('scout', 'coordinator'),
            ('resource_gatherer', 'transporter'),
            ('guard', 'worker')
        ]
        
        for pair in complementary_pairs:
            if (role_a in pair and role_b in pair) and role_a != role_b:
                return True
        
        return False
    
    def _ensure_connectivity(self):
        """Ensure graph connectivity through strategic bridges."""
        if not self.communication_graph.nodes():
            return
        
        # Find connected components
        components = list(nx.connected_components(self.communication_graph))
        
        if len(components) > 1:
            # Connect components through shortest distance bridges
            for i in range(len(components) - 1):
                comp_a = components[i]
                comp_b = components[i + 1]
                
                # Find closest pair between components
                min_distance = float('inf')
                bridge_pair = None
                
                for node_a in comp_a:
                    for node_b in comp_b:
                        pos_a = self.communication_graph.nodes[node_a]['position']
                        pos_b = self.communication_graph.nodes[node_b]['position']
                        distance = np.linalg.norm(pos_a - pos_b)
                        
                        if distance < min_distance:
                            min_distance = distance
                            bridge_pair = (node_a, node_b)
                
                # Add bridge edge
                if bridge_pair:
                    self.communication_graph.add_edge(
                        bridge_pair[0], 
                        bridge_pair[1],
                        weight=0.5,  # Lower weight for bridge connections
                        distance=min_distance,
                        is_bridge=True
                    )
    
    def _update_centrality_cache(self):
        """Update centrality measures cache."""
        if self.communication_graph.number_of_nodes() == 0:
            return
        
        self.centrality_cache = {
            'betweenness': nx.betweenness_centrality(self.communication_graph),
            'closeness': nx.closeness_centrality(self.communication_graph),
            'degree': nx.degree_centrality(self.communication_graph),
            'eigenvector': nx.eigenvector_centrality(self.communication_graph, max_iter=1000)
        }
    
    def get_agent_centrality(self, agent_id: int) -> Dict[str, float]:
        """Get centrality measures for specific agent."""
        return {
            metric: scores.get(agent_id, 0.0)
            for metric, scores in self.centrality_cache.items()
        }
    
    def identify_key_agents(self, metric: str = 'betweenness', top_k: int = 3) -> List[int]:
        """Identify key agents based on centrality metric."""
        if metric not in self.centrality_cache:
            return []
        
        sorted_agents = sorted(
            self.centrality_cache[metric].items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        return [agent_id for agent_id, _ in sorted_agents[:top_k]]
    
    def get_topology_metrics(self) -> Dict[str, Any]:
        """Get current topology metrics."""
        if not self.communication_graph.nodes():
            return {}
        
        metrics = {
            'num_nodes': self.communication_graph.number_of_nodes(),
            'num_edges': self.communication_graph.number_of_edges(),
            'density': nx.density(self.communication_graph),
            'avg_clustering': nx.average_clustering(self.communication_graph),
            'is_connected': nx.is_connected(self.communication_graph)
        }
        
        if nx.is_connected(self.communication_graph):
            metrics['diameter'] = nx.diameter(self.communication_graph)
            metrics['avg_path_length'] = nx.average_shortest_path_length(self.communication_graph)
        
        return metrics


class RoleAssignmentSystem:
    """Dynamic role assignment system for swarm agents."""
    
    def __init__(self, config: SwarmConfig):
        self.config = config
        self.available_roles = [
            'leader', 'follower', 'scout', 'coordinator', 
            'resource_gatherer', 'transporter', 'guard', 'worker'
        ]
        self.role_requirements = {}
        self.assignment_history = []
        
    def assign_roles(self, agent_states: List[AgentState], 
                    task_context: Dict[str, Any],
                    topology_graph: nx.Graph) -> Dict[int, str]:
        """Assign roles to agents based on context and network position."""
        num_agents = len(agent_states)
        
        # Determine role requirements based on task context
        role_needs = self._analyze_role_requirements(task_context, num_agents)
        
        # Score agents for each role
        role_scores = {}
        for agent in agent_states:
            agent_scores = self._score_agent_for_roles(agent, topology_graph, task_context)
            role_scores[agent.agent_id] = agent_scores
        
        # Optimal assignment using Hungarian algorithm approximation
        assignments = self._optimal_role_assignment(role_scores, role_needs)
        
        # Update agent states with new assignments
        for agent in agent_states:
            agent.role_assignment = assignments.get(agent.agent_id, 'worker')
        
        # Track assignment
        self.assignment_history.append({
            'timestep': len(self.assignment_history),
            'assignments': assignments.copy(),
            'role_distribution': self._get_role_distribution(assignments)
        })
        
        return assignments
    
    def _analyze_role_requirements(self, task_context: Dict[str, Any], num_agents: int) -> Dict[str, int]:
        """Analyze task requirements to determine needed roles."""
        task_type = task_context.get('task_type', 'cooperative')
        complexity = task_context.get('complexity', 'medium')
        
        # Base role distribution
        base_needs = {
            'leader': 1,
            'coordinator': max(1, num_agents // 6),
            'scout': max(1, num_agents // 8),
            'worker': num_agents // 2,
            'guard': num_agents // 10
        }
        
        # Adjust based on task type
        if task_type == 'exploration':
            base_needs['scout'] = max(2, num_agents // 4)
            base_needs['worker'] = num_agents // 3
        elif task_type == 'construction':
            base_needs['coordinator'] = max(2, num_agents // 4)
            base_needs['resource_gatherer'] = num_agents // 5
            base_needs['transporter'] = num_agents // 6
        elif task_type == 'search_rescue':
            base_needs['scout'] = max(3, num_agents // 3)
            base_needs['coordinator'] = max(2, num_agents // 5)
        
        # Normalize to ensure we don't exceed available agents
        total_assigned = sum(base_needs.values())
        if total_assigned > num_agents:
            scale_factor = num_agents / total_assigned
            base_needs = {role: max(1, int(count * scale_factor)) 
                         for role, count in base_needs.items()}
        
        # Fill remaining slots with workers
        remaining = num_agents - sum(base_needs.values())
        if remaining > 0:
            base_needs['worker'] = base_needs.get('worker', 0) + remaining
        
        return base_needs
    
    def _score_agent_for_roles(self, agent: AgentState, 
                              topology_graph: nx.Graph,
                              task_context: Dict[str, Any]) -> Dict[str, float]:
        """Score an agent's suitability for different roles."""
        scores = {}
        
        # Get agent's network position
        centrality = {}
        if agent.agent_id in topology_graph:
            centrality = {
                'betweenness': nx.betweenness_centrality(topology_graph).get(agent.agent_id, 0),
                'degree': nx.degree_centrality(topology_graph).get(agent.agent_id, 0),
                'closeness': nx.closeness_centrality(topology_graph).get(agent.agent_id, 0)
            }
        
        # Score for each role
        for role in self.available_roles:
            score = 0.5  # Base score
            
            if role == 'leader':
                # Leaders should have high centrality and communication history
                score += centrality.get('betweenness', 0) * 0.4
                score += centrality.get('degree', 0) * 0.3
                score += len(agent.communication_history) / 100 * 0.3
                
            elif role == 'coordinator':
                # Coordinators need good network position
                score += centrality.get('closeness', 0) * 0.5
                score += centrality.get('degree', 0) * 0.3
                
            elif role == 'scout':
                # Scouts should be mobile and have exploration history
                velocity_norm = np.linalg.norm(agent.velocity) if len(agent.velocity) > 0 else 0
                score += min(velocity_norm / 5.0, 1.0) * 0.6  # Mobility preference
                score += (1 - centrality.get('betweenness', 0)) * 0.2  # Prefer periphery
                
            elif role == 'guard':
                # Guards should be positioned strategically
                score += centrality.get('betweenness', 0) * 0.6  # Strategic positions
                
            elif role == 'worker':
                # Workers are generalists
                score = 0.6  # Neutral score for everyone
            
            # Apply behavioral signature if available
            if agent.behavioral_signature is not None:
                role_idx = self.available_roles.index(role)
                if role_idx < len(agent.behavioral_signature):
                    score += agent.behavioral_signature[role_idx].item() * 0.2
            
            scores[role] = max(0.0, min(1.0, score))
        
        return scores
    
    def _optimal_role_assignment(self, role_scores: Dict[int, Dict[str, float]], 
                                role_needs: Dict[str, int]) -> Dict[int, str]:
        """Perform optimal role assignment using greedy approximation."""
        assignments = {}
        available_agents = set(role_scores.keys())
        remaining_needs = role_needs.copy()
        
        # Greedy assignment: assign highest scoring agents to roles
        while available_agents and any(need > 0 for need in remaining_needs.values()):
            best_assignment = None
            best_score = -1
            
            for agent_id in available_agents:
                for role, need in remaining_needs.items():
                    if need > 0:
                        score = role_scores[agent_id][role]
                        if score > best_score:
                            best_score = score
                            best_assignment = (agent_id, role)
            
            if best_assignment:
                agent_id, role = best_assignment
                assignments[agent_id] = role
                available_agents.remove(agent_id)
                remaining_needs[role] -= 1
            else:
                break
        
        # Assign remaining agents as workers
        for agent_id in available_agents:
            assignments[agent_id] = 'worker'
        
        return assignments
    
    def _get_role_distribution(self, assignments: Dict[int, str]) -> Dict[str, int]:
        """Get distribution of roles in current assignment."""
        distribution = {}
        for role in assignments.values():
            distribution[role] = distribution.get(role, 0) + 1
        return distribution
    
    def get_assignment_stability(self, window_size: int = 10) -> float:
        """Compute stability of role assignments over time."""
        if len(self.assignment_history) < 2:
            return 1.0
        
        recent_assignments = self.assignment_history[-window_size:]
        
        # Compute assignment changes
        changes = 0
        comparisons = 0
        
        for i in range(1, len(recent_assignments)):
            prev_assignments = recent_assignments[i-1]['assignments']
            curr_assignments = recent_assignments[i]['assignments']
            
            for agent_id in set(prev_assignments.keys()) & set(curr_assignments.keys()):
                comparisons += 1
                if prev_assignments[agent_id] != curr_assignments[agent_id]:
                    changes += 1
        
        stability = 1.0 - (changes / max(comparisons, 1))
        return stability


class SwarmCoordinationEngine:
    """Main engine for emergent swarm coordination."""
    
    def __init__(self, config: SwarmConfig):
        self.config = config
        self.communication_protocol = EmergentCommunicationProtocol(config)
        self.topology_manager = TopologyManager(config)
        self.role_system = RoleAssignmentSystem(config)
        
        # Coordination networks
        self.coordination_network = self._build_coordination_network()
        self.consensus_mechanism = ConsensusMechanism(config)
        
        # Pheromone field for emergent coordination
        self.pheromone_field = PheromoneField(config)
        
        # Performance tracking
        self.coordination_metrics = []
        
    def _build_coordination_network(self) -> nn.Module:
        """Build neural network for coordination decision making."""
        return nn.Sequential(
            nn.Linear(self.config.coordination_dim * 3, 256),  # Self + neighbors + global
            nn.ReLU(),
            nn.LayerNorm(256),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, self.config.coordination_dim),
            nn.Tanh()
        )
    
    def coordinate_swarm(self, agent_states: List[AgentState], 
                        task_context: Dict[str, Any],
                        global_state: Optional[torch.Tensor] = None) -> Dict[str, Any]:
        """Perform one step of swarm coordination."""
        
        # Update topology
        topology = self.topology_manager.update_topology(agent_states)
        
        # Assign roles
        role_assignments = self.role_system.assign_roles(agent_states, task_context, topology)
        
        # Generate coordination decisions
        coordination_decisions = self._generate_coordination_decisions(
            agent_states, topology, global_state
        )
        
        # Update pheromone field
        self.pheromone_field.update(agent_states, coordination_decisions)
        
        # Achieve consensus on global decisions
        consensus_result = self.consensus_mechanism.reach_consensus(
            agent_states, coordination_decisions, topology
        )
        
        # Communication exchange
        communication_results = self._coordinate_communication(agent_states, topology)
        
        # Update agent coordination states
        self._update_agent_states(agent_states, coordination_decisions, consensus_result)
        
        # Compute coordination metrics
        metrics = self._compute_coordination_metrics(
            agent_states, topology, coordination_decisions, consensus_result
        )
        
        self.coordination_metrics.append(metrics)
        
        return {
            'coordination_decisions': coordination_decisions,
            'role_assignments': role_assignments,
            'topology_metrics': self.topology_manager.get_topology_metrics(),
            'consensus_result': consensus_result,
            'communication_metrics': communication_results,
            'coordination_metrics': metrics,
            'pheromone_field': self.pheromone_field.get_field_state()
        }
    
    def _generate_coordination_decisions(self, agent_states: List[AgentState],
                                       topology: nx.Graph,
                                       global_state: Optional[torch.Tensor]) -> Dict[int, torch.Tensor]:
        """Generate coordination decisions for each agent."""
        decisions = {}
        
        # Prepare global context
        if global_state is None:
            global_state = torch.zeros(self.config.coordination_dim)
        
        for agent in agent_states:
            # Gather neighbor information
            neighbor_info = self._gather_neighbor_information(agent, agent_states, topology)
            
            # Get pheromone influence
            pheromone_influence = self.pheromone_field.get_influence_at_position(agent.position)
            
            # Combine information
            agent_input = torch.cat([
                agent.coordination_state,
                neighbor_info,
                global_state,
                torch.tensor(pheromone_influence, dtype=torch.float32)
            ], dim=0)
            
            # Pad or truncate to expected size
            expected_size = self.config.coordination_dim * 3
            if len(agent_input) > expected_size:
                agent_input = agent_input[:expected_size]
            elif len(agent_input) < expected_size:
                padding = torch.zeros(expected_size - len(agent_input))
                agent_input = torch.cat([agent_input, padding])
            
            # Generate decision
            with torch.no_grad():
                decision = self.coordination_network(agent_input.unsqueeze(0)).squeeze(0)
            
            decisions[agent.agent_id] = decision
        
        return decisions
    
    def _gather_neighbor_information(self, agent: AgentState, 
                                   all_agents: List[AgentState],
                                   topology: nx.Graph) -> torch.Tensor:
        """Gather information from neighboring agents."""
        neighbor_info = torch.zeros(self.config.coordination_dim)
        
        if agent.agent_id not in topology:
            return neighbor_info
        
        neighbors = list(topology.neighbors(agent.agent_id))
        if not neighbors:
            return neighbor_info
        
        # Aggregate neighbor coordination states
        neighbor_states = []
        for neighbor_id in neighbors:
            neighbor_agent = next((a for a in all_agents if a.agent_id == neighbor_id), None)
            if neighbor_agent:
                # Weight by edge weight and trust
                edge_weight = topology[agent.agent_id][neighbor_id].get('weight', 1.0)
                trust_weight = agent.trust_scores.get(neighbor_id, 0.5)
                
                weighted_state = neighbor_agent.coordination_state * edge_weight * trust_weight
                neighbor_states.append(weighted_state)
        
        if neighbor_states:
            # Average neighbor information
            neighbor_info = torch.mean(torch.stack(neighbor_states), dim=0)
        
        return neighbor_info
    
    def _coordinate_communication(self, agent_states: List[AgentState], 
                                topology: nx.Graph) -> Dict[str, Any]:
        """Coordinate communication between agents."""
        communication_success = {}
        message_counts = {}
        
        for agent in agent_states:
            if agent.agent_id not in topology:
                continue
                
            neighbors = list(topology.neighbors(agent.agent_id))
            
            for neighbor_id in neighbors:
                # Generate and send message
                message_content = {
                    'sender_id': agent.agent_id,
                    'position': agent.position.tolist(),
                    'role': agent.role_assignment,
                    'coordination_state': agent.coordination_state.tolist()
                }
                
                try:
                    # Encode message
                    encoded_msg = self.communication_protocol.encode_message(
                        'coordination_sync', message_content
                    )
                    
                    # Simulate message transmission (with possible failure)
                    edge_weight = topology[agent.agent_id][neighbor_id].get('weight', 1.0)
                    success_prob = edge_weight * 0.9  # Base success rate
                    
                    if np.random.random() < success_prob:
                        communication_success[f"{agent.agent_id}->{neighbor_id}"] = True
                        
                        # Add to recipient's communication history
                        neighbor = next((a for a in agent_states if a.agent_id == neighbor_id), None)
                        if neighbor:
                            neighbor.communication_history.append({
                                'sender': agent.agent_id,
                                'message': encoded_msg,
                                'timestamp': time.time()
                            })
                    else:
                        communication_success[f"{agent.agent_id}->{neighbor_id}"] = False
                    
                    message_counts[agent.agent_id] = message_counts.get(agent.agent_id, 0) + 1
                    
                except Exception as e:
                    logger.warning(f"Communication failed between {agent.agent_id} and {neighbor_id}: {e}")
                    communication_success[f"{agent.agent_id}->{neighbor_id}"] = False
        
        # Update communication protocol based on success rates
        success_rates = {}
        for key, success in communication_success.items():
            agent_id = int(key.split('->')[0])
            if agent_id not in success_rates:
                success_rates[agent_id] = []
            success_rates[agent_id].append(float(success))
        
        avg_success_rates = {
            agent_id: np.mean(successes) 
            for agent_id, successes in success_rates.items()
        }
        
        # Adapt communication protocol
        self.communication_protocol.adapt_vocabulary({'coordination_sync': np.mean(list(avg_success_rates.values())) if avg_success_rates else 0.5})
        
        return {
            'total_messages': sum(message_counts.values()),
            'success_rate': np.mean(list(communication_success.values())) if communication_success else 0,
            'avg_messages_per_agent': np.mean(list(message_counts.values())) if message_counts else 0,
            'emergence_metrics': self.communication_protocol.get_emergence_metrics()
        }
    
    def _update_agent_states(self, agent_states: List[AgentState],
                           coordination_decisions: Dict[int, torch.Tensor],
                           consensus_result: Dict[str, Any]):
        """Update agent coordination states based on decisions and consensus."""
        for agent in agent_states:
            if agent.agent_id in coordination_decisions:
                # Update coordination state with decision
                decision = coordination_decisions[agent.agent_id]
                
                # Apply exponential moving average update
                alpha = self.config.adaptation_rate
                agent.coordination_state = (
                    alpha * decision + 
                    (1 - alpha) * agent.coordination_state
                )
                
                # Update behavioral signature based on recent actions
                if agent.behavioral_signature is None:
                    agent.behavioral_signature = torch.zeros(len(self.role_system.available_roles))
                
                # Decay previous signature and add current role preference
                agent.behavioral_signature *= 0.95
                if agent.role_assignment:
                    role_idx = self.role_system.available_roles.index(agent.role_assignment)
                    agent.behavioral_signature[role_idx] += 0.1
    
    def _compute_coordination_metrics(self, agent_states: List[AgentState],
                                    topology: nx.Graph,
                                    coordination_decisions: Dict[int, torch.Tensor],
                                    consensus_result: Dict[str, Any]) -> Dict[str, float]:
        """Compute metrics for coordination quality."""
        
        # Coordination alignment
        if len(coordination_decisions) > 1:
            decision_vectors = list(coordination_decisions.values())
            alignment_matrix = torch.stack([
                F.cosine_similarity(d1.unsqueeze(0), torch.stack(decision_vectors), dim=1)
                for d1 in decision_vectors
            ])
            avg_alignment = torch.mean(alignment_matrix).item()
        else:
            avg_alignment = 1.0
        
        # Role diversity
        role_counts = {}
        for agent in agent_states:
            role = agent.role_assignment or 'unassigned'
            role_counts[role] = role_counts.get(role, 0) + 1
        
        role_entropy = 0
        total_agents = len(agent_states)
        for count in role_counts.values():
            if count > 0:
                prob = count / total_agents
                role_entropy -= prob * np.log2(prob)
        
        # Communication efficiency
        topology_metrics = self.topology_manager.get_topology_metrics()
        comm_efficiency = topology_metrics.get('density', 0) * topology_metrics.get('avg_clustering', 0)
        
        # Trust network health
        avg_trust = 0
        trust_connections = 0
        for agent in agent_states:
            if agent.trust_scores:
                avg_trust += sum(agent.trust_scores.values())
                trust_connections += len(agent.trust_scores)
        
        avg_trust = avg_trust / max(trust_connections, 1)
        
        # Consensus strength
        consensus_strength = consensus_result.get('consensus_strength', 0.5)
        
        return {
            'coordination_alignment': avg_alignment,
            'role_diversity_entropy': role_entropy,
            'communication_efficiency': comm_efficiency,
            'average_trust': avg_trust,
            'consensus_strength': consensus_strength,
            'topology_connectivity': topology_metrics.get('density', 0),
            'swarm_coherence': (avg_alignment + consensus_strength + avg_trust) / 3
        }


class ConsensusMechanism:
    """Mechanism for achieving consensus in swarm decisions."""
    
    def __init__(self, config: SwarmConfig):
        self.config = config
        self.consensus_history = []
        
    def reach_consensus(self, agent_states: List[AgentState],
                       coordination_decisions: Dict[int, torch.Tensor],
                       topology: nx.Graph) -> Dict[str, Any]:
        """Reach consensus on global coordination decisions."""
        
        if not coordination_decisions:
            return {'consensus_reached': False, 'consensus_strength': 0.0}
        
        # Collect all decisions
        decision_vectors = list(coordination_decisions.values())
        agent_ids = list(coordination_decisions.keys())
        
        # Weight decisions by agent centrality
        weights = self._compute_decision_weights(agent_ids, topology)
        
        # Iterative consensus process
        consensus_vector, consensus_strength = self._iterative_consensus(
            decision_vectors, weights, agent_ids, topology
        )
        
        # Check if consensus threshold is met
        consensus_reached = consensus_strength >= self.config.consensus_threshold
        
        result = {
            'consensus_reached': consensus_reached,
            'consensus_vector': consensus_vector,
            'consensus_strength': consensus_strength,
            'participating_agents': agent_ids,
            'iterations': 5  # Fixed for now
        }
        
        self.consensus_history.append(result)
        
        return result
    
    def _compute_decision_weights(self, agent_ids: List[int], topology: nx.Graph) -> torch.Tensor:
        """Compute weights for agent decisions based on network position."""
        weights = torch.ones(len(agent_ids))
        
        if not topology.nodes():
            return weights
        
        # Get centrality measures
        centrality_measures = nx.betweenness_centrality(topology)
        
        for i, agent_id in enumerate(agent_ids):
            if agent_id in centrality_measures:
                # Weight by betweenness centrality (influence)
                weights[i] = 1.0 + centrality_measures[agent_id]
        
        # Normalize weights
        weights = weights / torch.sum(weights)
        
        return weights
    
    def _iterative_consensus(self, decision_vectors: List[torch.Tensor],
                           weights: torch.Tensor,
                           agent_ids: List[int],
                           topology: nx.Graph,
                           max_iterations: int = 5) -> Tuple[torch.Tensor, float]:
        """Iterative consensus algorithm."""
        
        current_decisions = torch.stack(decision_vectors)
        
        for iteration in range(max_iterations):
            # Compute weighted average
            weighted_average = torch.sum(current_decisions * weights.unsqueeze(-1), dim=0)
            
            # Update each agent's decision based on neighbors
            new_decisions = current_decisions.clone()
            
            for i, agent_id in enumerate(agent_ids):
                if agent_id in topology:
                    neighbors = list(topology.neighbors(agent_id))
                    neighbor_indices = [
                        j for j, aid in enumerate(agent_ids) if aid in neighbors
                    ]
                    
                    if neighbor_indices:
                        # Average with neighbors
                        neighbor_decisions = current_decisions[neighbor_indices]
                        neighbor_avg = torch.mean(neighbor_decisions, dim=0)
                        
                        # Blend with global average
                        alpha = 0.3  # Global influence
                        beta = 0.4   # Neighbor influence
                        gamma = 0.3  # Self influence
                        
                        new_decisions[i] = (
                            alpha * weighted_average +
                            beta * neighbor_avg +
                            gamma * current_decisions[i]
                        )
            
            current_decisions = new_decisions
        
        # Final consensus vector
        final_consensus = torch.sum(current_decisions * weights.unsqueeze(-1), dim=0)
        
        # Compute consensus strength (agreement level)
        similarities = F.cosine_similarity(current_decisions, final_consensus.unsqueeze(0), dim=1)
        consensus_strength = torch.mean(similarities).item()
        
        return final_consensus, consensus_strength


class PheromoneField:
    """Digital pheromone field for emergent coordination."""
    
    def __init__(self, config: SwarmConfig):
        self.config = config
        self.field_size = (100, 100)  # Grid size
        self.field_resolution = 1.0   # Meters per grid cell
        
        # Initialize pheromone layers
        self.pheromone_layers = {
            'exploration': np.zeros(self.field_size),
            'resource': np.zeros(self.field_size),
            'danger': np.zeros(self.field_size),
            'coordination': np.zeros(self.field_size)
        }
        
        self.decay_rates = {
            'exploration': 0.05,
            'resource': 0.02,
            'danger': 0.1,
            'coordination': 0.08
        }
        
    def update(self, agent_states: List[AgentState], 
              coordination_decisions: Dict[int, torch.Tensor]):
        """Update pheromone field based on agent actions."""
        
        # Decay existing pheromones
        for layer_name, decay_rate in self.decay_rates.items():
            self.pheromone_layers[layer_name] *= (1 - decay_rate)
        
        # Add new pheromones based on agent actions
        for agent in agent_states:
            grid_x, grid_y = self._position_to_grid(agent.position)
            
            if self._is_valid_grid_position(grid_x, grid_y):
                # Add exploration pheromone
                self.pheromone_layers['exploration'][grid_x, grid_y] += 0.1
                
                # Add role-specific pheromones
                if agent.role_assignment == 'scout':
                    self.pheromone_layers['exploration'][grid_x, grid_y] += 0.2
                elif agent.role_assignment == 'resource_gatherer':
                    self.pheromone_layers['resource'][grid_x, grid_y] += 0.3
                elif agent.role_assignment == 'guard':
                    self.pheromone_layers['danger'][grid_x, grid_y] += 0.15
                
                # Add coordination pheromone based on decision magnitude
                if agent.agent_id in coordination_decisions:
                    decision_magnitude = torch.norm(coordination_decisions[agent.agent_id]).item()
                    self.pheromone_layers['coordination'][grid_x, grid_y] += decision_magnitude * 0.1
        
        # Diffusion step (simple averaging with neighbors)
        self._apply_diffusion()
    
    def _position_to_grid(self, position: np.ndarray) -> Tuple[int, int]:
        """Convert world position to grid coordinates."""
        grid_x = int(position[0] / self.field_resolution + self.field_size[0] // 2)
        grid_y = int(position[1] / self.field_resolution + self.field_size[1] // 2)
        return grid_x, grid_y
    
    def _is_valid_grid_position(self, grid_x: int, grid_y: int) -> bool:
        """Check if grid position is valid."""
        return (0 <= grid_x < self.field_size[0] and 
                0 <= grid_y < self.field_size[1])
    
    def _apply_diffusion(self):
        """Apply diffusion to spread pheromones."""
        diffusion_rate = 0.1
        
        for layer_name, layer in self.pheromone_layers.items():
            # Simple 3x3 averaging kernel
            kernel = np.array([[0.05, 0.1, 0.05],
                              [0.1,  0.4, 0.1],
                              [0.05, 0.1, 0.05]])
            
            # Apply convolution (simplified)
            from scipy import ndimage
            diffused = ndimage.convolve(layer, kernel, mode='constant')
            self.pheromone_layers[layer_name] = (
                (1 - diffusion_rate) * layer + diffusion_rate * diffused
            )
    
    def get_influence_at_position(self, position: np.ndarray) -> np.ndarray:
        """Get pheromone influence vector at given position."""
        grid_x, grid_y = self._position_to_grid(position)
        
        influence = np.zeros(len(self.pheromone_layers))
        
        if self._is_valid_grid_position(grid_x, grid_y):
            for i, (layer_name, layer) in enumerate(self.pheromone_layers.items()):
                influence[i] = layer[grid_x, grid_y]
        
        return influence
    
    def get_field_state(self) -> Dict[str, Any]:
        """Get current state of pheromone field."""
        return {
            'total_pheromone': {
                layer: np.sum(field) for layer, field in self.pheromone_layers.items()
            },
            'max_concentrations': {
                layer: np.max(field) for layer, field in self.pheromone_layers.items()
            },
            'hotspots': {
                layer: self._find_hotspots(field) 
                for layer, field in self.pheromone_layers.items()
            }
        }
    
    def _find_hotspots(self, field: np.ndarray, threshold: float = 0.5) -> List[Tuple[int, int]]:
        """Find hotspots in pheromone field."""
        hotspots = []
        hotspot_positions = np.where(field > threshold)
        
        for x, y in zip(hotspot_positions[0], hotspot_positions[1]):
            hotspots.append((int(x), int(y)))
        
        return hotspots[:10]  # Limit to top 10 hotspots


def create_swarm_coordination_engine(config: Optional[SwarmConfig] = None) -> SwarmCoordinationEngine:
    """Factory function to create swarm coordination engine."""
    if config is None:
        config = SwarmConfig()
    
    engine = SwarmCoordinationEngine(config)
    
    logger.info(f"Created Swarm Coordination Engine for up to {config.max_agents} agents")
    logger.info(f"Communication range: {config.communication_range}m, Coordination dim: {config.coordination_dim}")
    
    return engine


def benchmark_swarm_coordination(engine: SwarmCoordinationEngine,
                               num_agents: int = 10,
                               num_timesteps: int = 100) -> Dict[str, float]:
    """Benchmark swarm coordination performance."""
    logger.info(f"Benchmarking Swarm Coordination with {num_agents} agents for {num_timesteps} timesteps")
    
    # Initialize agents
    agent_states = []
    for i in range(num_agents):
        agent = AgentState(
            agent_id=i,
            position=np.random.uniform(-10, 10, 3),
            velocity=np.random.uniform(-1, 1, 3),
            local_observation=torch.randn(64),
            coordination_state=torch.randn(engine.config.coordination_dim)
        )
        agent_states.append(agent)
    
    # Task context
    task_context = {
        'task_type': 'cooperative',
        'complexity': 'medium',
        'num_agents': num_agents
    }
    
    # Run coordination simulation
    start_time = time.time()
    coordination_metrics = []
    
    for timestep in range(num_timesteps):
        result = engine.coordinate_swarm(agent_states, task_context)
        coordination_metrics.append(result['coordination_metrics'])
        
        # Update agent positions (simple movement model)
        for agent in agent_states:
            agent.position += agent.velocity * 0.1
            agent.velocity += np.random.normal(0, 0.1, 3)
            agent.velocity = np.clip(agent.velocity, -2, 2)
    
    total_time = time.time() - start_time
    
    # Aggregate metrics
    if coordination_metrics:
        avg_metrics = {}
        for key in coordination_metrics[0].keys():
            values = [m[key] for m in coordination_metrics if key in m]
            avg_metrics[f'avg_{key}'] = np.mean(values) if values else 0
            avg_metrics[f'std_{key}'] = np.std(values) if values else 0
    else:
        avg_metrics = {}
    
    # Performance metrics
    performance_metrics = {
        'total_runtime': total_time,
        'avg_timestep_time': total_time / num_timesteps,
        'timesteps_per_second': num_timesteps / total_time,
        'agents_processed': num_agents * num_timesteps,
        'final_swarm_coherence': coordination_metrics[-1]['swarm_coherence'] if coordination_metrics else 0
    }
    
    results = {**avg_metrics, **performance_metrics}
    
    logger.info(f"Swarm Coordination Benchmark Results: {results}")
    
    return results
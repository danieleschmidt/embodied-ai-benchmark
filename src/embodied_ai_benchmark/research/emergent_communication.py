"""Emergent communication protocols with self-evolving vocabularies."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Any, Dict, List, Optional, Tuple, Set
from dataclasses import dataclass
from collections import defaultdict, Counter
import time
from scipy.stats import entropy
import networkx as nx

from ..core.base_agent import BaseAgent
from ..utils.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class Message:
    """Communication message between agents."""
    sender_id: str
    receiver_id: str
    content: torch.Tensor  # Symbolic content
    semantics: Optional[torch.Tensor]  # Semantic embedding
    timestamp: float
    success_feedback: Optional[bool] = None
    understanding_score: Optional[float] = None


@dataclass
class CommunicationEvent:
    """Record of communication event for analysis."""
    agent_pair: Tuple[str, str]
    message: Message
    context: Dict[str, Any]  # Environmental context
    outcome: str  # success, failure, partial
    coordination_improvement: float


class AttentionBasedVocabulary(nn.Module):
    """Self-organizing vocabulary using attention mechanisms."""
    
    def __init__(self,
                 vocab_size: int = 100,
                 symbol_dim: int = 64,
                 context_dim: int = 128,
                 num_heads: int = 8,
                 evolution_rate: float = 0.01):
        super().__init__()
        self.vocab_size = vocab_size
        self.symbol_dim = symbol_dim
        self.context_dim = context_dim
        self.evolution_rate = evolution_rate
        
        # Learnable symbol embeddings
        self.symbol_embeddings = nn.Parameter(
            torch.randn(vocab_size, symbol_dim) * 0.1
        )
        
        # Context encoder for environmental state
        self.context_encoder = nn.Sequential(
            nn.Linear(context_dim, context_dim),
            nn.ReLU(),
            nn.Linear(context_dim, symbol_dim)
        )
        
        # Multi-head attention for symbol-context association
        self.symbol_attention = nn.MultiheadAttention(
            embed_dim=symbol_dim,
            num_heads=num_heads,
            batch_first=True
        )
        
        # Semantic decoder
        self.semantic_decoder = nn.Sequential(
            nn.Linear(symbol_dim, symbol_dim * 2),
            nn.ReLU(),
            nn.Linear(symbol_dim * 2, symbol_dim),
            nn.Tanh()
        )
        
        # Usage tracking
        self.symbol_usage = torch.zeros(vocab_size)
        self.symbol_success = torch.zeros(vocab_size)
        self.co_occurrence = torch.zeros(vocab_size, vocab_size)
        
    def forward(self, 
                context: torch.Tensor,
                target_semantics: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate symbolic message from context.
        
        Args:
            context: Environmental context [batch_size, context_dim]
            target_semantics: Target semantic content [batch_size, symbol_dim]
            
        Returns:
            symbol_probs: Probability over vocabulary [batch_size, vocab_size]
            semantic_embedding: Semantic representation [batch_size, symbol_dim]
        """
        batch_size = context.size(0)
        
        # Encode context
        context_embed = self.context_encoder(context)  # [batch, symbol_dim]
        
        # Compute attention over vocabulary
        symbols = self.symbol_embeddings.unsqueeze(0).repeat(batch_size, 1, 1)  # [batch, vocab, symbol_dim]
        context_query = context_embed.unsqueeze(1)  # [batch, 1, symbol_dim]
        
        attended_symbols, attention_weights = self.symbol_attention(
            context_query, symbols, symbols
        )
        
        # Generate symbol probabilities
        symbol_logits = torch.matmul(context_embed.unsqueeze(1), symbols.transpose(1, 2)).squeeze(1)
        symbol_probs = F.softmax(symbol_logits, dim=-1)
        
        # Generate semantic embedding
        semantic_embedding = self.semantic_decoder(attended_symbols.squeeze(1))
        
        return symbol_probs, semantic_embedding
    
    def evolve_vocabulary(self, usage_feedback: Dict[int, float]):
        """Evolve vocabulary based on usage feedback."""
        for symbol_id, success_rate in usage_feedback.items():
            if symbol_id < self.vocab_size:
                # Update success tracking
                self.symbol_success[symbol_id] = (
                    0.9 * self.symbol_success[symbol_id] + 0.1 * success_rate
                )
                
                # Evolve symbol embedding based on success
                if success_rate > 0.8:
                    # Successful symbols become more distinct
                    noise = torch.randn_like(self.symbol_embeddings[symbol_id]) * self.evolution_rate
                    self.symbol_embeddings.data[symbol_id] += noise
                elif success_rate < 0.3:
                    # Failed symbols move toward successful ones
                    successful_symbols = torch.where(self.symbol_success > 0.7)[0]
                    if len(successful_symbols) > 0:
                        target = successful_symbols[torch.randint(len(successful_symbols), (1,))]
                        direction = self.symbol_embeddings[target] - self.symbol_embeddings[symbol_id]
                        self.symbol_embeddings.data[symbol_id] += self.evolution_rate * direction
        
        # Normalize embeddings
        self.symbol_embeddings.data = F.normalize(self.symbol_embeddings.data, dim=-1)
    
    def get_vocabulary_metrics(self) -> Dict[str, Any]:
        """Get metrics about vocabulary evolution."""
        usage_entropy = entropy(self.symbol_usage.detach().numpy() + 1e-8)
        success_mean = self.symbol_success.mean().item()
        
        # Compute vocabulary diversity (average pairwise distance)
        pairwise_dist = torch.cdist(self.symbol_embeddings, self.symbol_embeddings)
        diversity = pairwise_dist.mean().item()
        
        return {
            'usage_entropy': usage_entropy,
            'average_success_rate': success_mean,
            'vocabulary_diversity': diversity,
            'most_used_symbols': self.symbol_usage.topk(5).indices.tolist(),
            'most_successful_symbols': self.symbol_success.topk(5).indices.tolist()
        }


class EmergentCommProtocol:
    """Emergent communication protocol for multi-agent coordination."""
    
    def __init__(self,
                 num_agents: int,
                 vocab_size: int = 100,
                 max_message_length: int = 10,
                 bandwidth_limit: int = 5,  # Messages per timestep
                 emergence_threshold: float = 0.7):
        """
        Initialize emergent communication protocol.
        
        Args:
            num_agents: Number of participating agents
            vocab_size: Size of evolving vocabulary
            max_message_length: Maximum symbols per message
            bandwidth_limit: Maximum messages per timestep
            emergence_threshold: Threshold for detecting emergent patterns
        """
        self.num_agents = num_agents
        self.vocab_size = vocab_size
        self.max_message_length = max_message_length
        self.bandwidth_limit = bandwidth_limit
        self.emergence_threshold = emergence_threshold
        
        # Communication infrastructure
        self.message_queue = []
        self.conversation_history = defaultdict(list)
        self.protocol_metrics = defaultdict(float)
        
        # Vocabulary for each agent (initially shared, then diverges)
        self.agent_vocabularies = {}
        for i in range(num_agents):
            self.agent_vocabularies[f"agent_{i}"] = AttentionBasedVocabulary(
                vocab_size=vocab_size
            )
        
        # Emergent pattern detection
        self.pattern_detector = EmergentPatternDetector(vocab_size, max_message_length)
        
        # Communication graphs
        self.communication_graph = nx.DiGraph()
        self.communication_graph.add_nodes_from([f"agent_{i}" for i in range(num_agents)])
        
        # Success tracking
        self.communication_success = defaultdict(list)
        self.coordination_outcomes = []
        
    def send_message(self,
                    sender_id: str,
                    receiver_id: str,
                    context: torch.Tensor,
                    intent: str = "coordinate") -> Optional[Message]:
        """
        Send message from one agent to another.
        
        Args:
            sender_id: ID of sending agent
            receiver_id: ID of receiving agent  
            context: Environmental context for message generation
            intent: Communication intent (coordinate, inform, request)
            
        Returns:
            Generated message or None if bandwidth exceeded
        """
        # Check bandwidth limit
        recent_messages = [m for m in self.message_queue 
                          if time.time() - m.timestamp < 1.0]
        if len(recent_messages) >= self.bandwidth_limit:
            logger.debug(f"Bandwidth limit exceeded, dropping message from {sender_id}")
            return None
        
        # Generate message using sender's vocabulary
        sender_vocab = self.agent_vocabularies[sender_id]
        symbol_probs, semantics = sender_vocab(context.unsqueeze(0))
        
        # Sample symbols based on probabilities
        symbol_indices = torch.multinomial(symbol_probs[0], self.max_message_length, replacement=True)
        
        # Create message
        message = Message(
            sender_id=sender_id,
            receiver_id=receiver_id,
            content=symbol_indices,
            semantics=semantics[0],
            timestamp=time.time()
        )
        
        # Add to queue and history
        self.message_queue.append(message)
        self.conversation_history[(sender_id, receiver_id)].append(message)
        
        # Update communication graph
        if self.communication_graph.has_edge(sender_id, receiver_id):
            self.communication_graph[sender_id][receiver_id]['weight'] += 1
        else:
            self.communication_graph.add_edge(sender_id, receiver_id, weight=1)
        
        # Update vocabulary usage
        for symbol_id in symbol_indices:
            sender_vocab.symbol_usage[symbol_id.item()] += 1
        
        logger.debug(f"Message sent: {sender_id} -> {receiver_id}, symbols: {symbol_indices.tolist()}")
        
        return message
    
    def receive_message(self, 
                       receiver_id: str,
                       message: Message) -> Tuple[torch.Tensor, float]:
        """
        Process received message and decode meaning.
        
        Args:
            receiver_id: ID of receiving agent
            message: Message to process
            
        Returns:
            decoded_semantics: Semantic interpretation
            understanding_score: Confidence in understanding
        """
        receiver_vocab = self.agent_vocabularies[receiver_id]
        
        # Decode symbols to semantics
        symbol_embeddings = receiver_vocab.symbol_embeddings[message.content]
        decoded_semantics = symbol_embeddings.mean(dim=0)  # Simple averaging
        
        # Compute understanding score based on semantic similarity
        if message.semantics is not None:
            similarity = F.cosine_similarity(
                decoded_semantics.unsqueeze(0),
                message.semantics.unsqueeze(0)
            ).item()
            understanding_score = max(0.0, similarity)
        else:
            # Without ground truth, use vocabulary consistency
            understanding_score = 0.5
        
        # Update message with feedback
        message.understanding_score = understanding_score
        
        logger.debug(f"Message received: {message.sender_id} -> {receiver_id}, "
                    f"understanding: {understanding_score:.3f}")
        
        return decoded_semantics, understanding_score
    
    def provide_coordination_feedback(self,
                                    agent_pair: Tuple[str, str],
                                    coordination_success: bool,
                                    improvement_score: float = 0.0):
        """
        Provide feedback on coordination success after communication.
        
        Args:
            agent_pair: Tuple of (sender, receiver) agent IDs
            coordination_success: Whether coordination was successful
            improvement_score: Quantitative improvement in coordination
        """
        # Record outcome
        outcome = CommunicationEvent(
            agent_pair=agent_pair,
            message=self.conversation_history[agent_pair][-1] if self.conversation_history[agent_pair] else None,
            context={},
            outcome="success" if coordination_success else "failure",
            coordination_improvement=improvement_score
        )
        
        self.coordination_outcomes.append(outcome)
        self.communication_success[agent_pair].append(coordination_success)
        
        # Update vocabulary evolution
        if self.conversation_history[agent_pair]:
            recent_message = self.conversation_history[agent_pair][-1]
            success_rate = 1.0 if coordination_success else 0.0
            
            # Provide feedback to sender's vocabulary
            sender_vocab = self.agent_vocabularies[agent_pair[0]]
            usage_feedback = {}
            for symbol_id in recent_message.content:
                usage_feedback[symbol_id.item()] = success_rate
            sender_vocab.evolve_vocabulary(usage_feedback)
        
        logger.info(f"Coordination feedback: {agent_pair} -> {coordination_success} "
                   f"(improvement: {improvement_score:.3f})")
    
    def detect_emergent_protocols(self) -> Dict[str, Any]:
        """Detect emergent communication protocols."""
        if len(self.coordination_outcomes) < 20:
            return {"patterns_detected": []}
        
        # Analyze message patterns
        patterns = self.pattern_detector.detect_patterns(self.conversation_history)
        
        # Analyze success correlations
        success_patterns = {}
        for agent_pair, successes in self.communication_success.items():
            if len(successes) >= 5:
                success_rate = np.mean(successes[-10:])  # Recent success rate
                if success_rate > self.emergence_threshold:
                    success_patterns[agent_pair] = success_rate
        
        # Analyze vocabulary divergence
        vocab_divergence = self._compute_vocabulary_divergence()
        
        # Detect communication roles
        communication_roles = self._detect_communication_roles()
        
        return {
            "patterns_detected": patterns,
            "successful_pairs": success_patterns,
            "vocabulary_divergence": vocab_divergence,
            "communication_roles": communication_roles,
            "emergent_score": self._compute_emergence_score(patterns, success_patterns)
        }
    
    def _compute_vocabulary_divergence(self) -> Dict[str, float]:
        """Compute divergence between agent vocabularies."""
        divergences = {}
        agent_ids = list(self.agent_vocabularies.keys())
        
        for i, agent1 in enumerate(agent_ids):
            for agent2 in agent_ids[i+1:]:
                vocab1 = self.agent_vocabularies[agent1].symbol_embeddings
                vocab2 = self.agent_vocabularies[agent2].symbol_embeddings
                
                # Compute average cosine distance
                similarities = F.cosine_similarity(vocab1, vocab2, dim=1)
                divergence = 1.0 - similarities.mean().item()
                divergences[f"{agent1}_{agent2}"] = divergence
        
        return divergences
    
    def _detect_communication_roles(self) -> Dict[str, str]:
        """Detect specialized communication roles."""
        roles = {}
        
        for agent_id in self.agent_vocabularies.keys():
            # Analyze outgoing vs incoming messages
            outgoing = sum(1 for msg in self.message_queue if msg.sender_id == agent_id)
            incoming = sum(1 for msg in self.message_queue if msg.receiver_id == agent_id)
            
            # Analyze success rates
            sent_success = []
            for pair, successes in self.communication_success.items():
                if pair[0] == agent_id:
                    sent_success.extend(successes)
            
            avg_success = np.mean(sent_success) if sent_success else 0.5
            
            # Classify role
            if outgoing > incoming * 1.5 and avg_success > 0.7:
                roles[agent_id] = "coordinator"
            elif incoming > outgoing * 1.5:
                roles[agent_id] = "follower"
            elif avg_success > 0.8:
                roles[agent_id] = "expert_communicator"
            else:
                roles[agent_id] = "standard"
        
        return roles
    
    def _compute_emergence_score(self, 
                                patterns: List[Dict],
                                success_patterns: Dict) -> float:
        """Compute overall emergence score."""
        pattern_score = len(patterns) / 10.0  # Normalize by expected patterns
        success_score = len(success_patterns) / self.num_agents
        
        # Communication efficiency
        total_messages = len(self.message_queue)
        successful_coords = sum(len(successes) for successes in self.communication_success.values())
        efficiency_score = successful_coords / max(total_messages, 1)
        
        emergence_score = (pattern_score + success_score + efficiency_score) / 3.0
        return min(1.0, emergence_score)
    
    def get_protocol_metrics(self) -> Dict[str, Any]:
        """Get comprehensive protocol metrics."""
        metrics = {
            "total_messages": len(self.message_queue),
            "active_pairs": len(self.communication_success),
            "overall_success_rate": np.mean([
                np.mean(successes) for successes in self.communication_success.values()
                if successes
            ]) if self.communication_success else 0.0,
            "vocabulary_metrics": {
                agent_id: vocab.get_vocabulary_metrics()
                for agent_id, vocab in self.agent_vocabularies.items()
            },
            "communication_graph": {
                "nodes": self.communication_graph.number_of_nodes(),
                "edges": self.communication_graph.number_of_edges(),
                "density": nx.density(self.communication_graph)
            }
        }
        
        return metrics
    
    def save_protocol_state(self, filepath: str):
        """Save emergent protocol state."""
        state = {
            "vocabularies": {
                agent_id: vocab.state_dict()
                for agent_id, vocab in self.agent_vocabularies.items()
            },
            "conversation_history": dict(self.conversation_history),
            "communication_success": dict(self.communication_success),
            "coordination_outcomes": self.coordination_outcomes,
            "graph_edges": list(self.communication_graph.edges(data=True))
        }
        
        torch.save(state, filepath)
        logger.info(f"Saved protocol state to {filepath}")


class EmergentPatternDetector:
    """Detect emergent patterns in communication sequences."""
    
    def __init__(self, vocab_size: int, max_length: int):
        self.vocab_size = vocab_size
        self.max_length = max_length
        self.pattern_threshold = 0.6
        
    def detect_patterns(self, conversation_history: Dict) -> List[Dict[str, Any]]:
        """Detect recurring patterns in conversations."""
        patterns = []
        
        for agent_pair, messages in conversation_history.items():
            if len(messages) < 5:
                continue
            
            # Extract symbol sequences
            sequences = [msg.content.tolist() for msg in messages[-20:]]  # Recent messages
            
            # Find frequent subsequences
            frequent_patterns = self._find_frequent_subsequences(sequences)
            
            # Find semantic clusters
            semantic_clusters = self._find_semantic_clusters(messages[-20:])
            
            if frequent_patterns or semantic_clusters:
                patterns.append({
                    "agent_pair": agent_pair,
                    "frequent_subsequences": frequent_patterns,
                    "semantic_clusters": semantic_clusters,
                    "pattern_strength": len(frequent_patterns) + len(semantic_clusters)
                })
        
        return patterns
    
    def _find_frequent_subsequences(self, sequences: List[List[int]]) -> List[Tuple]:
        """Find frequently occurring subsequences."""
        subsequence_counts = Counter()
        
        for sequence in sequences:
            for length in range(2, min(5, len(sequence) + 1)):
                for start in range(len(sequence) - length + 1):
                    subseq = tuple(sequence[start:start + length])
                    subsequence_counts[subseq] += 1
        
        # Return patterns that appear frequently
        frequent = [subseq for subseq, count in subsequence_counts.items()
                   if count >= max(2, len(sequences) * 0.3)]
        
        return frequent
    
    def _find_semantic_clusters(self, messages: List[Message]) -> List[Dict]:
        """Find clusters of semantically similar messages."""
        if not messages or not all(msg.semantics is not None for msg in messages):
            return []
        
        # Stack semantic embeddings
        semantics = torch.stack([msg.semantics for msg in messages])
        
        # Compute pairwise similarities
        similarities = F.cosine_similarity(
            semantics.unsqueeze(1), semantics.unsqueeze(0), dim=2
        )
        
        # Find clusters of similar messages
        clusters = []
        processed = set()
        
        for i, msg in enumerate(messages):
            if i in processed:
                continue
            
            # Find similar messages
            similar_indices = torch.where(similarities[i] > self.pattern_threshold)[0]
            
            if len(similar_indices) >= 2:  # At least 2 similar messages
                cluster_messages = [messages[idx.item()] for idx in similar_indices]
                clusters.append({
                    "cluster_size": len(cluster_messages),
                    "average_similarity": similarities[i][similar_indices].mean().item(),
                    "representative_symbols": messages[i].content.tolist()
                })
                processed.update(similar_indices.tolist())
        
        return clusters
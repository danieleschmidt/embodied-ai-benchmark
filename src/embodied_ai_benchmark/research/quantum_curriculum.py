"""Quantum-inspired adaptive curriculum learning for embodied AI."""

import numpy as np
import torch
import torch.nn as nn
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import time
from scipy.stats import entropy
from scipy.special import softmax

from ..core.base_task import BaseTask
from ..core.base_agent import BaseAgent
from ..curriculum.llm_curriculum import PerformanceAnalysis
from ..utils.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class QuantumState:
    """Quantum superposition state for curriculum learning."""
    difficulty_amplitudes: np.ndarray  # Complex amplitudes for each difficulty level
    coherence_time: float  # How long the superposition lasts
    entanglement_matrix: np.ndarray  # Task-skill entanglement
    measurement_count: int  # Number of times this state has been measured


class QuantumTaskSelector(nn.Module):
    """Neural network for quantum task selection using attention mechanisms."""
    
    def __init__(self, 
                 num_difficulties: int = 10,
                 num_skills: int = 20,
                 embedding_dim: int = 128,
                 num_heads: int = 8):
        super().__init__()
        self.num_difficulties = num_difficulties
        self.num_skills = num_skills
        self.embedding_dim = embedding_dim
        
        # Task difficulty embeddings
        self.difficulty_embeddings = nn.Embedding(num_difficulties, embedding_dim)
        self.skill_embeddings = nn.Embedding(num_skills, embedding_dim)
        
        # Multi-head attention for task-skill correlation
        self.attention = nn.MultiheadAttention(
            embed_dim=embedding_dim,
            num_heads=num_heads,
            batch_first=True
        )
        
        # Quantum state predictor
        self.quantum_predictor = nn.Sequential(
            nn.Linear(embedding_dim * 2, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim // 2),
            nn.ReLU(), 
            nn.Linear(embedding_dim // 2, num_difficulties * 2)  # Real + imaginary parts
        )
        
        # Performance predictor for each difficulty
        self.performance_predictor = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim // 2),
            nn.ReLU(),
            nn.Linear(embedding_dim // 2, num_difficulties)
        )
        
    def forward(self, 
                agent_state: torch.Tensor, 
                skill_history: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for quantum task selection.
        
        Args:
            agent_state: Current agent performance state [batch_size, state_dim]
            skill_history: History of skill performance [batch_size, seq_len, num_skills]
            
        Returns:
            quantum_amplitudes: Complex amplitudes for difficulty levels
            predicted_performance: Expected performance for each difficulty
        """
        batch_size = agent_state.size(0)
        
        # Generate skill embeddings
        skill_indices = torch.arange(self.num_skills).unsqueeze(0).repeat(batch_size, 1)
        skill_embeds = self.skill_embeddings(skill_indices)  # [batch, num_skills, embed_dim]
        
        # Apply attention over skill history
        attended_skills, attention_weights = self.attention(
            skill_embeds, skill_embeds, skill_embeds
        )
        
        # Aggregate attended skills
        skill_context = attended_skills.mean(dim=1)  # [batch, embed_dim]
        
        # Combine agent state and skill context
        combined_state = torch.cat([agent_state, skill_context], dim=-1)
        
        # Predict quantum amplitudes (complex numbers)
        quantum_raw = self.quantum_predictor(combined_state)  # [batch, num_difficulties * 2]
        quantum_real = quantum_raw[:, :self.num_difficulties]
        quantum_imag = quantum_raw[:, self.num_difficulties:]
        quantum_amplitudes = torch.complex(quantum_real, quantum_imag)
        
        # Normalize to unit probability
        quantum_probs = torch.abs(quantum_amplitudes) ** 2
        quantum_probs = quantum_probs / (torch.sum(quantum_probs, dim=1, keepdim=True) + 1e-8)
        
        # Predict performance for each difficulty
        predicted_performance = self.performance_predictor(skill_context)
        predicted_performance = torch.sigmoid(predicted_performance)
        
        return quantum_amplitudes, predicted_performance


class QuantumAdaptiveCurriculum:
    """Quantum-inspired curriculum that maintains superposition of task difficulties."""
    
    def __init__(self,
                 num_difficulties: int = 10,
                 num_skills: int = 20,
                 coherence_time: float = 5.0,
                 entanglement_strength: float = 0.3,
                 measurement_threshold: float = 0.1):
        """
        Initialize quantum adaptive curriculum.
        
        Args:
            num_difficulties: Number of discrete difficulty levels
            num_skills: Number of distinct skills being learned
            coherence_time: How long quantum states maintain coherence
            entanglement_strength: Strength of task-skill entanglement
            measurement_threshold: Threshold for quantum measurement collapse
        """
        self.num_difficulties = num_difficulties
        self.num_skills = num_skills
        self.coherence_time = coherence_time
        self.entanglement_strength = entanglement_strength
        self.measurement_threshold = measurement_threshold
        
        # Initialize quantum task selector
        self.task_selector = QuantumTaskSelector(
            num_difficulties=num_difficulties,
            num_skills=num_skills
        )
        
        # Current quantum state
        self.quantum_state = self._initialize_quantum_state()
        
        # Learning history for neural network training
        self.training_data = []
        self.performance_history = []
        
        # Quantum measurement history
        self.measurement_history = []
        self.coherence_violations = 0
        
    def _initialize_quantum_state(self) -> QuantumState:
        """Initialize quantum superposition state."""
        # Start with uniform superposition
        amplitudes = np.ones(self.num_difficulties, dtype=complex) / np.sqrt(self.num_difficulties)
        
        # Random entanglement matrix
        entanglement = np.random.random((self.num_difficulties, self.num_skills)) * self.entanglement_strength
        entanglement = (entanglement + entanglement.T) / 2  # Make symmetric
        
        return QuantumState(
            difficulty_amplitudes=amplitudes,
            coherence_time=self.coherence_time,
            entanglement_matrix=entanglement,
            measurement_count=0
        )
    
    def get_superposition_difficulty(self, agent_state: Dict[str, Any]) -> Tuple[np.ndarray, float]:
        """
        Get task difficulty as quantum superposition.
        
        Args:
            agent_state: Current agent performance state
            
        Returns:
            difficulty_distribution: Probability distribution over difficulties
            quantum_entropy: Measure of quantum uncertainty
        """
        # Convert agent state to tensor
        state_vector = self._encode_agent_state(agent_state)
        skill_history = self._encode_skill_history()
        
        with torch.no_grad():
            quantum_amplitudes, predicted_performance = self.task_selector(
                state_vector.unsqueeze(0), 
                skill_history.unsqueeze(0)
            )
        
        # Convert to probability distribution
        probs = torch.abs(quantum_amplitudes[0]) ** 2
        difficulty_distribution = probs.numpy()
        
        # Calculate quantum entropy
        quantum_entropy = entropy(difficulty_distribution)
        
        # Update quantum state
        self.quantum_state.difficulty_amplitudes = quantum_amplitudes[0].numpy()
        
        return difficulty_distribution, quantum_entropy
    
    def measure_difficulty(self, 
                          agent_performance: PerformanceAnalysis,
                          exploration_bonus: float = 0.1) -> int:
        """
        Perform quantum measurement to collapse to specific difficulty.
        
        Args:
            agent_performance: Recent performance analysis
            exploration_bonus: Bonus for exploring uncertain difficulties
            
        Returns:
            Selected difficulty level (0 to num_difficulties-1)
        """
        # Get current superposition
        probs = np.abs(self.quantum_state.difficulty_amplitudes) ** 2
        
        # Add exploration bonus based on uncertainty
        uncertainty = entropy(probs)
        exploration_factor = 1.0 + exploration_bonus * uncertainty
        
        # Adjust probabilities based on recent performance
        performance_factor = self._compute_performance_factor(agent_performance)
        adjusted_probs = probs * performance_factor * exploration_factor
        adjusted_probs = adjusted_probs / np.sum(adjusted_probs)
        
        # Quantum measurement (probabilistic collapse)
        selected_difficulty = np.random.choice(self.num_difficulties, p=adjusted_probs)
        
        # Record measurement
        self.measurement_history.append({
            'timestamp': time.time(),
            'probabilities': probs.copy(),
            'selected_difficulty': selected_difficulty,
            'performance_factor': performance_factor,
            'uncertainty': uncertainty
        })
        
        self.quantum_state.measurement_count += 1
        
        # Check for coherence violation
        if uncertainty < self.measurement_threshold:
            self.coherence_violations += 1
            self._decohere_state()
        
        logger.info(f"Quantum measurement: selected difficulty {selected_difficulty}, "
                   f"uncertainty {uncertainty:.3f}")
        
        return selected_difficulty
    
    def _compute_performance_factor(self, performance: PerformanceAnalysis) -> np.ndarray:
        """Compute how performance affects difficulty probabilities."""
        factor = np.ones(self.num_difficulties)
        
        success_rate = performance.success_rate
        efficiency = performance.efficiency_score
        learning_velocity = performance.learning_velocity
        
        # If performing well, increase probability of harder tasks
        if success_rate > 0.8 and efficiency > 0.6:
            factor[self.num_difficulties//2:] *= 1.5  # Favor harder tasks
            factor[:self.num_difficulties//2] *= 0.7  # Reduce easier tasks
        
        # If struggling, increase probability of easier tasks  
        elif success_rate < 0.4:
            factor[:self.num_difficulties//2] *= 1.5  # Favor easier tasks
            factor[self.num_difficulties//2:] *= 0.5  # Reduce harder tasks
        
        # If learning quickly, add variance
        if learning_velocity > 0.2:
            factor += np.random.normal(0, 0.1, self.num_difficulties)
        
        return np.clip(factor, 0.1, 3.0)  # Prevent extreme values
    
    def _encode_agent_state(self, agent_state: Dict[str, Any]) -> torch.Tensor:
        """Encode agent state into tensor representation."""
        # Extract key performance metrics
        features = [
            agent_state.get('success_rate', 0.0),
            agent_state.get('efficiency_score', 0.0), 
            agent_state.get('learning_velocity', 0.0),
            agent_state.get('average_steps', 500.0) / 1000.0,  # Normalize
            len(agent_state.get('failure_modes', [])) / 5.0,   # Normalize
            len(agent_state.get('skill_gaps', [])) / 5.0,      # Normalize
            len(agent_state.get('mastery_indicators', [])) / 5.0,  # Normalize
            agent_state.get('confidence_level', 0.5)
        ]
        
        return torch.tensor(features, dtype=torch.float32)
    
    def _encode_skill_history(self) -> torch.Tensor:
        """Encode skill learning history."""
        # Placeholder - in real implementation would track detailed skill progression
        if len(self.performance_history) == 0:
            return torch.zeros(10, self.num_skills)
        
        # Use recent performance history as proxy for skills
        history_length = min(10, len(self.performance_history))
        skill_matrix = torch.zeros(history_length, self.num_skills)
        
        for i, perf in enumerate(self.performance_history[-history_length:]):
            # Map performance attributes to skill dimensions
            skill_matrix[i, 0] = perf.success_rate
            skill_matrix[i, 1] = perf.efficiency_score
            skill_matrix[i, 2] = perf.learning_velocity
            # Fill remaining skills with derived metrics
            for j in range(3, self.num_skills):
                skill_matrix[i, j] = np.random.random() * perf.success_rate
        
        return skill_matrix
    
    def _decohere_state(self):
        """Handle quantum decoherence when uncertainty becomes too low."""
        logger.info("Quantum decoherence detected - reinitializing superposition")
        
        # Add random phase to restore superposition
        random_phase = np.random.uniform(0, 2*np.pi, self.num_difficulties)
        self.quantum_state.difficulty_amplitudes *= np.exp(1j * random_phase)
        
        # Renormalize
        norm = np.sqrt(np.sum(np.abs(self.quantum_state.difficulty_amplitudes)**2))
        self.quantum_state.difficulty_amplitudes /= norm
    
    def update_entanglement(self, 
                           task_difficulty: int, 
                           skill_performance: Dict[str, float]):
        """Update task-skill entanglement based on observed correlations."""
        for skill_idx, performance in enumerate(skill_performance.values()):
            if skill_idx < self.num_skills:
                # Update entanglement strength based on correlation
                correlation = performance - 0.5  # Center around 0
                self.quantum_state.entanglement_matrix[task_difficulty, skill_idx] += \
                    0.1 * correlation * self.entanglement_strength
        
        # Keep entanglement matrix bounded
        self.quantum_state.entanglement_matrix = np.clip(
            self.quantum_state.entanglement_matrix, 
            -self.entanglement_strength, 
            self.entanglement_strength
        )
    
    def generate_quantum_task_batch(self, 
                                   batch_size: int,
                                   agent_state: Dict[str, Any]) -> List[Tuple[int, float]]:
        """
        Generate batch of tasks using quantum parallel processing.
        
        Args:
            batch_size: Number of tasks to generate
            agent_state: Current agent state
            
        Returns:
            List of (difficulty, uncertainty) tuples
        """
        difficulty_dist, quantum_entropy = self.get_superposition_difficulty(agent_state)
        
        # Generate multiple quantum measurements in parallel
        tasks = []
        with ThreadPoolExecutor(max_workers=min(batch_size, 8)) as executor:
            futures = []
            for _ in range(batch_size):
                future = executor.submit(np.random.choice, 
                                       self.num_difficulties, 
                                       p=difficulty_dist)
                futures.append(future)
            
            for future in futures:
                difficulty = future.result()
                uncertainty = quantum_entropy
                tasks.append((difficulty, uncertainty))
        
        return tasks
    
    def train_quantum_selector(self, 
                              performance_data: List[Dict[str, Any]],
                              num_epochs: int = 100,
                              learning_rate: float = 0.001):
        """Train the quantum task selector neural network."""
        if len(performance_data) < 10:
            logger.warning("Insufficient data for training quantum selector")
            return
        
        optimizer = torch.optim.Adam(self.task_selector.parameters(), lr=learning_rate)
        criterion = nn.MSELoss()
        
        # Prepare training data
        states = []
        skill_histories = []
        target_performances = []
        
        for data in performance_data[-1000:]:  # Use recent data
            state = self._encode_agent_state(data['agent_state'])
            skill_hist = self._encode_skill_history()
            performance = data['actual_performance']
            
            states.append(state)
            skill_histories.append(skill_hist)
            target_performances.append(torch.tensor(performance, dtype=torch.float32))
        
        # Convert to batches
        states = torch.stack(states)
        skill_histories = torch.stack(skill_histories)
        target_performances = torch.stack(target_performances)
        
        # Training loop
        self.task_selector.train()
        for epoch in range(num_epochs):
            optimizer.zero_grad()
            
            quantum_amplitudes, predicted_performance = self.task_selector(
                states, skill_histories
            )
            
            # Loss combines performance prediction and quantum consistency
            performance_loss = criterion(predicted_performance, target_performances)
            
            # Quantum consistency loss (amplitudes should sum to 1)
            quantum_probs = torch.abs(quantum_amplitudes) ** 2
            quantum_loss = criterion(
                torch.sum(quantum_probs, dim=1),
                torch.ones(quantum_probs.size(0))
            )
            
            total_loss = performance_loss + 0.1 * quantum_loss
            total_loss.backward()
            optimizer.step()
            
            if epoch % 20 == 0:
                logger.info(f"Quantum training epoch {epoch}: "
                           f"perf_loss={performance_loss:.4f}, "
                           f"quantum_loss={quantum_loss:.4f}")
        
        self.task_selector.eval()
        logger.info("Quantum selector training completed")
    
    def get_quantum_metrics(self) -> Dict[str, Any]:
        """Get quantum curriculum metrics for analysis."""
        probs = np.abs(self.quantum_state.difficulty_amplitudes) ** 2
        
        return {
            'quantum_entropy': entropy(probs),
            'coherence_time_remaining': self.quantum_state.coherence_time,
            'measurement_count': self.quantum_state.measurement_count,
            'coherence_violations': self.coherence_violations,
            'entanglement_strength': np.mean(np.abs(self.quantum_state.entanglement_matrix)),
            'difficulty_distribution': probs.tolist(),
            'most_probable_difficulty': np.argmax(probs),
            'quantum_variance': np.var(probs)
        }
    
    def save_quantum_state(self, filepath: str):
        """Save quantum curriculum state for reproducibility."""
        state_dict = {
            'quantum_state': {
                'difficulty_amplitudes': self.quantum_state.difficulty_amplitudes.tolist(),
                'coherence_time': self.quantum_state.coherence_time,
                'entanglement_matrix': self.quantum_state.entanglement_matrix.tolist(),
                'measurement_count': self.quantum_state.measurement_count
            },
            'parameters': {
                'num_difficulties': self.num_difficulties,
                'num_skills': self.num_skills,
                'entanglement_strength': self.entanglement_strength,
                'measurement_threshold': self.measurement_threshold
            },
            'history': {
                'measurement_history': self.measurement_history,
                'coherence_violations': self.coherence_violations,
                'training_data': self.training_data[-100:]  # Save recent training data
            },
            'neural_network': self.task_selector.state_dict()
        }
        
        torch.save(state_dict, filepath)
        logger.info(f"Saved quantum curriculum state to {filepath}")
    
    def load_quantum_state(self, filepath: str):
        """Load quantum curriculum state."""
        state_dict = torch.load(filepath)
        
        # Restore quantum state
        self.quantum_state.difficulty_amplitudes = np.array(
            state_dict['quantum_state']['difficulty_amplitudes'], dtype=complex
        )
        self.quantum_state.coherence_time = state_dict['quantum_state']['coherence_time']
        self.quantum_state.entanglement_matrix = np.array(
            state_dict['quantum_state']['entanglement_matrix']
        )
        self.quantum_state.measurement_count = state_dict['quantum_state']['measurement_count']
        
        # Restore history
        self.measurement_history = state_dict['history']['measurement_history']
        self.coherence_violations = state_dict['history']['coherence_violations']
        self.training_data = state_dict['history']['training_data']
        
        # Restore neural network
        self.task_selector.load_state_dict(state_dict['neural_network'])
        
        logger.info(f"Loaded quantum curriculum state from {filepath}")
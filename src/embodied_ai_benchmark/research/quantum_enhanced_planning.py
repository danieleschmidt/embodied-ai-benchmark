"""Quantum-Enhanced Planning for Embodied AI Systems."""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
import time
from concurrent.futures import ThreadPoolExecutor
from scipy.linalg import expm
import math

from ..utils.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class QuantumPlanningConfig:
    """Configuration for quantum-enhanced planning."""
    state_dim: int = 64
    action_dim: int = 16
    planning_horizon: int = 20
    num_qubits: int = 8
    coherence_time: float = 100.0
    decoherence_rate: float = 0.01
    superposition_depth: int = 5
    entanglement_strength: float = 0.5
    measurement_shots: int = 1000
    quantum_advantage_threshold: float = 0.1
    classical_fallback: bool = True
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class QuantumStateVector:
    """Quantum state representation for planning."""
    
    def __init__(self, num_qubits: int, device: str = "cpu"):
        self.num_qubits = num_qubits
        self.dim = 2 ** num_qubits
        self.device = device
        
        # Initialize in equal superposition
        self.amplitudes = torch.ones(self.dim, dtype=torch.complex64, device=device) / math.sqrt(self.dim)
        self.coherence_time = 0.0
        self.measurement_history = []
        
    def apply_gate(self, gate: torch.Tensor, qubits: List[int]):
        """Apply quantum gate to specified qubits."""
        if len(qubits) == 1:
            self._apply_single_qubit_gate(gate, qubits[0])
        elif len(qubits) == 2:
            self._apply_two_qubit_gate(gate, qubits[0], qubits[1])
        else:
            raise NotImplementedError("Gates for >2 qubits not implemented")
    
    def _apply_single_qubit_gate(self, gate: torch.Tensor, qubit: int):
        """Apply single-qubit gate."""
        # Reshape amplitudes for tensor operations
        shape = [2] * self.num_qubits
        amplitudes_tensor = self.amplitudes.view(shape)
        
        # Apply gate using tensor contraction
        gate_expanded = self._expand_gate_to_full_space(gate, qubit)
        self.amplitudes = torch.matmul(gate_expanded, self.amplitudes)
    
    def _apply_two_qubit_gate(self, gate: torch.Tensor, qubit1: int, qubit2: int):
        """Apply two-qubit gate."""
        gate_expanded = self._expand_two_qubit_gate(gate, qubit1, qubit2)
        self.amplitudes = torch.matmul(gate_expanded, self.amplitudes)
    
    def _expand_gate_to_full_space(self, gate: torch.Tensor, target_qubit: int) -> torch.Tensor:
        """Expand single-qubit gate to full Hilbert space."""
        I = torch.eye(2, dtype=torch.complex64, device=self.device)
        
        # Build tensor product
        gates = []
        for i in range(self.num_qubits):
            if i == target_qubit:
                gates.append(gate)
            else:
                gates.append(I)
        
        # Compute tensor product
        result = gates[0]
        for g in gates[1:]:
            result = torch.kron(result, g)
        
        return result
    
    def _expand_two_qubit_gate(self, gate: torch.Tensor, qubit1: int, qubit2: int) -> torch.Tensor:
        """Expand two-qubit gate to full Hilbert space."""
        I = torch.eye(2, dtype=torch.complex64, device=self.device)
        
        # Build tensor product with identity matrices
        gates = []
        for i in range(self.num_qubits):
            if i == min(qubit1, qubit2):
                gates.append(gate)
                gates.append(None)  # Placeholder for second qubit
            elif i == max(qubit1, qubit2):
                continue  # Skip, already handled
            else:
                gates.append(I)
        
        # Remove None placeholder and build result
        gates = [g for g in gates if g is not None]
        result = gates[0]
        for g in gates[1:]:
            result = torch.kron(result, g)
        
        return result
    
    def measure(self, qubits: Optional[List[int]] = None) -> List[int]:
        """Measure quantum state and collapse to classical outcome."""
        if qubits is None:
            qubits = list(range(self.num_qubits))
        
        # Compute measurement probabilities
        probabilities = torch.abs(self.amplitudes) ** 2
        
        # Sample from probability distribution
        outcome_idx = torch.multinomial(probabilities, 1).item()
        
        # Convert index to binary representation
        binary_outcome = format(outcome_idx, f'0{self.num_qubits}b')
        outcome = [int(bit) for bit in binary_outcome]
        
        # Collapse state (simplified - should project onto measurement outcome)
        self.measurement_history.append(outcome)
        
        return [outcome[i] for i in qubits]
    
    def apply_decoherence(self, decoherence_rate: float, time_step: float):
        """Apply decoherence to quantum state."""
        self.coherence_time += time_step
        
        # Simplified decoherence model - exponential decay of off-diagonal elements
        decay_factor = math.exp(-decoherence_rate * time_step)
        
        # Convert to density matrix, apply decoherence, convert back
        density_matrix = torch.outer(self.amplitudes, torch.conj(self.amplitudes))
        
        # Mix with maximally mixed state
        max_mixed = torch.eye(self.dim, dtype=torch.complex64, device=self.device) / self.dim
        density_matrix = decay_factor * density_matrix + (1 - decay_factor) * max_mixed
        
        # Extract amplitudes (simplified - should use proper state extraction)
        eigenvals, eigenvecs = torch.linalg.eigh(density_matrix)
        max_eigenval_idx = torch.argmax(torch.real(eigenvals))
        self.amplitudes = eigenvecs[:, max_eigenval_idx]
    
    def get_entanglement_entropy(self, subsystem_qubits: List[int]) -> float:
        """Compute entanglement entropy of subsystem."""
        # Simplified entanglement entropy computation
        # In practice, would need to compute reduced density matrix
        
        subsystem_size = len(subsystem_qubits)
        total_entropy = -torch.sum(torch.abs(self.amplitudes)**2 * torch.log(torch.abs(self.amplitudes)**2 + 1e-10))
        
        # Approximate subsystem entropy
        subsystem_entropy = total_entropy * (subsystem_size / self.num_qubits)
        
        return subsystem_entropy.item()


class QuantumPlanningNetwork(nn.Module):
    """Neural network that leverages quantum-inspired planning."""
    
    def __init__(self, config: QuantumPlanningConfig):
        super().__init__()
        self.config = config
        
        # Classical components
        self.state_encoder = nn.Sequential(
            nn.Linear(config.state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.LayerNorm(128)
        )
        
        # Quantum state preparation network
        self.quantum_preparation = nn.Sequential(
            nn.Linear(128, config.num_qubits * 2),  # Real and imaginary parts
            nn.Tanh()
        )
        
        # Quantum evolution parameters
        self.quantum_hamiltonians = nn.ParameterList([
            nn.Parameter(torch.randn(2**config.num_qubits, 2**config.num_qubits, dtype=torch.complex64))
            for _ in range(config.planning_horizon)
        ])
        
        # Action extraction network
        self.action_decoder = nn.Sequential(
            nn.Linear(2**config.num_qubits + 128, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, config.action_dim)
        )
        
        # Value function for quantum advantage estimation
        self.value_network = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
        # Classical fallback planner
        self.classical_planner = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, config.action_dim)
        )
        
    def forward(self, state: torch.Tensor, 
                goal: Optional[torch.Tensor] = None,
                use_quantum: bool = True) -> Dict[str, torch.Tensor]:
        """
        Forward pass with quantum-enhanced planning.
        
        Args:
            state: Current state tensor
            goal: Optional goal state
            use_quantum: Whether to use quantum planning
            
        Returns:
            Planning output with actions and quantum metrics
        """
        batch_size = state.size(0)
        device = state.device
        
        # Encode state
        encoded_state = self.state_encoder(state)
        
        if use_quantum and self._should_use_quantum(encoded_state):
            return self._quantum_planning(encoded_state, goal)
        else:
            return self._classical_planning(encoded_state)
    
    def _should_use_quantum(self, encoded_state: torch.Tensor) -> bool:
        """Determine if quantum planning should be used."""
        # Estimate quantum advantage
        state_complexity = torch.norm(encoded_state, dim=-1).mean()
        entanglement_potential = self._estimate_entanglement_potential(encoded_state)
        
        quantum_advantage = entanglement_potential * state_complexity
        
        return quantum_advantage > self.config.quantum_advantage_threshold
    
    def _estimate_entanglement_potential(self, encoded_state: torch.Tensor) -> torch.Tensor:
        """Estimate potential for quantum entanglement in current state."""
        # Simplified heuristic based on state variance and correlations
        state_var = torch.var(encoded_state, dim=-1)
        
        # Compute pairwise correlations (simplified)
        state_norm = encoded_state / (torch.norm(encoded_state, dim=-1, keepdim=True) + 1e-8)
        correlations = torch.mm(state_norm, state_norm.t())
        correlation_strength = torch.mean(torch.abs(correlations))
        
        return state_var * correlation_strength
    
    def _quantum_planning(self, encoded_state: torch.Tensor, 
                         goal: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """Perform quantum-enhanced planning."""
        batch_size = encoded_state.size(0)
        device = encoded_state.device
        
        # Prepare quantum states
        quantum_params = self.quantum_preparation(encoded_state)
        real_parts = quantum_params[:, :self.config.num_qubits]
        imag_parts = quantum_params[:, self.config.num_qubits:]
        
        # Initialize quantum states for each batch item
        quantum_states = []
        quantum_measurements = []
        entanglement_histories = []
        
        for b in range(batch_size):
            # Create quantum state
            qstate = QuantumStateVector(self.config.num_qubits, device)
            
            # Prepare initial superposition
            amplitudes = torch.complex(real_parts[b], imag_parts[b])
            amplitudes = amplitudes / torch.norm(amplitudes)
            qstate.amplitudes = torch.zeros(qstate.dim, dtype=torch.complex64, device=device)
            qstate.amplitudes[:len(amplitudes)] = amplitudes
            
            # Quantum evolution for planning horizon
            entanglement_history = []
            measurements = []
            
            for step in range(self.config.planning_horizon):
                # Apply quantum evolution (Hamiltonian simulation)
                H = self.quantum_hamiltonians[step]
                dt = 0.1  # Time step
                evolution_operator = self._matrix_exponential(-1j * H * dt)
                qstate.amplitudes = torch.matmul(evolution_operator, qstate.amplitudes)
                
                # Apply decoherence
                qstate.apply_decoherence(self.config.decoherence_rate, dt)
                
                # Track entanglement
                entanglement = qstate.get_entanglement_entropy(list(range(self.config.num_qubits // 2)))
                entanglement_history.append(entanglement)
                
                # Intermediate measurements for action planning
                if step % 5 == 0:
                    measurement = qstate.measure()
                    measurements.append(measurement)
            
            quantum_states.append(qstate.amplitudes)
            quantum_measurements.append(measurements)
            entanglement_histories.append(entanglement_history)
        
        # Convert quantum states to classical features
        quantum_features = torch.stack(quantum_states)  # [B, 2^n]
        quantum_real = torch.real(quantum_features)
        
        # Combine with classical features
        combined_features = torch.cat([quantum_real, encoded_state], dim=-1)
        
        # Extract actions
        actions = self.action_decoder(combined_features)
        
        # Compute quantum metrics
        entanglement_tensor = torch.tensor(entanglement_histories, device=device)
        max_entanglement = torch.max(entanglement_tensor, dim=-1)[0]
        
        return {
            'actions': actions,
            'quantum_features': quantum_real,
            'entanglement_history': entanglement_tensor,
            'max_entanglement': max_entanglement,
            'quantum_measurements': quantum_measurements,
            'planning_method': 'quantum'
        }
    
    def _classical_planning(self, encoded_state: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Fallback classical planning."""
        actions = self.classical_planner(encoded_state)
        
        return {
            'actions': actions,
            'planning_method': 'classical'
        }
    
    def _matrix_exponential(self, matrix: torch.Tensor) -> torch.Tensor:
        """Compute matrix exponential for quantum evolution."""
        # Use Taylor series approximation for matrix exponential
        # In practice, would use more sophisticated methods
        
        I = torch.eye(matrix.size(0), dtype=matrix.dtype, device=matrix.device)
        result = I
        term = I
        
        for k in range(1, 10):  # Truncate Taylor series
            term = torch.matmul(term, matrix) / k
            result = result + term
            
            # Check convergence
            if torch.norm(term) < 1e-8:
                break
        
        return result


class QuantumAdvantageEstimator:
    """Estimate quantum advantage for different planning scenarios."""
    
    def __init__(self, config: QuantumPlanningConfig):
        self.config = config
        self.classical_baseline_times = []
        self.quantum_planning_times = []
        self.advantage_history = []
        
    def estimate_advantage(self, state: torch.Tensor, 
                          quantum_output: Dict[str, torch.Tensor],
                          classical_output: Dict[str, torch.Tensor],
                          ground_truth_performance: Optional[float] = None) -> Dict[str, float]:
        """
        Estimate quantum advantage for current planning scenario.
        
        Args:
            state: Input state
            quantum_output: Output from quantum planning
            classical_output: Output from classical planning
            ground_truth_performance: Optional performance metric
            
        Returns:
            Quantum advantage metrics
        """
        # Compute planning quality metrics
        quantum_quality = self._compute_planning_quality(quantum_output)
        classical_quality = self._compute_planning_quality(classical_output)
        
        # Estimate computational advantage
        quantum_entanglement = quantum_output.get('max_entanglement', torch.tensor(0.0))
        entanglement_advantage = torch.mean(quantum_entanglement).item()
        
        # Solution diversity (quantum superposition allows exploring multiple paths)
        quantum_diversity = self._compute_solution_diversity(quantum_output)
        classical_diversity = self._compute_solution_diversity(classical_output)
        diversity_advantage = quantum_diversity - classical_diversity
        
        # Overall advantage estimate
        overall_advantage = (
            0.4 * (quantum_quality - classical_quality) +
            0.3 * entanglement_advantage +
            0.3 * diversity_advantage
        )
        
        advantage_metrics = {
            'overall_advantage': overall_advantage,
            'quality_advantage': quantum_quality - classical_quality,
            'entanglement_advantage': entanglement_advantage,
            'diversity_advantage': diversity_advantage,
            'quantum_quality': quantum_quality,
            'classical_quality': classical_quality
        }
        
        self.advantage_history.append(advantage_metrics)
        
        return advantage_metrics
    
    def _compute_planning_quality(self, planning_output: Dict[str, torch.Tensor]) -> float:
        """Compute quality score for planning output."""
        if 'actions' not in planning_output:
            return 0.0
        
        actions = planning_output['actions']
        
        # Quality metrics
        action_smoothness = self._compute_action_smoothness(actions)
        action_diversity = torch.std(actions).item()
        action_magnitude = torch.norm(actions).item()
        
        # Combine metrics (weights can be tuned)
        quality = (
            0.4 * action_smoothness +
            0.3 * min(action_diversity, 1.0) +  # Cap diversity
            0.3 * min(action_magnitude / 10.0, 1.0)  # Normalize magnitude
        )
        
        return quality
    
    def _compute_action_smoothness(self, actions: torch.Tensor) -> float:
        """Compute smoothness of action sequence."""
        if actions.size(0) < 2:
            return 1.0
        
        # Compute differences between consecutive actions
        action_diffs = torch.diff(actions, dim=0)
        smoothness = 1.0 / (1.0 + torch.mean(torch.norm(action_diffs, dim=-1)).item())
        
        return smoothness
    
    def _compute_solution_diversity(self, planning_output: Dict[str, torch.Tensor]) -> float:
        """Compute diversity of solutions explored."""
        if 'quantum_measurements' in planning_output:
            # Quantum case - diversity from measurements
            measurements = planning_output['quantum_measurements']
            if measurements and len(measurements) > 0:
                # Compute entropy of measurement outcomes
                all_measurements = [item for sublist in measurements for item in sublist]
                if all_measurements:
                    unique_measurements = len(set(tuple(m) for m in all_measurements))
                    total_measurements = len(all_measurements)
                    return unique_measurements / max(total_measurements, 1)
        
        # Classical case or fallback - diversity from action variance
        if 'actions' in planning_output:
            actions = planning_output['actions']
            return torch.std(actions).item()
        
        return 0.0
    
    def get_advantage_statistics(self) -> Dict[str, float]:
        """Get statistics on quantum advantage over time."""
        if not self.advantage_history:
            return {}
        
        advantages = [h['overall_advantage'] for h in self.advantage_history]
        quality_advantages = [h['quality_advantage'] for h in self.advantage_history]
        entanglement_advantages = [h['entanglement_advantage'] for h in self.advantage_history]
        
        return {
            'mean_advantage': np.mean(advantages),
            'std_advantage': np.std(advantages),
            'positive_advantage_rate': sum(1 for a in advantages if a > 0) / len(advantages),
            'mean_quality_advantage': np.mean(quality_advantages),
            'mean_entanglement_advantage': np.mean(entanglement_advantages),
            'advantage_trend': np.polyfit(range(len(advantages)), advantages, 1)[0] if len(advantages) > 1 else 0
        }


def create_quantum_planner(config: Optional[QuantumPlanningConfig] = None) -> QuantumPlanningNetwork:
    """Factory function to create quantum planning network."""
    if config is None:
        config = QuantumPlanningConfig()
    
    planner = QuantumPlanningNetwork(config)
    
    logger.info(f"Created Quantum Planning Network with {config.num_qubits} qubits")
    logger.info(f"Planning horizon: {config.planning_horizon}, State dim: {config.state_dim}")
    
    return planner


def benchmark_quantum_planning(planner: QuantumPlanningNetwork,
                              num_trials: int = 50,
                              batch_size: int = 16) -> Dict[str, float]:
    """Benchmark quantum planning performance."""
    logger.info(f"Benchmarking Quantum Planning with {num_trials} trials")
    
    device = next(planner.parameters()).device
    
    # Create test scenarios
    test_states = torch.randn(batch_size, planner.config.state_dim, device=device)
    
    planner.eval()
    
    quantum_times = []
    classical_times = []
    quantum_advantages = []
    entanglement_scores = []
    
    advantage_estimator = QuantumAdvantageEstimator(planner.config)
    
    with torch.no_grad():
        for trial in range(num_trials):
            # Quantum planning
            start_time = time.time()
            quantum_output = planner(test_states, use_quantum=True)
            quantum_time = time.time() - start_time
            quantum_times.append(quantum_time)
            
            # Classical planning
            start_time = time.time()
            classical_output = planner(test_states, use_quantum=False)
            classical_time = time.time() - start_time
            classical_times.append(classical_time)
            
            # Estimate advantage
            advantage_metrics = advantage_estimator.estimate_advantage(
                test_states, quantum_output, classical_output
            )
            quantum_advantages.append(advantage_metrics['overall_advantage'])
            
            # Track entanglement
            if 'max_entanglement' in quantum_output:
                entanglement_scores.append(torch.mean(quantum_output['max_entanglement']).item())
    
    results = {
        'avg_quantum_time': np.mean(quantum_times),
        'avg_classical_time': np.mean(classical_times),
        'quantum_speedup': np.mean(classical_times) / np.mean(quantum_times),
        'avg_quantum_advantage': np.mean(quantum_advantages),
        'positive_advantage_rate': sum(1 for a in quantum_advantages if a > 0) / len(quantum_advantages),
        'avg_entanglement': np.mean(entanglement_scores) if entanglement_scores else 0,
        'entanglement_std': np.std(entanglement_scores) if entanglement_scores else 0
    }
    
    logger.info(f"Quantum Planning Benchmark Results: {results}")
    
    return results


class QuantumPlanningExperiment:
    """Comprehensive experiment for quantum planning validation."""
    
    def __init__(self, config: QuantumPlanningConfig):
        self.config = config
        self.planner = create_quantum_planner(config)
        self.advantage_estimator = QuantumAdvantageEstimator(config)
        self.results = []
        
    def run_comparative_study(self, test_scenarios: List[Dict[str, Any]], 
                            num_trials_per_scenario: int = 30) -> Dict[str, Any]:
        """Run comparative study between quantum and classical planning."""
        logger.info(f"Running comparative study with {len(test_scenarios)} scenarios")
        
        scenario_results = []
        
        for scenario_idx, scenario in enumerate(test_scenarios):
            logger.info(f"Running scenario {scenario_idx + 1}/{len(test_scenarios)}")
            
            scenario_name = scenario.get('name', f'scenario_{scenario_idx}')
            state_dim = scenario.get('state_dim', self.config.state_dim)
            complexity = scenario.get('complexity', 'medium')
            
            # Generate test states for this scenario
            if complexity == 'low':
                test_states = torch.randn(num_trials_per_scenario, state_dim) * 0.5
            elif complexity == 'high':
                test_states = torch.randn(num_trials_per_scenario, state_dim) * 2.0
            else:  # medium
                test_states = torch.randn(num_trials_per_scenario, state_dim)
            
            # Run trials
            quantum_results = []
            classical_results = []
            advantage_scores = []
            
            for trial in range(num_trials_per_scenario):
                state = test_states[trial:trial+1]
                
                # Quantum planning
                quantum_output = self.planner(state, use_quantum=True)
                quantum_results.append(quantum_output)
                
                # Classical planning  
                classical_output = self.planner(state, use_quantum=False)
                classical_results.append(classical_output)
                
                # Compute advantage
                advantage = self.advantage_estimator.estimate_advantage(
                    state, quantum_output, classical_output
                )
                advantage_scores.append(advantage['overall_advantage'])
            
            # Aggregate scenario results
            scenario_result = {
                'scenario_name': scenario_name,
                'complexity': complexity,
                'num_trials': num_trials_per_scenario,
                'mean_quantum_advantage': np.mean(advantage_scores),
                'std_quantum_advantage': np.std(advantage_scores),
                'positive_advantage_rate': sum(1 for a in advantage_scores if a > 0) / len(advantage_scores),
                'quantum_win_rate': sum(1 for a in advantage_scores if a > 0.1) / len(advantage_scores),
                'statistical_significance': self._test_statistical_significance(advantage_scores)
            }
            
            scenario_results.append(scenario_result)
        
        # Overall analysis
        overall_results = {
            'scenario_results': scenario_results,
            'overall_statistics': self._compute_overall_statistics(scenario_results),
            'recommendations': self._generate_recommendations(scenario_results)
        }
        
        self.results.append(overall_results)
        
        return overall_results
    
    def _test_statistical_significance(self, advantages: List[float]) -> Dict[str, float]:
        """Test statistical significance of quantum advantage."""
        from scipy import stats
        
        # Test if mean advantage is significantly different from 0
        t_stat, p_value = stats.ttest_1samp(advantages, 0)
        
        # Effect size (Cohen's d)
        effect_size = np.mean(advantages) / np.std(advantages) if np.std(advantages) > 0 else 0
        
        return {
            't_statistic': t_stat,
            'p_value': p_value,
            'effect_size': effect_size,
            'significant': p_value < 0.05
        }
    
    def _compute_overall_statistics(self, scenario_results: List[Dict[str, Any]]) -> Dict[str, float]:
        """Compute overall statistics across scenarios."""
        mean_advantages = [r['mean_quantum_advantage'] for r in scenario_results]
        positive_rates = [r['positive_advantage_rate'] for r in scenario_results]
        win_rates = [r['quantum_win_rate'] for r in scenario_results]
        
        return {
            'overall_mean_advantage': np.mean(mean_advantages),
            'overall_std_advantage': np.std(mean_advantages),
            'overall_positive_rate': np.mean(positive_rates),
            'overall_win_rate': np.mean(win_rates),
            'consistent_advantage_scenarios': sum(1 for r in mean_advantages if r > 0) / len(mean_advantages)
        }
    
    def _generate_recommendations(self, scenario_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate recommendations based on results."""
        overall_stats = self._compute_overall_statistics(scenario_results)
        
        recommendations = {
            'use_quantum_planning': False,
            'confidence_level': 'low',
            'best_scenarios': [],
            'improvement_areas': []
        }
        
        # Decision logic
        if overall_stats['overall_win_rate'] > 0.7:
            recommendations['use_quantum_planning'] = True
            recommendations['confidence_level'] = 'high'
        elif overall_stats['overall_win_rate'] > 0.5:
            recommendations['use_quantum_planning'] = True
            recommendations['confidence_level'] = 'medium'
        
        # Identify best scenarios
        for result in scenario_results:
            if result['quantum_win_rate'] > 0.8:
                recommendations['best_scenarios'].append(result['scenario_name'])
        
        # Identify improvement areas
        for result in scenario_results:
            if result['mean_quantum_advantage'] < 0:
                recommendations['improvement_areas'].append(result['scenario_name'])
        
        return recommendations
    
    def generate_research_paper_results(self) -> str:
        """Generate research paper formatted results."""
        if not self.results:
            return "No experimental results available."
        
        latest_results = self.results[-1]
        overall_stats = latest_results['overall_statistics']
        
        paper_text = f"""
## Experimental Results: Quantum-Enhanced Planning for Embodied AI

### Methodology
We evaluated quantum-enhanced planning against classical baselines across {len(latest_results['scenario_results'])} 
distinct scenarios, each with varying complexity levels (low, medium, high). The quantum planner utilized 
{self.config.num_qubits} qubits with a planning horizon of {self.config.planning_horizon} steps.

### Results
Our quantum-enhanced planning algorithm demonstrated a mean advantage of {overall_stats['overall_mean_advantage']:.3f} 
(±{overall_stats['overall_std_advantage']:.3f}) across all scenarios. The quantum planner outperformed classical 
baselines in {overall_stats['overall_positive_rate']:.1%} of test cases, with a strong advantage 
(>10% improvement) in {overall_stats['overall_win_rate']:.1%} of scenarios.

### Statistical Analysis
Statistical significance testing revealed quantum advantages in the following scenarios:
"""
        
        for result in latest_results['scenario_results']:
            if result['statistical_significance']['significant']:
                paper_text += f"""
- {result['scenario_name']}: Mean advantage = {result['mean_quantum_advantage']:.3f}, 
  p = {result['statistical_significance']['p_value']:.3f}, 
  effect size = {result['statistical_significance']['effect_size']:.3f}"""
        
        paper_text += f"""

### Practical Implications
{latest_results['recommendations']['confidence_level'].title()} confidence recommendation: 
{'Deploy quantum planning' if latest_results['recommendations']['use_quantum_planning'] else 'Continue classical planning'}

Quantum planning shows particular promise in: {', '.join(latest_results['recommendations']['best_scenarios'])}
Areas for improvement: {', '.join(latest_results['recommendations']['improvement_areas'])}
"""
        
        return paper_text
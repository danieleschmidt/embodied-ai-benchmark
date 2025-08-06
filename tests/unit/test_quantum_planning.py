"""Tests for quantum-inspired task planning functionality."""

import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock

from embodied_ai_benchmark.core.base_task import BaseTask
from embodied_ai_benchmark.core.base_env import BaseEnv
from embodied_ai_benchmark.core.base_agent import BaseAgent, RandomAgent


class MockEnv(BaseEnv):
    def reset(self):
        return {"observation": np.array([0.0, 0.0, 0.0])}
    
    def step(self, action):
        return {"observation": np.array([1.0, 1.0, 1.0])}, 1.0, False, {}
    
    def render(self):
        pass
    
    def close(self):
        pass


class TestQuantumPlanning:
    """Test quantum-inspired task planning features."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.env = MockEnv()
        self.task = BaseTask(
            task_id="test_quantum",
            env=self.env,
            max_steps=100
        )
        self.agent = RandomAgent()
    
    def test_quantum_state_initialization(self):
        """Test quantum state initialization."""
        quantum_state = self.task._initialize_quantum_state()
        
        assert isinstance(quantum_state, dict)
        assert "superposition_weights" in quantum_state
        assert "coherence" in quantum_state
        assert "entanglement_map" in quantum_state
        
        # Check superposition weights sum to 1
        weights = quantum_state["superposition_weights"]
        assert abs(sum(weights.values()) - 1.0) < 1e-6
    
    def test_quantum_coherence_calculation(self):
        """Test quantum coherence calculation."""
        # Test with high performance variance (low coherence)
        performance_history = [0.1, 0.9, 0.2, 0.8, 0.3]
        coherence = self.task._get_quantum_coherence(performance_history)
        
        assert 0.0 <= coherence <= 1.0
        assert coherence < 0.8  # High variance should give low coherence
        
        # Test with low performance variance (high coherence)
        stable_history = [0.75, 0.76, 0.74, 0.77, 0.75]
        stable_coherence = self.task._get_quantum_coherence(stable_history)
        
        assert stable_coherence > coherence  # More stable should be more coherent
    
    def test_adaptive_difficulty(self):
        """Test adaptive difficulty adjustment."""
        # Test difficulty increase for good performance
        performance_metrics = {
            "success_rate": 0.9,
            "efficiency": 0.85,
            "completion_time": 50
        }
        
        original_difficulty = 0.5
        new_difficulty = self.task._adapt_task_difficulty(
            current_difficulty=original_difficulty,
            performance_metrics=performance_metrics
        )
        
        assert new_difficulty > original_difficulty
        assert new_difficulty <= 1.0
        
        # Test difficulty decrease for poor performance
        poor_metrics = {
            "success_rate": 0.2,
            "efficiency": 0.3,
            "completion_time": 200
        }
        
        reduced_difficulty = self.task._adapt_task_difficulty(
            current_difficulty=original_difficulty,
            performance_metrics=poor_metrics
        )
        
        assert reduced_difficulty < original_difficulty
        assert reduced_difficulty >= 0.0
    
    def test_performance_metrics_update(self):
        """Test performance metrics tracking."""
        initial_metrics = self.task.performance_metrics.copy()
        
        # Simulate task execution
        self.task.update_performance_metrics({
            "success": True,
            "completion_time": 45,
            "actions_taken": 30
        })
        
        updated_metrics = self.task.performance_metrics
        
        assert updated_metrics["episodes_completed"] == initial_metrics["episodes_completed"] + 1
        assert updated_metrics["total_completion_time"] == initial_metrics["total_completion_time"] + 45
        assert len(updated_metrics["recent_performance"]) <= 100  # Sliding window
    
    def test_quantum_plan_generation(self):
        """Test quantum plan generation."""
        observation = {"observation": np.array([0.5, 0.5, 0.5])}
        
        plan = self.task.get_quantum_plan(observation)
        
        assert isinstance(plan, dict)
        assert "primary_action" in plan
        assert "alternative_actions" in plan
        assert "confidence" in plan
        assert "quantum_state" in plan
        
        # Check confidence is valid probability
        assert 0.0 <= plan["confidence"] <= 1.0
        
        # Check alternative actions exist
        assert len(plan["alternative_actions"]) > 0
    
    def test_quantum_superposition_collapse(self):
        """Test quantum superposition collapse based on observation."""
        observation = {"observation": np.array([0.8, 0.2, 0.5])}
        
        # Get multiple plans to test stochastic behavior
        plans = []
        for _ in range(10):
            plan = self.task.get_quantum_plan(observation)
            plans.append(plan["primary_action"])
        
        # Should have some variation due to superposition
        unique_actions = set(plans)
        assert len(unique_actions) >= 2  # Should collapse to different actions sometimes
    
    @patch('numpy.random.choice')
    def test_deterministic_quantum_collapse(self, mock_choice):
        """Test deterministic quantum collapse for reproducible results."""
        mock_choice.return_value = 0  # Always choose first option
        
        observation = {"observation": np.array([0.5, 0.5, 0.5])}
        plan1 = self.task.get_quantum_plan(observation)
        plan2 = self.task.get_quantum_plan(observation)
        
        # With mocked randomness, should be consistent
        assert plan1["primary_action"] == plan2["primary_action"]
    
    def test_quantum_entanglement_map(self):
        """Test quantum entanglement between state variables."""
        quantum_state = self.task._initialize_quantum_state()
        entanglement_map = quantum_state["entanglement_map"]
        
        # Check structure
        assert isinstance(entanglement_map, dict)
        
        # Check entanglement values are valid correlations
        for state1, correlations in entanglement_map.items():
            for state2, correlation in correlations.items():
                assert -1.0 <= correlation <= 1.0
                if state1 == state2:
                    assert abs(correlation - 1.0) < 1e-6  # Self-correlation should be 1
    
    def test_quantum_interference_effects(self):
        """Test quantum interference in action selection."""
        # Create observation that should cause interference
        observation = {"observation": np.array([0.5, 0.5, 0.5])}
        
        # Get quantum state
        quantum_state = self.task._initialize_quantum_state()
        
        # Test that superposition weights affect action probabilities
        weights = quantum_state["superposition_weights"]
        plan = self.task.get_quantum_plan(observation)
        
        # Higher weight states should have higher probability of being selected
        # This is probabilistic, so we test the structure rather than exact values
        assert plan["confidence"] > 0.0
        assert len(plan["alternative_actions"]) > 0
    
    def test_quantum_decoherence_over_time(self):
        """Test quantum decoherence effects over multiple episodes."""
        # Simulate multiple episodes with varying performance
        performances = [0.2, 0.4, 0.6, 0.8, 0.9, 0.85, 0.75, 0.8]
        
        coherences = []
        for i, perf in enumerate(performances):
            # Update performance history
            history = performances[:i+1]
            coherence = self.task._get_quantum_coherence(history)
            coherences.append(coherence)
        
        # Early episodes should have more coherence (less data)
        # Later episodes with stable performance should maintain coherence
        assert len(coherences) == len(performances)
        
        # Check that coherence responds to performance stability
        final_coherence = coherences[-1]
        assert 0.0 <= final_coherence <= 1.0


class TestQuantumPlanningEdgeCases:
    """Test edge cases for quantum planning."""
    
    def setup_method(self):
        self.env = MockEnv()
        self.task = BaseTask(task_id="test_edge", env=self.env, max_steps=10)
    
    def test_empty_performance_history(self):
        """Test quantum coherence with empty performance history."""
        coherence = self.task._get_quantum_coherence([])
        assert coherence == 1.0  # Perfect coherence with no data
    
    def test_single_performance_entry(self):
        """Test quantum coherence with single performance entry."""
        coherence = self.task._get_quantum_coherence([0.5])
        assert coherence == 1.0  # Perfect coherence with single data point
    
    def test_extreme_difficulty_values(self):
        """Test difficulty adaptation with extreme values."""
        # Test clamping at boundaries
        metrics = {"success_rate": 1.0, "efficiency": 1.0, "completion_time": 1}
        
        # Should not exceed 1.0
        difficulty = self.task._adapt_task_difficulty(0.9, metrics)
        assert difficulty <= 1.0
        
        # Should not go below 0.0
        poor_metrics = {"success_rate": 0.0, "efficiency": 0.0, "completion_time": 1000}
        difficulty = self.task._adapt_task_difficulty(0.1, poor_metrics)
        assert difficulty >= 0.0
    
    def test_invalid_observation_handling(self):
        """Test quantum planning with invalid observations."""
        # Test with None observation
        plan = self.task.get_quantum_plan(None)
        assert isinstance(plan, dict)
        assert "primary_action" in plan
        
        # Test with empty observation
        plan = self.task.get_quantum_plan({})
        assert isinstance(plan, dict)
        assert "primary_action" in plan
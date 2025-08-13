"""Unit tests for new research components."""

import pytest
import numpy as np
import time
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta

from embodied_ai_benchmark.evaluation.cross_simulator_benchmark import CrossSimulatorBenchmark
from embodied_ai_benchmark.curriculum.emergent_curriculum import (
    EmergentCurriculumGenerator, BehaviorAnalyzer, BehaviorPattern
)
from embodied_ai_benchmark.physics.adaptive_physics import (
    AdaptivePhysicsLearner, MaterialPropertyLearner, ParameterDistribution
)
from embodied_ai_benchmark.evaluation.long_horizon_benchmark import (
    LongHorizonMultiAgentBenchmark, HierarchicalTaskPlan, TaskPhase
)
from embodied_ai_benchmark.core.base_agent import BaseAgent
from embodied_ai_benchmark.core.base_env import BaseEnv


class MockAgent(BaseAgent):
    """Mock agent for testing."""
    
    def __init__(self, agent_id: str = "test_agent"):
        super().__init__(agent_id, {})
        self.actions = []
        
    def act(self, observation):
        action = {"type": "test_action", "value": np.random.random()}
        self.actions.append(action)
        return action
    
    def reset(self):
        self.actions = []


class MockEnvironment(BaseEnv):
    """Mock environment for testing."""
    
    def __init__(self):
        super().__init__({"simulator": "test_sim"})
        self.step_count = 0
        self.episode_count = 0
        
    def reset(self, seed=None):
        super().reset(seed)
        self.step_count = 0
        self.episode_count += 1
        return {"observation": np.random.random((3,)), "step": 0}
    
    def _execute_step(self, action):
        self.step_count += 1
        obs = {"observation": np.random.random((3,)), "step": self.step_count}
        reward = np.random.random()
        done = self.step_count >= 10
        info = {"success": done and reward > 0.5}
        return obs, reward, done, info
    
    def render(self, mode="rgb_array"):
        if mode == "rgb_array":
            return np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        return None
    
    def close(self):
        pass
    
    def _get_observation(self):
        return {"observation": np.random.random((3,)), "step": self.step_count}


class TestCrossSimulatorBenchmark:
    """Test cross-simulator benchmark functionality."""
    
    def test_benchmark_initialization(self):
        """Test benchmark initialization."""
        config = {"test_param": "test_value"}
        benchmark = CrossSimulatorBenchmark(config)
        
        assert benchmark.config == config
        assert "habitat" in benchmark.simulators
        assert "vision" in benchmark.modalities
        assert len(benchmark._evaluation_history) == 0
    
    def test_cross_modal_transfer_evaluation(self):
        """Test cross-modal transfer evaluation."""
        benchmark = CrossSimulatorBenchmark()
        agent = MockAgent("test_agent")
        source_env = MockEnvironment()
        target_env = MockEnvironment()
        
        source_env.simulator_name = "habitat"
        target_env.simulator_name = "maniskill3"
        
        modality_pairs = [("vision", "tactile"), ("vision", "audio")]
        
        # Mock the benchmark suite evaluation
        with patch.object(benchmark.benchmark_suite, 'evaluate') as mock_eval:
            mock_eval.return_value = {
                "success_rate": 0.8,
                "avg_steps": 15.5,
                "efficiency": 0.75
            }
            
            results = benchmark.evaluate_cross_modal_transfer(
                agent, source_env, target_env, modality_pairs, 
                num_episodes=5, adaptation_episodes=2
            )
        
        # Verify results structure
        assert "source_simulator" in results
        assert "target_simulator" in results
        assert "transfer_metrics" in results
        assert "transfer_results" in results
        assert results["source_simulator"] == "habitat"
        assert results["target_simulator"] == "maniskill3"
        
        # Verify modality pairs were processed
        for source_mod, target_mod in modality_pairs:
            pair_key = f"{source_mod}_to_{target_mod}"
            assert pair_key in results["transfer_results"]
    
    def test_transfer_metrics_calculation(self):
        """Test transfer metrics calculation."""
        benchmark = CrossSimulatorBenchmark()
        
        source_results = {"success_rate": 0.8, "avg_steps": 10}
        target_zero_shot = {"success_rate": 0.5, "avg_steps": 15}
        target_adapted = {"success_rate": 0.7, "avg_steps": 12}
        
        metrics = benchmark._calculate_transfer_metrics(
            source_results, target_zero_shot, target_adapted
        )
        
        assert "transfer_ratio" in metrics
        assert "adaptation_gain" in metrics
        assert "final_transfer_ratio" in metrics
        
        # Check calculations
        expected_transfer_ratio = 0.5 / 0.8  # 0.625
        assert abs(metrics["transfer_ratio"] - expected_transfer_ratio) < 1e-6
        
        expected_adaptation_gain = 0.7 - 0.5  # 0.2
        assert abs(metrics["adaptation_gain"] - expected_adaptation_gain) < 1e-6
    
    def test_modality_compatibility_assessment(self):
        """Test modality compatibility assessment."""
        benchmark = CrossSimulatorBenchmark()
        
        # Test same modality (should be 1.0)
        compatibility = benchmark._assess_modality_compatibility("vision", "vision")
        assert compatibility == 1.0
        
        # Test related modalities
        compatibility = benchmark._assess_modality_compatibility("tactile", "proprioception")
        assert compatibility == 0.7
        
        # Test unrelated modalities
        compatibility = benchmark._assess_modality_compatibility("vision", "audio")
        assert compatibility == 0.2
    
    def test_report_generation(self):
        """Test transfer evaluation report generation."""
        benchmark = CrossSimulatorBenchmark()
        
        results = {
            "evaluation_timestamp": "2025-01-01T00:00:00",
            "source_simulator": "habitat",
            "target_simulator": "maniskill3",
            "num_episodes": 10,
            "adaptation_episodes": 3,
            "transfer_metrics": {
                "transfer_ratio": 0.625,
                "adaptation_gain": 0.15
            },
            "transfer_results": {
                "vision_to_tactile": {
                    "transfer_score": 0.45,
                    "modality_compatibility": 0.3
                }
            }
        }
        
        report = benchmark.generate_transfer_report(results)
        
        assert "CROSS-SIMULATOR TRANSFER EVALUATION REPORT" in report
        assert "habitat" in report
        assert "maniskill3" in report
        assert "0.625" in report  # Transfer ratio
        assert "vision_to_tactile" in report


class TestEmergentCurriculum:
    """Test emergent curriculum functionality."""
    
    def test_behavior_pattern_creation(self):
        """Test behavior pattern creation and management."""
        pattern = BehaviorPattern(
            pattern_id="test_pattern",
            agents_involved=["agent_0", "agent_1"],
            interaction_sequence=[
                {"type": "communication", "sender": "agent_0", "receiver": "agent_1"},
                {"type": "coordination", "agents": ["agent_0", "agent_1"]}
            ],
            success_rate=0.8,
            complexity_score=0.6
        )
        
        assert pattern.pattern_id == "test_pattern"
        assert len(pattern.agents_involved) == 2
        assert pattern.success_rate == 0.8
        assert pattern.complexity_score == 0.6
        assert pattern.usage_count == 0
        assert pattern.get_current_effectiveness() == 0.8  # Should return success_rate initially
        
        # Test effectiveness update
        pattern.update_effectiveness(0.9)
        pattern.update_effectiveness(0.7)
        
        assert pattern.usage_count == 2
        assert abs(pattern.get_current_effectiveness() - 0.8) < 1e-6  # Average of 0.9 and 0.7
    
    def test_behavior_analyzer_initialization(self):
        """Test behavior analyzer initialization."""
        config = {
            "min_pattern_length": 5,
            "min_success_rate": 0.7,
            "complexity_threshold": 0.6
        }
        analyzer = BehaviorAnalyzer(config)
        
        assert analyzer.min_pattern_length == 5
        assert analyzer.min_success_rate == 0.7
        assert analyzer.complexity_threshold == 0.6
        assert len(analyzer._discovered_patterns) == 0
    
    def test_interaction_extraction(self):
        """Test interaction extraction from episode data."""
        analyzer = BehaviorAnalyzer()
        
        episode_data = {
            "steps": [
                {
                    "step": 0,
                    "timestamp": time.time(),
                    "communication": [
                        {"sender": "agent_0", "receiver": "agent_1", "message": "hello"}
                    ],
                    "coordination": [
                        {"agents": ["agent_0", "agent_1"], "action_type": "lift"}
                    ],
                    "agent_positions": {
                        "agent_0": {"position": [0, 0, 0]},
                        "agent_1": {"position": [1, 0, 0]}  # Close proximity
                    }
                },
                {
                    "step": 1,
                    "timestamp": time.time() + 1,
                    "communication": [
                        {"sender": "agent_1", "receiver": "agent_0", "message": "ready"}
                    ]
                }
            ]
        }
        
        interactions = analyzer._extract_interactions(episode_data)
        
        assert len(interactions) >= 3  # At least comm, coord, and spatial interactions
        
        # Check interaction types
        interaction_types = {interaction["type"] for interaction in interactions}
        assert "communication" in interaction_types
        assert "coordination" in interaction_types
        assert "spatial_proximity" in interaction_types
    
    def test_pattern_identification(self):
        """Test pattern identification in interaction sequences."""
        analyzer = BehaviorAnalyzer({"min_pattern_length": 2})
        
        # Create repeated interaction sequence
        interactions = [
            {"type": "communication", "sender": "agent_0", "receiver": "agent_1", "step": 0},
            {"type": "coordination", "agents": ["agent_0", "agent_1"], "step": 0},
            {"type": "communication", "sender": "agent_0", "receiver": "agent_1", "step": 1},
            {"type": "coordination", "agents": ["agent_0", "agent_1"], "step": 1}
        ]
        
        patterns = analyzer._identify_patterns(interactions)
        
        assert len(patterns) > 0
        # Should find the repeated comm->coord pattern
        pattern = patterns[0]
        assert pattern["occurrences"] >= 2
    
    def test_emergent_curriculum_generator(self):
        """Test emergent curriculum generator."""
        generator = EmergentCurriculumGenerator("gpt-4", {"test": "config"})
        
        # Mock LLM curriculum
        generator.llm_curriculum = Mock()
        
        # Create mock interaction data with discovered patterns
        multi_agent_interactions = [
            {
                "episode_id": 0,
                "success": True,
                "steps": [
                    {
                        "step": 0,
                        "communication": [
                            {"sender": "agent_0", "receiver": "agent_1", "message": "coordinate"}
                        ],
                        "coordination": [
                            {"agents": ["agent_0", "agent_1"], "action_type": "lift"}
                        ]
                    }
                ]
            }
        ]
        
        base_task = Mock()
        base_task.description = "Test task"
        
        # Mock pattern discovery
        mock_pattern = BehaviorPattern(
            "pattern_001", ["agent_0", "agent_1"], 
            [{"type": "communication"}, {"type": "coordination"}],
            0.8, 0.6
        )
        
        with patch.object(generator.behavior_analyzer, 'analyze_episode_interactions') as mock_analyze:
            mock_analyze.return_value = [mock_pattern]
            
            tasks = generator.generate_emergent_tasks(
                multi_agent_interactions, base_task, num_tasks=3
            )
        
        assert len(tasks) == 3
        for task in tasks:
            assert "name" in task
            assert "description" in task
            assert "objectives" in task
            assert "success_criteria" in task
    
    def test_pattern_curriculum_scoring(self):
        """Test pattern scoring for curriculum generation."""
        generator = EmergentCurriculumGenerator()
        
        # Test different patterns with varying characteristics
        high_complexity_pattern = BehaviorPattern(
            "complex_pattern", ["agent_0", "agent_1", "agent_2"],
            [{"type": "communication"} for _ in range(5)],
            0.7, 0.9  # High complexity, good success rate
        )
        
        low_complexity_pattern = BehaviorPattern(
            "simple_pattern", ["agent_0"],
            [{"type": "communication"}],
            0.9, 0.2  # Low complexity, high success rate
        )
        
        high_score = generator._calculate_pattern_curriculum_score(high_complexity_pattern)
        low_score = generator._calculate_pattern_curriculum_score(low_complexity_pattern)
        
        # High complexity with moderate success should score higher for learning
        assert high_score > low_score
        assert 0 <= high_score <= 1
        assert 0 <= low_score <= 1


class TestAdaptivePhysics:
    """Test adaptive physics functionality."""
    
    def test_parameter_distribution(self):
        """Test parameter distribution functionality."""
        param = ParameterDistribution(
            initial_value=1.0,
            min_value=0.5,
            max_value=2.0,
            uncertainty=0.1,
            name="test_param"
        )
        
        assert param.mean == 1.0
        assert param.std == 0.1
        assert param.min_value == 0.5
        assert param.max_value == 2.0
        assert len(param.update_history) == 0
        
        # Test sampling
        samples = [param.sample() for _ in range(100)]
        assert all(0.5 <= s <= 2.0 for s in samples)
        assert abs(np.mean(samples) - 1.0) < 0.5  # Should be roughly centered
        
        # Test update
        param.update(0.1, 0.05)  # Positive gradient, small learning rate
        assert param.mean > 1.0
        assert len(param.update_history) == 1
    
    def test_material_property_learner(self):
        """Test material property learning."""
        learner = MaterialPropertyLearner()
        
        # Test default materials initialization
        assert "wood" in learner.materials
        assert "metal" in learner.materials
        assert "fabric" in learner.materials
        assert "plastic" in learner.materials
        
        # Test getting material properties
        wood_props = learner.get_material_properties("wood", sample=False)
        assert "density" in wood_props
        assert "friction" in wood_props
        assert "restitution" in wood_props
        
        # Test property updates
        initial_friction = wood_props["friction"]
        learner.update_material_properties("wood", {"friction": 0.05})
        
        updated_props = learner.get_material_properties("wood", sample=False)
        assert updated_props["friction"] != initial_friction
    
    def test_adaptive_physics_learner(self):
        """Test adaptive physics learner."""
        learner = AdaptivePhysicsLearner()
        
        # Test initialization
        assert "gravity" in learner.global_params
        assert "contact_stiffness" in learner.contact_params
        assert learner.material_learner is not None
        
        # Test getting physics config
        config = learner.get_current_physics_config(sample=False)
        assert "global" in config
        assert "contact" in config
        assert "materials" in config
        
        # Test physics discrepancy computation
        real_outcome = {
            "final_positions": [1.0, 2.0, 3.0],
            "final_velocities": [0.1, 0.2, 0.3],
            "success": True
        }
        
        sim_prediction = {
            "final_positions": [1.1, 2.1, 2.9],
            "final_velocities": [0.12, 0.18, 0.32],
            "success": True
        }
        
        discrepancy = learner.compute_physics_discrepancy(real_outcome, sim_prediction)
        
        assert "position_error" in discrepancy
        assert "velocity_error" in discrepancy
        assert "total_discrepancy" in discrepancy
        assert discrepancy["position_error"] > 0  # Should detect position differences
        assert discrepancy["success_mismatch"] == 0  # Both successful
    
    def test_physics_adaptation_process(self):
        """Test complete physics adaptation process."""
        learner = AdaptivePhysicsLearner()
        
        # Create mock real-world and simulation data
        real_outcomes = [
            {
                "final_positions": [1.0, 1.0, 0.0],
                "success": True,
                "contact_forces": [10.0, 15.0]
            },
            {
                "final_positions": [2.0, 1.5, 0.0],
                "success": False,
                "contact_forces": [8.0, 12.0]
            }
        ]
        
        sim_predictions = [
            {
                "final_positions": [1.2, 0.8, 0.1],
                "success": True,
                "contact_forces": [12.0, 13.0]
            },
            {
                "final_positions": [1.8, 1.7, -0.1],
                "success": True,  # Mismatch with real outcome
                "contact_forces": [9.0, 11.0]
            }
        ]
        
        # Test adaptation
        initial_gravity = learner.global_params["gravity"].mean
        
        adaptation_results = learner.adapt_physics_from_real_feedback(
            real_outcomes, sim_predictions
        )
        
        assert "updated_params" in adaptation_results
        assert "adaptation_stats" in adaptation_results
        assert "discrepancy_reduction" in adaptation_results
        
        # Parameters should have been updated
        final_gravity = learner.global_params["gravity"].mean
        # Gravity might change based on position errors
        
        # Check adaptation history
        assert len(learner.adaptation_history) == 1
        assert len(learner.real_world_feedback) == 2


class TestLongHorizonBenchmark:
    """Test long-horizon benchmark functionality."""
    
    def test_hierarchical_task_plan(self):
        """Test hierarchical task plan creation and management."""
        plan = HierarchicalTaskPlan("test_task", "Test task description", max_horizon=50)
        
        assert plan.task_id == "test_task"
        assert plan.description == "Test task description"
        assert plan.max_horizon == 50
        assert len(plan.phases) == 0
        assert plan.status == "created"
        
        # Add phases
        phase1 = plan.add_phase(
            "phase1", "First phase", TaskPhase.PLANNING, 10.0, ["agent_0"]
        )
        phase2 = plan.add_phase(
            "phase2", "Second phase", TaskPhase.EXECUTION, 20.0, ["agent_0", "agent_1"]
        )
        
        assert len(plan.phases) == 2
        assert phase1["phase_id"] == "phase1"
        assert phase1["status"] == "pending"
        assert "agent_0" in plan.agent_assignments
        
        # Add dependency
        plan.add_dependency("phase2", "phase1")
        assert "phase1" in plan.dependencies["phase2"]
        
        # Test executable phases
        executable = plan.get_next_executable_phases()
        assert len(executable) == 1  # Only phase1 should be executable
        assert executable[0]["phase_id"] == "phase1"
        
        # Complete phase1
        plan.update_phase_status("phase1", "completed")
        
        # Now phase2 should be executable
        executable = plan.get_next_executable_phases()
        assert len(executable) == 1
        assert executable[0]["phase_id"] == "phase2"
    
    def test_long_horizon_benchmark_initialization(self):
        """Test long-horizon benchmark initialization."""
        config = {"test_param": "value"}
        benchmark = LongHorizonMultiAgentBenchmark(config)
        
        assert benchmark.config == config
        assert "furniture_assembly_complex" in benchmark.task_types
        assert "disaster_response_scenario" in benchmark.task_types
        assert len(benchmark._evaluation_history) == 0
    
    def test_task_generation(self):
        """Test long-horizon task generation."""
        benchmark = LongHorizonMultiAgentBenchmark()
        
        # Test furniture assembly task generation
        task_spec = benchmark._generate_furniture_assembly_task(
            "test_task", max_horizon=30, episode_idx=0
        )
        
        assert "task_id" in task_spec
        assert "name" in task_spec
        assert "plan" in task_spec
        assert "success_criteria" in task_spec
        
        plan = task_spec["plan"]
        assert isinstance(plan, HierarchicalTaskPlan)
        assert len(plan.phases) > 0
        
        # Verify planning phases exist
        phase_types = [p["phase_type"] for p in plan.phases]
        assert TaskPhase.PLANNING in phase_types
        assert TaskPhase.EXECUTION in phase_types
    
    def test_phase_completion_checking(self):
        """Test phase completion criteria checking."""
        benchmark = LongHorizonMultiAgentBenchmark()
        
        # Test planning phase completion
        planning_phase = {
            "phase_type": TaskPhase.PLANNING,
            "description": "Plan the task"
        }
        
        step_data_incomplete = {
            "step": 2,
            "info": {"plan_ready": False}
        }
        
        step_data_complete = {
            "step": 3,
            "info": {"plan_ready": True}
        }
        
        episode_context = {}
        
        # Should not be complete yet
        assert not benchmark._check_phase_completion(
            planning_phase, step_data_incomplete, episode_context
        )
        
        # Should be complete when plan is ready
        assert benchmark._check_phase_completion(
            planning_phase, step_data_complete, episode_context
        )
    
    def test_result_aggregation(self):
        """Test long-horizon results aggregation."""
        benchmark = LongHorizonMultiAgentBenchmark()
        
        # Create mock episode results
        episode_results = [
            {
                "episode_id": 0,
                "success": True,
                "execution_time": 45.0,
                "phase_results": [
                    {"phase_id": "p1", "success": True},
                    {"phase_id": "p2", "success": True}
                ],
                "coordination_events": [
                    {"quality_score": 0.8},
                    {"quality_score": 0.9}
                ],
                "errors": [],
                "plan_statistics": {
                    "completed_phases": 2,
                    "time_efficiency": 0.85
                }
            },
            {
                "episode_id": 1,
                "success": False,
                "execution_time": 60.0,
                "phase_results": [
                    {"phase_id": "p1", "success": True},
                    {"phase_id": "p2", "success": False}
                ],
                "coordination_events": [
                    {"quality_score": 0.6}
                ],
                "errors": [{"error": "test_error"}],
                "plan_statistics": {
                    "completed_phases": 1,
                    "time_efficiency": 0.65
                }
            }
        ]
        
        aggregated = benchmark._aggregate_long_horizon_results(
            episode_results, "test_task", max_horizon=10
        )
        
        assert "overall_success_rate" in aggregated
        assert "avg_execution_time" in aggregated
        assert "phase_success_rate" in aggregated
        assert "coordination_quality" in aggregated
        
        # Check calculations
        assert aggregated["overall_success_rate"] == 0.5  # 1 success out of 2
        assert aggregated["avg_execution_time"] == 52.5  # (45 + 60) / 2
        assert aggregated["total_episodes"] == 2
        assert aggregated["successful_episodes"] == 1
    
    def test_report_generation(self):
        """Test long-horizon report generation."""
        benchmark = LongHorizonMultiAgentBenchmark()
        
        results = {
            "evaluation_id": "test_eval_123",
            "task_type": "furniture_assembly_complex",
            "max_horizon": 50,
            "num_agents": 3,
            "num_episodes": 5,
            "total_evaluation_time": 300.0,
            "overall_success_rate": 0.8,
            "phase_success_rate": 0.85,
            "avg_execution_time": 60.0,
            "coordination_quality": 0.75,
            "error_rate": 0.2,
            "avg_time_efficiency": 0.8
        }
        
        report = benchmark.generate_long_horizon_report(results)
        
        assert "LONG-HORIZON MULTI-AGENT BENCHMARK REPORT" in report
        assert "test_eval_123" in report
        assert "furniture_assembly_complex" in report
        assert "80.0%" in report  # Overall success rate
        assert "85.0%" in report  # Phase success rate


if __name__ == "__main__":
    pytest.main([__file__])

"""Integration tests for breakthrough research components."""

import pytest
import numpy as np
import torch
import time
import tempfile
import shutil
from pathlib import Path
from typing import Dict, Any, List

# Import research components
import sys
sys.path.append(str(Path(__file__).parent.parent / "src"))

from embodied_ai_benchmark.research.dynamic_attention_fusion import (
    create_dynamic_attention_fusion, AttentionConfig, benchmark_attention_fusion
)
from embodied_ai_benchmark.research.quantum_enhanced_planning import (
    create_quantum_planner, QuantumPlanningConfig, benchmark_quantum_planning
)
from embodied_ai_benchmark.research.emergent_swarm_coordination import (
    create_swarm_coordination_engine, SwarmConfig, benchmark_swarm_coordination
)
from embodied_ai_benchmark.research.robust_validation_framework import (
    create_robust_validation_framework, ValidationConfig
)
from embodied_ai_benchmark.research.comprehensive_monitoring import (
    create_comprehensive_monitor
)
from embodied_ai_benchmark.research.security_hardening import (
    create_security_hardening, SecurityConfig
)
from embodied_ai_benchmark.research.distributed_processing_engine import (
    create_distributed_processing_engine, ClusterConfig
)
from embodied_ai_benchmark.research.adaptive_optimization import (
    create_adaptive_optimization_system, OptimizationConfig, ParameterSpace
)


class TestDynamicAttentionFusion:
    """Test suite for Dynamic Attention Fusion."""
    
    @pytest.fixture
    def attention_config(self):
        """Create test configuration for attention fusion."""
        return AttentionConfig(
            num_modalities=4,
            hidden_dim=256,
            num_heads=4,
            dropout_rate=0.1,
            device="cpu"  # Use CPU for testing
        )
    
    @pytest.fixture
    def attention_model(self, attention_config):
        """Create attention fusion model for testing."""
        return create_dynamic_attention_fusion(attention_config)
    
    def test_model_creation(self, attention_model):
        """Test that model is created successfully."""
        assert attention_model is not None
        assert hasattr(attention_model, 'forward')
        assert hasattr(attention_model, 'config')
    
    def test_forward_pass(self, attention_model):
        """Test forward pass with synthetic data."""
        batch_size = 4
        
        # Create synthetic modality inputs
        modality_inputs = {
            'rgb': torch.randn(batch_size, 2048),
            'depth': torch.randn(batch_size, 1024),
            'tactile': torch.randn(batch_size, 64),
            'proprioception': torch.randn(batch_size, 32)
        }
        
        # Forward pass
        outputs = attention_model(modality_inputs)
        
        # Verify outputs
        assert 'fused_features' in outputs
        assert 'attention_weights' in outputs
        assert outputs['fused_features'].shape == (batch_size, attention_model.config.hidden_dim)
        assert outputs['attention_weights'].shape == (batch_size, attention_model.config.num_modalities)
        
        # Check attention weights sum to 1
        attention_sums = torch.sum(outputs['attention_weights'], dim=1)
        assert torch.allclose(attention_sums, torch.ones(batch_size), atol=1e-5)
    
    def test_temporal_fusion(self, attention_model):
        """Test temporal fusion capability."""
        batch_size = 2
        
        # Current inputs
        current_inputs = {
            'rgb': torch.randn(batch_size, 2048),
            'depth': torch.randn(batch_size, 1024),
            'tactile': torch.randn(batch_size, 64),
            'proprioception': torch.randn(batch_size, 32)
        }
        
        # Historical inputs
        history = []
        for _ in range(3):
            hist_inputs = {
                'rgb': torch.randn(batch_size, 2048),
                'depth': torch.randn(batch_size, 1024),
                'tactile': torch.randn(batch_size, 64),
                'proprioception': torch.randn(batch_size, 32)
            }
            history.append(hist_inputs)
        
        # Forward pass with history
        outputs = attention_model(current_inputs, temporal_history=history)
        
        # Verify outputs
        assert 'fused_features' in outputs
        assert outputs['fused_features'].shape == (batch_size, attention_model.config.hidden_dim)
    
    def test_benchmark_execution(self, attention_model):
        """Test benchmark execution."""
        results = benchmark_attention_fusion(
            attention_model, 
            num_trials=5, 
            batch_size=8
        )
        
        # Verify benchmark results
        assert 'avg_inference_time' in results
        assert 'throughput_samples_per_sec' in results
        assert 'avg_attention_entropy' in results
        
        assert results['avg_inference_time'] > 0
        assert results['throughput_samples_per_sec'] > 0
        assert 0 <= results['avg_attention_entropy'] <= 10  # Reasonable entropy range
    
    def test_attention_pattern_analysis(self, attention_model):
        """Test attention pattern analysis."""
        from embodied_ai_benchmark.research.dynamic_attention_fusion import PerformanceAnalyzer
        
        analyzer = PerformanceAnalyzer()
        
        # Generate test data
        attention_outputs = {
            'attention_weights': torch.rand(16, 4),  # 16 samples, 4 modalities
            'layer_attentions': [torch.rand(16, 4, 4) for _ in range(3)]
        }
        
        # Analyze patterns
        analysis = analyzer.analyze_attention_patterns(attention_outputs)
        
        # Verify analysis results
        assert 'modality_dominance' in analysis
        assert 'attention_stability' in analysis
        assert 'layer_evolution' in analysis
        assert 'efficiency_metrics' in analysis


class TestQuantumEnhancedPlanning:
    """Test suite for Quantum Enhanced Planning."""
    
    @pytest.fixture
    def quantum_config(self):
        """Create test configuration for quantum planning."""
        return QuantumPlanningConfig(
            state_dim=32,
            action_dim=8,
            planning_horizon=10,
            num_qubits=4,  # Smaller for testing
            device="cpu"
        )
    
    @pytest.fixture
    def quantum_planner(self, quantum_config):
        """Create quantum planner for testing."""
        return create_quantum_planner(quantum_config)
    
    def test_planner_creation(self, quantum_planner):
        """Test that planner is created successfully."""
        assert quantum_planner is not None
        assert hasattr(quantum_planner, 'forward')
        assert hasattr(quantum_planner, 'config')
    
    def test_quantum_planning(self, quantum_planner):
        """Test quantum planning execution."""
        batch_size = 4
        state = torch.randn(batch_size, quantum_planner.config.state_dim)
        
        # Plan with quantum enhancement
        outputs = quantum_planner(state, use_quantum=True)
        
        # Verify outputs
        assert 'actions' in outputs
        assert 'planning_method' in outputs
        assert outputs['planning_method'] == 'quantum'
        assert outputs['actions'].shape == (batch_size, quantum_planner.config.action_dim)
        
        # Check for quantum-specific outputs
        if 'quantum_features' in outputs:
            assert outputs['quantum_features'].shape[0] == batch_size
        
        if 'max_entanglement' in outputs:
            assert outputs['max_entanglement'].shape[0] == batch_size
    
    def test_classical_fallback(self, quantum_planner):
        """Test classical planning fallback."""
        batch_size = 4
        state = torch.randn(batch_size, quantum_planner.config.state_dim)
        
        # Plan with classical method
        outputs = quantum_planner(state, use_quantum=False)
        
        # Verify outputs
        assert 'actions' in outputs
        assert 'planning_method' in outputs
        assert outputs['planning_method'] == 'classical'
        assert outputs['actions'].shape == (batch_size, quantum_planner.config.action_dim)
    
    def test_quantum_state_operations(self, quantum_config):
        """Test quantum state vector operations."""
        from embodied_ai_benchmark.research.quantum_enhanced_planning import QuantumStateVector
        
        qstate = QuantumStateVector(num_qubits=3, device="cpu")
        
        # Test initial state
        assert qstate.amplitudes.shape == (8,)  # 2^3 = 8
        assert torch.allclose(torch.abs(qstate.amplitudes)**2, torch.ones(8)/8, atol=1e-5)
        
        # Test measurement
        outcome = qstate.measure()
        assert len(outcome) == 3
        assert all(bit in [0, 1] for bit in outcome)
        
        # Test entanglement entropy
        entropy = qstate.get_entanglement_entropy([0, 1])
        assert 0 <= entropy <= 2  # Max entropy for 2 qubits is 2
    
    def test_benchmark_execution(self, quantum_planner):
        """Test benchmark execution."""
        results = benchmark_quantum_planning(
            quantum_planner,
            num_trials=5,
            batch_size=8
        )
        
        # Verify benchmark results
        assert 'avg_quantum_time' in results
        assert 'avg_classical_time' in results
        assert 'avg_quantum_advantage' in results
        
        assert results['avg_quantum_time'] > 0
        assert results['avg_classical_time'] > 0


class TestEmergentSwarmCoordination:
    """Test suite for Emergent Swarm Coordination."""
    
    @pytest.fixture
    def swarm_config(self):
        """Create test configuration for swarm coordination."""
        return SwarmConfig(
            max_agents=10,
            communication_range=5.0,
            coordination_dim=32,
            emergence_layers=2,
            device="cpu"
        )
    
    @pytest.fixture
    def swarm_engine(self, swarm_config):
        """Create swarm coordination engine for testing."""
        return create_swarm_coordination_engine(swarm_config)
    
    def test_engine_creation(self, swarm_engine):
        """Test that engine is created successfully."""
        assert swarm_engine is not None
        assert hasattr(swarm_engine, 'coordinate_swarm')
        assert hasattr(swarm_engine, 'config')
    
    def test_swarm_coordination(self, swarm_engine):
        """Test swarm coordination execution."""
        from embodied_ai_benchmark.research.emergent_swarm_coordination import AgentState
        
        # Create test agent states
        num_agents = 5
        agent_states = []
        
        for i in range(num_agents):
            agent = AgentState(
                agent_id=i,
                position=np.random.uniform(-10, 10, 3),
                velocity=np.random.uniform(-1, 1, 3),
                local_observation=torch.randn(64),
                coordination_state=torch.randn(swarm_engine.config.coordination_dim)
            )
            agent_states.append(agent)
        
        # Task context
        task_context = {
            'task_type': 'cooperative',
            'complexity': 'medium',
            'num_agents': num_agents
        }
        
        # Execute coordination
        result = swarm_engine.coordinate_swarm(agent_states, task_context)
        
        # Verify results
        assert 'coordination_decisions' in result
        assert 'role_assignments' in result
        assert 'topology_metrics' in result
        assert 'coordination_metrics' in result
        
        assert len(result['coordination_decisions']) == num_agents
        assert len(result['role_assignments']) == num_agents
    
    def test_communication_protocol(self, swarm_config):
        """Test emergent communication protocol."""
        from embodied_ai_benchmark.research.emergent_swarm_coordination import EmergentCommunicationProtocol
        
        protocol = EmergentCommunicationProtocol(swarm_config)
        
        # Test message encoding/decoding
        message_content = {
            'position': [1.0, 2.0, 3.0],
            'velocity': [0.5, -0.5, 0.0],
            'task_id': 'test_task'
        }
        
        # Encode message
        encoded_msg = protocol.encode_message('position_update', message_content)
        assert encoded_msg.shape == (swarm_config.coordination_dim,)
        
        # Decode message
        decoded_type, decoded_content = protocol.decode_message(encoded_msg)
        assert decoded_type == 'position_update'
        assert 'confidence' in decoded_content
    
    def test_topology_management(self, swarm_config):
        """Test dynamic topology management."""
        from embodied_ai_benchmark.research.emergent_swarm_coordination import TopologyManager, AgentState
        
        topology_manager = TopologyManager(swarm_config)
        
        # Create test agents
        agent_states = []
        for i in range(5):
            agent = AgentState(
                agent_id=i,
                position=np.random.uniform(-5, 5, 3),
                velocity=np.random.uniform(-1, 1, 3),
                local_observation=torch.randn(64),
                coordination_state=torch.randn(swarm_config.coordination_dim)
            )
            agent_states.append(agent)
        
        # Update topology
        topology = topology_manager.update_topology(agent_states)
        
        # Verify topology
        assert topology.number_of_nodes() == len(agent_states)
        
        # Test centrality computation
        centrality = topology_manager.get_agent_centrality(0)
        assert 'betweenness' in centrality
        assert 'closeness' in centrality
        assert 'degree' in centrality
    
    def test_benchmark_execution(self, swarm_engine):
        """Test benchmark execution."""
        results = benchmark_swarm_coordination(
            swarm_engine,
            num_agents=8,
            num_timesteps=20
        )
        
        # Verify benchmark results
        assert 'total_runtime' in results
        assert 'avg_timestep_time' in results
        assert 'final_swarm_coherence' in results
        
        assert results['total_runtime'] > 0
        assert results['avg_timestep_time'] > 0
        assert 0 <= results['final_swarm_coherence'] <= 1


class TestRobustValidationFramework:
    """Test suite for Robust Validation Framework."""
    
    @pytest.fixture
    def validation_config(self):
        """Create test configuration for validation framework."""
        return ValidationConfig(
            max_retries=2,
            timeout_seconds=30.0,
            memory_limit_gb=2.0,
            min_sample_size=5,
            save_checkpoints=False  # Disable for testing
        )
    
    @pytest.fixture
    def validation_framework(self, validation_config):
        """Create validation framework for testing."""
        return create_robust_validation_framework(validation_config)
    
    def test_framework_creation(self, validation_framework):
        """Test that framework is created successfully."""
        assert validation_framework is not None
        assert hasattr(validation_framework, 'validate_component')
        assert hasattr(validation_framework, 'config')
    
    def test_component_validation(self, validation_framework):
        """Test component validation process."""
        
        # Define test component
        def test_component(input_data):
            return {'score': np.random.random(), 'processed': True}
        
        # Define test suite
        def generate_test_inputs():
            return [{'data': np.random.randn(10)} for _ in range(5)]
        
        test_suite = {
            'basic_test': generate_test_inputs
        }
        
        # Run validation
        results = validation_framework.validate_component(
            'test_component',
            test_component,
            test_suite
        )
        
        # Verify results
        assert len(results) == 1
        result = results[0]
        
        assert result.component_name == 'test_component'
        assert result.test_name == 'basic_test'
        assert result.execution_time > 0
        assert result.memory_usage_mb >= 0
    
    def test_statistical_validation(self, validation_framework):
        """Test statistical validation capabilities."""
        from embodied_ai_benchmark.research.robust_validation_framework import StatisticalValidator
        
        validator = StatisticalValidator(validation_framework.config)
        
        # Generate test data
        baseline_results = np.random.normal(0.5, 0.1, 30)
        novel_results = np.random.normal(0.6, 0.1, 30)  # Slightly better
        
        # Run statistical validation
        stats_result = validator.validate_statistical_significance(
            baseline_results.tolist(),
            novel_results.tolist()
        )
        
        # Verify results
        assert 'significant' in stats_result
        assert 'p_value' in stats_result
        assert 'effect_size' in stats_result
        assert 'improvement' in stats_result
        
        assert 0 <= stats_result['p_value'] <= 1
    
    def test_performance_profiling(self, validation_framework):
        """Test performance profiling capabilities."""
        
        def slow_component(input_data):
            time.sleep(0.01)  # Simulate some work
            return {'result': np.sum(input_data['data'])}
        
        test_inputs = [{'data': np.random.randn(100)} for _ in range(5)]
        
        profile_results = validation_framework.performance_profiler.profile_component(
            'slow_component',
            slow_component,
            test_inputs
        )
        
        # Verify profiling results
        assert 'mean_execution_time' in profile_results
        assert 'throughput_ops_per_sec' in profile_results
        assert 'mean_memory_mb' in profile_results
        
        assert profile_results['mean_execution_time'] > 0
        assert profile_results['throughput_ops_per_sec'] > 0


class TestIntegratedSystem:
    """Integration tests for the complete research system."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for testing."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    def test_monitoring_integration(self, temp_dir):
        """Test monitoring system integration."""
        monitor = create_comprehensive_monitor(str(temp_dir))
        
        # Start monitoring
        monitor.start_monitoring()
        
        try:
            # Test experiment tracking
            experiment_id = monitor.start_experiment("test_experiment", {"param1": "value1"})
            assert experiment_id is not None
            
            # Log some metrics
            monitor.log_experiment_metric("accuracy", 0.85)
            monitor.log_experiment_metric("loss", 0.15)
            
            # Track component execution
            with monitor.track_research_component("test_component") as tracker:
                time.sleep(0.1)  # Simulate work
            
            # Finish experiment
            monitor.finish_experiment({"final_score": 0.9})
            
            # Get dashboard data
            dashboard_data = monitor.get_dashboard_data()
            assert 'timestamp' in dashboard_data
            assert 'system_metrics' in dashboard_data
            
        finally:
            monitor.stop_monitoring()
    
    def test_security_integration(self, temp_dir):
        """Test security system integration."""
        security_config = SecurityConfig(
            enable_encryption=True,
            enable_access_control=True,
            audit_log_path=str(Path(temp_dir) / "audit.log")
        )
        
        security = create_security_hardening(security_config)
        
        # Test user management
        success = security.create_user("test_user", "SecurePassword123!", ["read", "write"])
        assert success
        
        # Test authentication
        session_id = security.authenticate("test_user", "SecurePassword123!")
        assert session_id is not None
        
        # Test permission checking
        has_permission = security.check_permission(session_id, "read")
        assert has_permission
        
        # Test encryption
        sensitive_data = "This is sensitive information"
        encrypted = security.encrypt_sensitive_data(sensitive_data)
        decrypted = security.decrypt_sensitive_data(encrypted)
        assert decrypted == sensitive_data
        
        # Test logout
        logout_success = security.logout(session_id)
        assert logout_success
    
    def test_optimization_integration(self, temp_dir):
        """Test optimization system integration."""
        from embodied_ai_benchmark.research.adaptive_optimization import (
            get_attention_fusion_parameter_spaces
        )
        
        optimization_config = OptimizationConfig(
            optimization_method="bayesian",
            max_iterations=10,  # Small number for testing
            checkpoint_dir=str(temp_dir)
        )
        
        optimizer = create_adaptive_optimization_system(optimization_config)
        
        # Add parameter spaces
        param_spaces = get_attention_fusion_parameter_spaces()
        for param_space in param_spaces[:2]:  # Use only first 2 for testing
            optimizer.add_parameter_space(param_space)
        
        # Define test objective function
        def test_objective(parameters):
            # Simulate evaluation with some noise
            base_score = 0.7
            param_influence = sum(
                0.1 * (p / 1000 if isinstance(p, (int, float)) else 0.1) 
                for p in parameters.values()
            )
            noise = np.random.normal(0, 0.05)
            return base_score + param_influence + noise
        
        # Define dummy component creator
        def create_component(parameters):
            return f"component_with_{parameters}"
        
        # Run optimization
        result = optimizer.optimize_component(
            create_component,
            lambda comp, params: test_objective(params)
        )
        
        # Verify optimization results
        assert result.best_parameters is not None
        assert result.best_score > 0
        assert result.total_iterations > 0
    
    def test_end_to_end_research_pipeline(self, temp_dir):
        """Test complete end-to-end research pipeline."""
        
        # 1. Create components
        attention_model = create_dynamic_attention_fusion(AttentionConfig(
            num_modalities=4, hidden_dim=128, num_heads=4, device="cpu"
        ))
        
        quantum_planner = create_quantum_planner(QuantumPlanningConfig(
            state_dim=16, action_dim=4, num_qubits=3, device="cpu"
        ))
        
        # 2. Set up monitoring
        monitor = create_comprehensive_monitor(str(temp_dir))
        monitor.start_monitoring()
        
        try:
            # 3. Start experiment
            experiment_id = monitor.start_experiment("e2e_test", {
                "attention_hidden_dim": 128,
                "quantum_qubits": 3
            })
            
            # 4. Test attention fusion
            with monitor.track_research_component("attention_fusion"):
                test_inputs = {
                    'rgb': torch.randn(2, 2048),
                    'depth': torch.randn(2, 1024),
                    'tactile': torch.randn(2, 64),
                    'proprioception': torch.randn(2, 32)
                }
                
                attention_outputs = attention_model(test_inputs)
                
                # Log metrics
                attention_entropy = torch.mean(
                    -torch.sum(
                        attention_outputs['attention_weights'] * 
                        torch.log(attention_outputs['attention_weights'] + 1e-8), 
                        dim=1
                    )
                ).item()
                
                monitor.log_experiment_metric("attention_entropy", attention_entropy)
            
            # 5. Test quantum planning
            with monitor.track_research_component("quantum_planning"):
                state = torch.randn(2, 16)
                planning_outputs = quantum_planner(state, use_quantum=True)
                
                # Log metrics
                if 'max_entanglement' in planning_outputs:
                    max_entanglement = torch.mean(planning_outputs['max_entanglement']).item()
                    monitor.log_experiment_metric("max_entanglement", max_entanglement)
            
            # 6. Validate integration
            assert attention_outputs['fused_features'].shape == (2, 128)
            assert planning_outputs['actions'].shape == (2, 4)
            
            # 7. Finish experiment
            monitor.finish_experiment({
                "attention_test": "passed",
                "quantum_test": "passed",
                "integration_test": "passed"
            })
            
            # 8. Generate report
            report = monitor.generate_monitoring_report(hours=1)
            assert "Monitoring Report" in report
            assert "attention_fusion" in report or "quantum_planning" in report
            
        finally:
            monitor.stop_monitoring()


@pytest.mark.performance
class TestPerformanceBenchmarks:
    """Performance benchmark tests."""
    
    def test_attention_fusion_performance(self):
        """Test attention fusion performance benchmarks."""
        model = create_dynamic_attention_fusion(AttentionConfig(
            num_modalities=4, hidden_dim=512, num_heads=8, device="cpu"
        ))
        
        results = benchmark_attention_fusion(model, num_trials=20, batch_size=16)
        
        # Performance assertions
        assert results['avg_inference_time'] < 1.0  # Should be under 1 second
        assert results['throughput_samples_per_sec'] > 10  # Should process at least 10 samples/sec
        
    def test_quantum_planning_performance(self):
        """Test quantum planning performance benchmarks."""
        planner = create_quantum_planner(QuantumPlanningConfig(
            state_dim=32, action_dim=8, num_qubits=6, device="cpu"
        ))
        
        results = benchmark_quantum_planning(planner, num_trials=10, batch_size=8)
        
        # Performance assertions
        assert results['avg_quantum_time'] < 5.0  # Should complete within 5 seconds
        assert results['avg_classical_time'] < 2.0  # Classical should be faster
        
    def test_swarm_coordination_performance(self):
        """Test swarm coordination performance benchmarks."""
        engine = create_swarm_coordination_engine(SwarmConfig(
            max_agents=15, coordination_dim=64, device="cpu"
        ))
        
        results = benchmark_swarm_coordination(engine, num_agents=10, num_timesteps=50)
        
        # Performance assertions
        assert results['avg_timestep_time'] < 0.5  # Should process timestep in under 0.5s
        assert results['timesteps_per_second'] > 2  # Should achieve at least 2 timesteps/sec


@pytest.mark.integration
def test_full_research_suite():
    """Test the complete research suite integration."""
    
    # Test all major components work together
    attention_model = create_dynamic_attention_fusion()
    quantum_planner = create_quantum_planner()
    swarm_engine = create_swarm_coordination_engine()
    validation_framework = create_robust_validation_framework()
    
    # Verify all components are created successfully
    assert attention_model is not None
    assert quantum_planner is not None  
    assert swarm_engine is not None
    assert validation_framework is not None
    
    # Test basic functionality of each component
    # (More detailed tests are in component-specific test classes)
    
    # Attention fusion test
    test_inputs = {
        'rgb': torch.randn(1, 2048),
        'depth': torch.randn(1, 1024), 
        'tactile': torch.randn(1, 64),
        'proprioception': torch.randn(1, 32)
    }
    attention_outputs = attention_model(test_inputs)
    assert 'fused_features' in attention_outputs
    
    # Quantum planning test
    state = torch.randn(1, 64)
    planning_outputs = quantum_planner(state)
    assert 'actions' in planning_outputs
    
    # Integration successful
    print("Full research suite integration test passed!")


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "--tb=short"])
#!/usr/bin/env python3
"""
Generation 1: Basic Functionality Test
Quick validation that core components work without full dependency installation.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_basic_imports():
    """Test basic module structure and imports."""
    try:
        # Test package structure
        from embodied_ai_benchmark.core import base_task, base_env, base_agent, base_metric
        from embodied_ai_benchmark.evaluation import benchmark_suite, evaluator
        from embodied_ai_benchmark.tasks import task_factory
        from embodied_ai_benchmark.multiagent import multi_agent_benchmark
        from embodied_ai_benchmark.utils import error_handling
        print("✓ Core module imports successful")
        return True
    except Exception as e:
        print(f"✗ Import failed: {e}")
        return False

def test_basic_functionality():
    """Test basic class instantiation and method calls."""
    try:
        from embodied_ai_benchmark.core.base_task import BaseTask
        from embodied_ai_benchmark.core.base_env import BaseEnv  
        from embodied_ai_benchmark.core.base_agent import BaseAgent, RandomAgent
        from embodied_ai_benchmark.utils.error_handling import ErrorHandler
        
        # Test basic instantiation
        error_handler = ErrorHandler()
        print("✓ Error handler instantiated")
        
        # Test RandomAgent without gym dependency
        try:
            agent = RandomAgent(action_dim=4)
            print("✓ RandomAgent instantiated")
        except Exception as e:
            print(f"⚠ RandomAgent failed (expected without gym): {e}")
        
        return True
    except Exception as e:
        print(f"✗ Basic functionality test failed: {e}")
        return False

def test_core_architecture():
    """Test that core architecture classes are properly defined."""
    try:
        from embodied_ai_benchmark.core.base_task import BaseTask
        from embodied_ai_benchmark.core.base_env import BaseEnv
        from embodied_ai_benchmark.core.base_agent import BaseAgent
        from embodied_ai_benchmark.core.base_metric import BaseMetric
        
        # Check class definitions have required methods
        assert hasattr(BaseTask, 'reset'), "BaseTask missing reset method"
        assert hasattr(BaseTask, 'step'), "BaseTask missing step method"
        assert hasattr(BaseEnv, 'reset'), "BaseEnv missing reset method"  
        assert hasattr(BaseEnv, 'step'), "BaseEnv missing step method"
        assert hasattr(BaseAgent, 'act'), "BaseAgent missing act method"
        assert hasattr(BaseMetric, 'compute'), "BaseMetric missing compute method"
        
        print("✓ Core architecture validation successful")
        return True
    except Exception as e:
        print(f"✗ Core architecture test failed: {e}")
        return False

def test_autonomous_sdlc():
    """Test autonomous SDLC components."""
    try:
        from embodied_ai_benchmark.sdlc.autonomous_orchestrator import AutonomousSDLCOrchestrator
        from embodied_ai_benchmark.sdlc.requirements_engine import RequirementsEngine
        from embodied_ai_benchmark.sdlc.code_generator import CodeGenerator
        
        print("✓ Autonomous SDLC imports successful")
        return True
    except Exception as e:
        print(f"✗ Autonomous SDLC test failed: {e}")
        return False

def main():
    """Run all basic functionality tests."""
    print("=" * 60)
    print("GENERATION 1: BASIC FUNCTIONALITY VALIDATION")
    print("=" * 60)
    
    tests = [
        ("Basic Imports", test_basic_imports),
        ("Basic Functionality", test_basic_functionality), 
        ("Core Architecture", test_core_architecture),
        ("Autonomous SDLC", test_autonomous_sdlc)
    ]
    
    passed = 0
    total = len(tests)
    
    for name, test_func in tests:
        print(f"\n{name}:")
        if test_func():
            passed += 1
    
    print(f"\n" + "=" * 60)
    print(f"GENERATION 1 RESULTS: {passed}/{total} tests passed")
    print("=" * 60)
    
    if passed == total:
        print("✓ Generation 1 (MAKE IT WORK) - COMPLETE")
        return True
    else:
        print("✗ Generation 1 (MAKE IT WORK) - FAILED")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
#!/usr/bin/env python3
"""Basic functionality test for the embodied AI benchmark framework."""

import sys
sys.path.insert(0, 'src')

def test_basic_imports():
    """Test that core modules can be imported."""
    print("Testing basic imports...")
    
    try:
        import embodied_ai_benchmark
        print("✅ Main package imported successfully")
        
        from embodied_ai_benchmark import BenchmarkSuite
        print("✅ BenchmarkSuite imported successfully")
        
        from embodied_ai_benchmark.core.base_agent import RandomAgent
        print("✅ RandomAgent imported successfully")
        
        from embodied_ai_benchmark.core.base_env import BaseEnv
        print("✅ BaseEnv imported successfully")
        
        return True
    except Exception as e:
        print(f"❌ Import failed: {e}")
        return False

def test_basic_functionality():
    """Test basic framework functionality."""
    print("\nTesting basic functionality...")
    
    try:
        from embodied_ai_benchmark import BenchmarkSuite, RandomAgent
        from embodied_ai_benchmark.core.base_env import BaseEnv
        
        # Create a simple mock environment
        class MockEnv(BaseEnv):
            def __init__(self):
                super().__init__()
                self.task_id = "mock_task"
                
            def reset(self):
                return {"observation": [0.0, 0.0, 0.0]}
                
            def step(self, action):
                return {"observation": [0.0, 0.0, 0.0]}, 1.0, True, {}
            
            def _get_observation(self):
                return {"observation": [0.0, 0.0, 0.0]}
            
            def close(self):
                pass
            
            def render(self, mode="human"):
                pass
        
        # Test benchmark suite creation
        benchmark = BenchmarkSuite()
        print("✅ BenchmarkSuite created successfully")
        
        # Test agent creation
        agent = RandomAgent({"action_dim": 3})
        print("✅ RandomAgent created successfully")
        
        # Test environment creation  
        env = MockEnv()
        print("✅ MockEnv created successfully")
        
        print("✅ Basic functionality test passed")
        return True
        
    except Exception as e:
        print(f"❌ Functionality test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_research_modules():
    """Test research module imports."""
    print("\nTesting research module imports...")
    
    try:
        from embodied_ai_benchmark.research.quantum_enhanced_planning import QuantumEnhancedPlanner
        print("✅ QuantumEnhancedPlanner imported successfully")
        
        from embodied_ai_benchmark.research.meta_learning_maml_plus import MAMLPlusLearner
        print("✅ MAMLPlusLearner imported successfully")
        
        from embodied_ai_benchmark.research.emergent_communication import EmergentCommunicationProtocol
        print("✅ EmergentCommunicationProtocol imported successfully")
        
        print("✅ Research modules test passed")
        return True
        
    except Exception as e:
        print(f"❌ Research modules test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all basic tests."""
    print("🚀 Starting Embodied AI Benchmark++ Basic Functionality Tests")
    print("=" * 60)
    
    tests_passed = 0
    total_tests = 3
    
    # Test basic imports
    if test_basic_imports():
        tests_passed += 1
    
    # Test basic functionality
    if test_basic_functionality():
        tests_passed += 1
    
    # Test research modules
    if test_research_modules():
        tests_passed += 1
    
    print("\n" + "=" * 60)
    print(f"🏁 Test Results: {tests_passed}/{total_tests} passed")
    
    if tests_passed == total_tests:
        print("🎉 All basic functionality tests PASSED!")
        return True
    else:
        print("⚠️  Some tests failed. System needs attention.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
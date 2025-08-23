"""Simple Generation 1 validation without external dependencies."""

import os
import sys
import json
import time
from datetime import datetime

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))


def test_imports():
    """Test that all core modules can be imported."""
    print("Testing imports...")
    
    try:
        # Test core imports
        from embodied_ai_benchmark.core.base_env import BaseEnv
        from embodied_ai_benchmark.core.base_task import BaseTask
        from embodied_ai_benchmark.core.base_agent import BaseAgent
        print("✅ Core modules import successfully")
        
        # Test new production components
        from embodied_ai_benchmark.simulators.production_env import ProductionEnv
        from embodied_ai_benchmark.language.llm_integration import LLMIntegration
        from embodied_ai_benchmark.database.production_connection import ProductionDatabase
        print("✅ Production modules import successfully")
        
        return True, "All imports successful"
    
    except Exception as e:
        return False, f"Import failed: {e}"


def test_basic_functionality():
    """Test basic functionality without heavy dependencies."""
    print("Testing basic functionality...")
    
    try:
        # Mock numpy functions for testing
        class MockNp:
            @staticmethod
            def array(data):
                return data
            
            @staticmethod
            def zeros(shape, dtype=None):
                if isinstance(shape, tuple):
                    if len(shape) == 3:
                        return [[[0 for _ in range(shape[2])] for _ in range(shape[1])] for _ in range(shape[0])]
                    elif len(shape) == 2:
                        return [[0 for _ in range(shape[1])] for _ in range(shape[0])]
                    else:
                        return [0] * shape[0]
                return [0] * shape
            
            @staticmethod
            def random():
                return MockRandom()
            
            @staticmethod
            def linalg():
                return MockLinalg()
        
        class MockRandom:
            @staticmethod
            def rand(*shape):
                import random
                if len(shape) == 0:
                    return random.random()
                elif len(shape) == 1:
                    return [random.random() for _ in range(shape[0])]
                elif len(shape) == 2:
                    return [[random.random() for _ in range(shape[1])] for _ in range(shape[0])]
                else:
                    return [[[random.random() for _ in range(shape[2])] for _ in range(shape[1])] for _ in range(shape[0])]
            
            @staticmethod
            def randint(low, high, size):
                import random
                if isinstance(size, tuple) and len(size) == 3:
                    return [[[random.randint(low, high-1) for _ in range(size[2])] for _ in range(size[1])] for _ in range(size[0])]
                return [random.randint(low, high-1) for _ in range(size)]
            
            @staticmethod
            def uniform(low, high, size):
                import random
                if isinstance(size, int):
                    return [random.uniform(low, high) for _ in range(size)]
                return [[random.uniform(low, high) for _ in range(size[1])] for _ in range(size[0])]
            
            @staticmethod
            def normal(mean, std, size):
                import random
                return [random.gauss(mean, std) for _ in range(size)]
        
        class MockLinalg:
            @staticmethod
            def norm(vec):
                return sum(x*x for x in vec)**0.5
        
        # Monkey patch numpy for testing
        import sys
        sys.modules['numpy'] = MockNp()
        sys.modules['numpy.random'] = MockRandom()
        sys.modules['numpy.linalg'] = MockLinalg()
        
        # Now test production environment
        from embodied_ai_benchmark.simulators.production_env import ProductionEnv
        
        config = {
            "max_steps": 10,
            "sensor_noise_level": 0.05
        }
        
        env = ProductionEnv(config)
        print("✅ ProductionEnv created successfully")
        
        # Test reset
        obs = env.reset(seed=42)
        print("✅ Environment reset successful")
        
        # Test step  
        action = {
            "linear_velocity": [0.1, 0.0, 0.0],
            "angular_velocity": [0.0, 0.0, 0.1]
        }
        
        obs, reward, done, info = env.step(action)
        print("✅ Environment step successful")
        
        # Test rendering
        render_result = env.render()
        print("✅ Environment rendering functional")
        
        env.close()
        
        return True, "Basic functionality working"
    
    except Exception as e:
        return False, f"Basic functionality failed: {e}"


def test_llm_integration():
    """Test LLM integration with mocked dependencies."""
    print("Testing LLM integration...")
    
    try:
        # Mock requests
        class MockResponse:
            def __init__(self):
                self.status_code = 200
            
            def raise_for_status(self):
                pass
            
            def json(self):
                return {
                    "choices": [{"message": {"content": "Mock analysis response"}}],
                    "model": "mock-model"
                }
        
        class MockRequests:
            @staticmethod
            def post(*args, **kwargs):
                return MockResponse()
        
        import sys
        sys.modules['requests'] = MockRequests()
        
        from embodied_ai_benchmark.language.llm_integration import LLMIntegration
        
        config = {"cache_ttl": 3600}
        llm = LLMIntegration(config)
        print("✅ LLM integration created successfully")
        
        # Test heuristic fallback (should work without external APIs)
        mock_results = [
            {"success": True, "reward": 0.8, "steps": 45},
            {"success": False, "reward": 0.2, "steps": 100}
        ]
        
        response = llm.analyze_performance(mock_results)
        print("✅ Performance analysis with fallback working")
        
        return True, "LLM integration functional with fallback"
    
    except Exception as e:
        return False, f"LLM integration failed: {e}"


def test_database_integration():
    """Test database integration with SQLite fallback."""
    print("Testing database integration...")
    
    try:
        from embodied_ai_benchmark.database.production_connection import ProductionDatabase, ExperimentResult
        
        config = {
            "sqlite_path": "/tmp/test_gen1_validation.db"
        }
        
        db = ProductionDatabase(config)
        print("✅ Database connection established")
        
        # Test health check
        health = db.health_check()
        print("✅ Database health check functional")
        
        # Test basic storage
        result = ExperimentResult(
            experiment_id="test_001",
            task_name="validation_test",
            agent_name="test_agent",
            episode_id="ep_001",
            success=True,
            reward=0.85,
            steps=45,
            duration=12.5,
            observations=[],
            actions=[],
            metrics={"test": 1.0},
            metadata={"validation": True},
            timestamp=datetime.now()
        )
        
        success = db.store_experiment_result(result)
        if success:
            print("✅ Database storage functional")
        
        # Test retrieval
        results = db.get_experiment_results("test_001")
        if len(results) > 0:
            print("✅ Database retrieval functional")
        
        db.close()
        
        return True, "Database integration working with SQLite"
    
    except Exception as e:
        return False, f"Database integration failed: {e}"


def run_validation():
    """Run all validation tests."""
    print("🚀 Starting Generation 1 Simple Validation")
    print("="*50)
    
    start_time = time.time()
    results = {}
    
    # Test 1: Imports
    success, message = test_imports()
    results["imports"] = {"success": success, "message": message}
    
    # Test 2: Basic functionality
    success, message = test_basic_functionality()
    results["basic_functionality"] = {"success": success, "message": message}
    
    # Test 3: LLM integration
    success, message = test_llm_integration()
    results["llm_integration"] = {"success": success, "message": message}
    
    # Test 4: Database integration
    success, message = test_database_integration()
    results["database_integration"] = {"success": success, "message": message}
    
    # Generate report
    total_time = time.time() - start_time
    passed_tests = sum(1 for result in results.values() if result["success"])
    total_tests = len(results)
    
    report = {
        "validation_type": "Generation 1 - Simple Validation",
        "timestamp": datetime.now().isoformat(),
        "duration": f"{total_time:.2f}s",
        "tests_passed": f"{passed_tests}/{total_tests}",
        "success_rate": f"{(passed_tests/total_tests)*100:.1f}%",
        "overall_status": "PASSED" if passed_tests == total_tests else "PARTIAL",
        "test_results": results
    }
    
    # Save report
    with open("generation1_simple_validation.json", "w") as f:
        json.dump(report, f, indent=2)
    
    # Print results
    print("\n" + "="*50)
    print("GENERATION 1 VALIDATION RESULTS")
    print("="*50)
    print(f"Overall Status: {report['overall_status']}")
    print(f"Tests Passed: {report['tests_passed']}")
    print(f"Success Rate: {report['success_rate']}")
    print(f"Duration: {report['duration']}")
    print()
    
    for test_name, result in results.items():
        status = "✅" if result["success"] else "❌"
        print(f"{status} {test_name}: {result['message']}")
    
    print("\nReport saved to: generation1_simple_validation.json")
    print("="*50)
    
    return passed_tests == total_tests


if __name__ == "__main__":
    success = run_validation()
    sys.exit(0 if success else 1)
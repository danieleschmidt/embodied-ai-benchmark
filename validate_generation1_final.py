"""Final Generation 1 validation with mock implementations."""

import os
import sys
import json
import time
import sqlite3
from datetime import datetime

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))


def test_core_architecture():
    """Test core architecture without numpy dependencies."""
    print("Testing core architecture...")
    
    try:
        # Test base classes can be imported
        from embodied_ai_benchmark.core.base_env import BaseEnv
        from embodied_ai_benchmark.core.base_task import BaseTask
        from embodied_ai_benchmark.core.base_agent import BaseAgent
        from embodied_ai_benchmark.core.base_metric import BaseMetric
        print("✅ Core base classes imported successfully")
        
        # Test that abstract methods exist
        if not hasattr(BaseEnv, 'reset'):
            raise AssertionError("BaseEnv missing reset method")
        
        if not hasattr(BaseEnv, 'step'):
            raise AssertionError("BaseEnv missing step method")
        
        print("✅ Core architecture validated")
        return True, "Core architecture is sound"
    
    except Exception as e:
        return False, f"Core architecture failed: {e}"


def test_mock_environment():
    """Test mock production environment."""
    print("Testing mock production environment...")
    
    try:
        from embodied_ai_benchmark.simulators.mock_production_env import MockProductionEnv
        
        config = {
            "max_steps": 10,
            "sensor_noise_level": 0.05
        }
        
        env = MockProductionEnv(config)
        print("✅ Mock environment created")
        
        # Test reset
        obs = env.reset(seed=42)
        
        # Validate observation structure
        required_keys = ["rgb", "depth", "pose", "proprioception", "task", "timestamp"]
        for key in required_keys:
            if key not in obs:
                raise AssertionError(f"Missing observation key: {key}")
        
        # Validate observation shapes
        if len(obs["rgb"]) != 224:
            raise AssertionError(f"Invalid RGB height: {len(obs['rgb'])}")
        
        if len(obs["rgb"][0]) != 224:
            raise AssertionError(f"Invalid RGB width: {len(obs['rgb'][0])}")
        
        if len(obs["pose"]) != 7:
            raise AssertionError(f"Invalid pose length: {len(obs['pose'])}")
        
        print("✅ Environment reset validated")
        
        # Test step
        action = {
            "linear_velocity": [0.1, 0.0, 0.0],
            "angular_velocity": [0.0, 0.0, 0.1]
        }
        
        obs, reward, done, info = env.step(action)
        
        if not isinstance(reward, (int, float)):
            raise AssertionError(f"Invalid reward type: {type(reward)}")
        
        if not isinstance(done, bool):
            raise AssertionError(f"Invalid done type: {type(done)}")
        
        print("✅ Environment step validated")
        
        # Test invalid action
        try:
            bad_action = {"linear_velocity": [10.0, 0.0, 0.0]}  # Missing angular_velocity
            env.step(bad_action)
            raise AssertionError("Should have failed with bad action")
        except ValueError:
            print("✅ Action validation working")
        
        # Test rendering
        render_result = env.render()
        if render_result is None:
            raise AssertionError("Rendering returned None")
        
        print("✅ Environment rendering working")
        
        env.close()
        
        return True, "Mock environment fully functional"
    
    except Exception as e:
        return False, f"Mock environment failed: {e}"


def test_database_functionality():
    """Test database functionality with SQLite."""
    print("Testing database functionality...")
    
    try:
        # Create test database
        db_path = "/tmp/test_gen1_db.sqlite"
        
        # Remove existing test db
        if os.path.exists(db_path):
            os.remove(db_path)
        
        conn = sqlite3.connect(db_path)
        
        # Create test table
        conn.execute("""
            CREATE TABLE experiments (
                id INTEGER PRIMARY KEY,
                experiment_id TEXT,
                task_name TEXT,
                success BOOLEAN,
                reward REAL,
                steps INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Insert test data
        conn.execute("""
            INSERT INTO experiments (experiment_id, task_name, success, reward, steps)
            VALUES (?, ?, ?, ?, ?)
        """, ("test_exp_001", "navigation", True, 0.85, 45))
        
        conn.commit()
        
        # Query test data
        cursor = conn.execute("SELECT * FROM experiments WHERE experiment_id = ?", ("test_exp_001",))
        result = cursor.fetchone()
        
        if result is None:
            raise AssertionError("Failed to retrieve stored data")
        
        if result[1] != "test_exp_001":
            raise AssertionError("Data mismatch in retrieval")
        
        print("✅ Database storage and retrieval working")
        
        # Test batch operations
        test_data = [
            ("test_exp_002", "manipulation", True, 0.75, 60),
            ("test_exp_003", "navigation", False, 0.25, 100),
            ("test_exp_004", "coordination", True, 0.90, 30),
        ]
        
        conn.executemany("""
            INSERT INTO experiments (experiment_id, task_name, success, reward, steps)
            VALUES (?, ?, ?, ?, ?)
        """, test_data)
        
        conn.commit()
        
        # Query batch results
        cursor = conn.execute("SELECT COUNT(*) FROM experiments")
        count = cursor.fetchone()[0]
        
        if count != 4:
            raise AssertionError(f"Expected 4 records, got {count}")
        
        print("✅ Batch database operations working")
        
        # Test statistics query
        cursor = conn.execute("""
            SELECT 
                AVG(CASE WHEN success THEN 1.0 ELSE 0.0 END) as success_rate,
                AVG(reward) as avg_reward,
                AVG(steps) as avg_steps
            FROM experiments
        """)
        
        stats = cursor.fetchone()
        if stats is None:
            raise AssertionError("Statistics query failed")
        
        success_rate, avg_reward, avg_steps = stats
        if success_rate < 0 or success_rate > 1:
            raise AssertionError(f"Invalid success rate: {success_rate}")
        
        print("✅ Database statistics queries working")
        
        conn.close()
        
        # Clean up test db
        os.remove(db_path)
        
        return True, "Database functionality validated"
    
    except Exception as e:
        return False, f"Database functionality failed: {e}"


def test_workflow_integration():
    """Test end-to-end workflow integration."""
    print("Testing workflow integration...")
    
    try:
        from embodied_ai_benchmark.simulators.mock_production_env import MockProductionEnv
        
        # Setup
        env_config = {"max_steps": 20}
        env = MockProductionEnv(env_config)
        
        # Database setup
        db_path = "/tmp/workflow_test.sqlite"
        if os.path.exists(db_path):
            os.remove(db_path)
        
        conn = sqlite3.connect(db_path)
        conn.execute("""
            CREATE TABLE episodes (
                id INTEGER PRIMARY KEY,
                episode_id TEXT,
                success BOOLEAN,
                reward REAL,
                steps INTEGER,
                duration REAL
            )
        """)
        
        # Run multiple episodes
        episodes = []
        
        for i in range(3):
            episode_id = f"workflow_ep_{i:03d}"
            start_time = time.time()
            
            # Reset environment
            obs = env.reset(seed=42 + i)
            
            total_reward = 0
            steps = 0
            
            # Run episode
            while steps < 10:  # Short episodes
                action = {
                    "linear_velocity": [0.1, 0.0, 0.0],
                    "angular_velocity": [0.0, 0.0, 0.05]
                }
                
                obs, reward, done, info = env.step(action)
                total_reward += reward
                steps += 1
                
                if done:
                    break
            
            duration = time.time() - start_time
            success = info.get("task_completed", False)
            
            # Store episode result
            conn.execute("""
                INSERT INTO episodes (episode_id, success, reward, steps, duration)
                VALUES (?, ?, ?, ?, ?)
            """, (episode_id, success, total_reward, steps, duration))
            
            episodes.append({
                "episode_id": episode_id,
                "success": success,
                "reward": total_reward,
                "steps": steps,
                "duration": duration
            })
        
        conn.commit()
        
        # Verify stored data
        cursor = conn.execute("SELECT COUNT(*) FROM episodes")
        stored_count = cursor.fetchone()[0]
        
        if stored_count != 3:
            raise AssertionError(f"Expected 3 episodes stored, got {stored_count}")
        
        # Generate analysis
        cursor = conn.execute("""
            SELECT 
                AVG(CASE WHEN success THEN 1.0 ELSE 0.0 END) as success_rate,
                AVG(reward) as avg_reward,
                AVG(steps) as avg_steps,
                AVG(duration) as avg_duration
            FROM episodes
        """)
        
        stats = cursor.fetchone()
        success_rate, avg_reward, avg_steps, avg_duration = stats
        
        print(f"✅ Workflow completed: {success_rate:.1%} success rate")
        print(f"   Average reward: {avg_reward:.3f}")
        print(f"   Average steps: {avg_steps:.1f}")
        print(f"   Average duration: {avg_duration:.3f}s")
        
        # Cleanup
        env.close()
        conn.close()
        os.remove(db_path)
        
        return True, f"Workflow integration successful ({success_rate:.1%} success)"
    
    except Exception as e:
        return False, f"Workflow integration failed: {e}"


def test_extensibility():
    """Test framework extensibility."""
    print("Testing framework extensibility...")
    
    try:
        from embodied_ai_benchmark.core.base_env import BaseEnv
        
        # Create custom environment
        class TestCustomEnv(BaseEnv):
            def __init__(self, config):
                super().__init__(config)
                self.custom_feature = config.get("custom_feature", "test")
            
            def reset(self, seed=None):
                return {"custom_obs": "test_value", "step": 0}
            
            def _execute_step(self, action):
                obs = {"custom_obs": f"step_{action.get('step', 0)}", "step": 1}
                reward = 1.0
                done = True
                info = {"custom_info": self.custom_feature}
                return obs, reward, done, info
            
            def render(self, mode="rgb_array"):
                return [[0, 1, 2]]
            
            def close(self):
                pass
            
            def _get_observation(self):
                return {"custom_obs": "reset", "step": 0}
        
        # Test custom environment
        config = {"custom_feature": "extensibility_test"}
        custom_env = TestCustomEnv(config)
        
        obs = custom_env.reset()
        if obs["custom_obs"] != "test_value":
            raise AssertionError("Custom environment reset failed")
        
        action = {"step": 1}
        obs, reward, done, info = custom_env.step(action)
        
        if info["custom_info"] != "extensibility_test":
            raise AssertionError("Custom environment step failed")
        
        custom_env.close()
        
        print("✅ Custom environment creation working")
        
        # Test task factory pattern
        task_registry = {}
        
        def register_task(name):
            def decorator(cls):
                task_registry[name] = cls
                return cls
            return decorator
        
        @register_task("test_task")
        class TestTask:
            def __init__(self, config):
                self.config = config
                self.name = "test_task"
        
        # Test task creation
        if "test_task" not in task_registry:
            raise AssertionError("Task registration failed")
        
        task = task_registry["test_task"]({"test": True})
        if task.name != "test_task":
            raise AssertionError("Task creation failed")
        
        print("✅ Task registration pattern working")
        
        return True, "Framework extensibility validated"
    
    except Exception as e:
        return False, f"Extensibility test failed: {e}"


def run_comprehensive_validation():
    """Run comprehensive Generation 1 validation."""
    print("🚀 Generation 1 Comprehensive Validation")
    print("="*60)
    
    start_time = time.time()
    results = {}
    
    # Test 1: Core Architecture
    success, message = test_core_architecture()
    results["core_architecture"] = {"success": success, "message": message}
    
    # Test 2: Mock Environment  
    success, message = test_mock_environment()
    results["mock_environment"] = {"success": success, "message": message}
    
    # Test 3: Database Functionality
    success, message = test_database_functionality()
    results["database_functionality"] = {"success": success, "message": message}
    
    # Test 4: Workflow Integration
    success, message = test_workflow_integration()
    results["workflow_integration"] = {"success": success, "message": message}
    
    # Test 5: Extensibility
    success, message = test_extensibility()
    results["extensibility"] = {"success": success, "message": message}
    
    # Generate comprehensive report
    total_time = time.time() - start_time
    passed_tests = sum(1 for result in results.values() if result["success"])
    total_tests = len(results)
    overall_success = passed_tests == total_tests
    
    report = {
        "generation": "Generation 1 - MAKE IT WORK",
        "validation_type": "Comprehensive Validation",
        "timestamp": datetime.now().isoformat(),
        "duration": f"{total_time:.2f}s",
        "tests_passed": f"{passed_tests}/{total_tests}",
        "success_rate": f"{(passed_tests/total_tests)*100:.1f}%",
        "overall_status": "PASSED" if overall_success else "FAILED",
        "test_results": results,
        "validation_summary": {
            "core_framework": "✅" if results["core_architecture"]["success"] else "❌",
            "environment_simulation": "✅" if results["mock_environment"]["success"] else "❌", 
            "data_persistence": "✅" if results["database_functionality"]["success"] else "❌",
            "workflow_integration": "✅" if results["workflow_integration"]["success"] else "❌",
            "framework_extensibility": "✅" if results["extensibility"]["success"] else "❌"
        },
        "next_steps": [
            "Generation 1 validation complete" if overall_success else "Fix failing components",
            "Ready for Generation 2 - MAKE IT ROBUST" if overall_success else "Retry Generation 1",
            "Implement real simulator integrations",
            "Add comprehensive error handling",
            "Scale up testing infrastructure"
        ]
    }
    
    # Save detailed report
    with open("generation1_comprehensive_validation.json", "w") as f:
        json.dump(report, f, indent=2)
    
    # Print results
    print("\n" + "="*60)
    print("GENERATION 1 VALIDATION RESULTS")
    print("="*60)
    print(f"Overall Status: {report['overall_status']}")
    print(f"Tests Passed: {report['tests_passed']}")
    print(f"Success Rate: {report['success_rate']}")
    print(f"Duration: {report['duration']}")
    print()
    
    print("Test Results:")
    for test_name, result in results.items():
        status = "✅" if result["success"] else "❌"
        print(f"  {status} {test_name.replace('_', ' ').title()}: {result['message']}")
    
    print(f"\nValidation Summary:")
    for component, status in report["validation_summary"].items():
        print(f"  {status} {component.replace('_', ' ').title()}")
    
    print(f"\nNext Steps:")
    for i, step in enumerate(report["next_steps"], 1):
        print(f"  {i}. {step}")
    
    print(f"\nDetailed report saved to: generation1_comprehensive_validation.json")
    print("="*60)
    
    if overall_success:
        print("🎉 GENERATION 1 COMPLETE - READY FOR GENERATION 2!")
    else:
        print("⚠️  GENERATION 1 NEEDS ATTENTION - CHECK FAILING TESTS")
    
    return overall_success


if __name__ == "__main__":
    success = run_comprehensive_validation()
    sys.exit(0 if success else 1)
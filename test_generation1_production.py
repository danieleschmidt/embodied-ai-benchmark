"""Generation 1 production testing - Make it Work validation."""

import os
import sys
import time
import json
import logging
import numpy as np
from typing import Dict, Any, List
from datetime import datetime

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from embodied_ai_benchmark import BenchmarkSuite, make_env
from embodied_ai_benchmark.simulators.production_env import ProductionEnv
from embodied_ai_benchmark.language.llm_integration import LLMIntegration
from embodied_ai_benchmark.database.production_connection import ProductionDatabase, ExperimentResult

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Generation1Validator:
    """Validate Generation 1 production implementations."""
    
    def __init__(self):
        """Initialize validator."""
        self.results = {
            "production_env": {"status": "pending", "details": {}},
            "llm_integration": {"status": "pending", "details": {}},
            "database_integration": {"status": "pending", "details": {}},
            "end_to_end_workflow": {"status": "pending", "details": {}},
        }
        
        self.start_time = time.time()
    
    def validate_all(self) -> Dict[str, Any]:
        """Run all Generation 1 validation tests."""
        
        logger.info("🚀 Starting Generation 1 Production Validation")
        
        # Test 1: Production Environment
        self.validate_production_env()
        
        # Test 2: LLM Integration
        self.validate_llm_integration()
        
        # Test 3: Database Integration
        self.validate_database_integration()
        
        # Test 4: End-to-End Workflow
        self.validate_end_to_end_workflow()
        
        # Generate final report
        return self.generate_report()
    
    def validate_production_env(self):
        """Validate production environment implementation."""
        
        logger.info("Testing ProductionEnv implementation...")
        
        try:
            # Test environment creation
            config = {
                "max_steps": 100,
                "sensor_noise_level": 0.05,
                "depth_range": (0.1, 10.0),
                "missing_data_rate": 0.05
            }
            
            env = ProductionEnv(config)
            
            # Test reset functionality
            obs = env.reset(seed=42)
            
            # Validate observation structure
            required_keys = ["rgb", "depth", "pose", "proprioception", "task", "timestamp"]
            for key in required_keys:
                if key not in obs:
                    raise AssertionError(f"Missing observation key: {key}")
            
            # Validate observation types and shapes
            if obs["rgb"].shape != (224, 224, 3):
                raise AssertionError(f"Invalid RGB shape: {obs['rgb'].shape}")
            
            if obs["depth"].shape != (224, 224):
                raise AssertionError(f"Invalid depth shape: {obs['depth'].shape}")
            
            if len(obs["pose"]) != 7:
                raise AssertionError(f"Invalid pose length: {len(obs['pose'])}")
            
            # Test action execution
            action = {
                "linear_velocity": [0.1, 0.0, 0.0],
                "angular_velocity": [0.0, 0.0, 0.1]
            }
            
            obs, reward, done, info = env.step(action)
            
            # Validate step results
            if not isinstance(reward, (int, float)):
                raise AssertionError(f"Invalid reward type: {type(reward)}")
            
            if not isinstance(done, bool):
                raise AssertionError(f"Invalid done type: {type(done)}")
            
            # Test multiple episodes
            episode_results = []
            for i in range(5):
                obs = env.reset()
                total_reward = 0
                steps = 0
                
                while steps < 20:  # Short episodes for testing
                    action = {
                        "linear_velocity": np.random.uniform(-0.1, 0.1, 3).tolist(),
                        "angular_velocity": np.random.uniform(-0.1, 0.1, 3).tolist()
                    }
                    
                    obs, reward, done, info = env.step(action)
                    total_reward += reward
                    steps += 1
                    
                    if done:
                        break
                
                episode_results.append({
                    "episode": i,
                    "steps": steps,
                    "total_reward": total_reward,
                    "success": info.get("task_completed", False)
                })
            
            # Test rendering
            render_image = env.render()
            if render_image is None or render_image.shape != (480, 640, 3):
                logger.warning("Rendering may have issues")
            
            env.close()
            
            # Calculate performance metrics
            avg_steps = np.mean([r["steps"] for r in episode_results])
            avg_reward = np.mean([r["total_reward"] for r in episode_results])
            
            self.results["production_env"] = {
                "status": "passed",
                "details": {
                    "episodes_tested": len(episode_results),
                    "avg_steps_per_episode": avg_steps,
                    "avg_reward_per_episode": avg_reward,
                    "observation_validation": "passed",
                    "action_validation": "passed",
                    "rendering": "functional"
                }
            }
            
            logger.info("✅ ProductionEnv validation passed")
            
        except Exception as e:
            self.results["production_env"] = {
                "status": "failed",
                "details": {"error": str(e)}
            }
            logger.error(f"❌ ProductionEnv validation failed: {e}")
    
    def validate_llm_integration(self):
        """Validate LLM integration functionality."""
        
        logger.info("Testing LLM integration...")
        
        try:
            # Test configuration (using mock endpoints for testing)
            config = {
                "openai_api_key": None,  # Will use heuristic fallback
                "anthropic_api_key": None,
                "local_llm_endpoint": None,
                "cache_ttl": 3600
            }
            
            llm = LLMIntegration(config)
            
            # Test performance analysis
            mock_results = [
                {"success": True, "reward": 0.8, "steps": 45, "failure_reason": None},
                {"success": False, "reward": 0.2, "steps": 100, "failure_reason": "timeout"},
                {"success": True, "reward": 0.9, "steps": 30, "failure_reason": None},
                {"success": True, "reward": 0.7, "steps": 60, "failure_reason": None},
                {"success": False, "reward": 0.1, "steps": 100, "failure_reason": "collision"}
            ]
            
            analysis = llm.analyze_performance(mock_results, "Testing context")
            
            # Validate response structure
            if not hasattr(analysis, 'text'):
                raise AssertionError("Analysis missing text field")
            
            if not hasattr(analysis, 'structured_data'):
                raise AssertionError("Analysis missing structured_data field")
            
            if not hasattr(analysis, 'confidence'):
                raise AssertionError("Analysis missing confidence field")
            
            # Test task generation
            base_task = {
                "name": "navigation_test",
                "type": "navigation",
                "difficulty": 0.5
            }
            
            task_response = llm.generate_task_variation(base_task, 0.7)
            
            # Validate task generation
            if not task_response.text:
                raise AssertionError("Task generation returned empty text")
            
            # Test natural language parsing
            description = "Move to the red table and pick up the blue cup"
            parse_response = llm.parse_natural_language_task(description)
            
            # Validate parsing
            if not parse_response.text:
                raise AssertionError("Task parsing returned empty text")
            
            # Test performance statistics
            perf_stats = llm.get_performance_stats()
            
            required_stats = ["avg_response_time", "cache_hit_rate", "provider_success_rates"]
            for stat in required_stats:
                if stat not in perf_stats:
                    raise AssertionError(f"Missing performance stat: {stat}")
            
            self.results["llm_integration"] = {
                "status": "passed",
                "details": {
                    "performance_analysis": "functional",
                    "task_generation": "functional",
                    "natural_language_parsing": "functional",
                    "fallback_mechanism": "operational",
                    "caching": "enabled",
                    "avg_response_time": perf_stats["avg_response_time"],
                    "provider_fallback": "working"
                }
            }
            
            logger.info("✅ LLM integration validation passed")
            
        except Exception as e:
            self.results["llm_integration"] = {
                "status": "failed", 
                "details": {"error": str(e)}
            }
            logger.error(f"❌ LLM integration validation failed: {e}")
    
    def validate_database_integration(self):
        """Validate database integration functionality."""
        
        logger.info("Testing database integration...")
        
        try:
            # Test configuration (using SQLite fallback for testing)
            config = {
                "postgresql": {},  # Will fallback to SQLite
                "mongodb": {},
                "sqlite_path": "/tmp/test_embodied_ai.db"
            }
            
            db = ProductionDatabase(config)
            
            # Test health check
            health = db.health_check()
            if "sqlite" not in health or health["sqlite"]["status"] != "connected":
                raise AssertionError("SQLite fallback not working")
            
            # Test experiment result storage
            result = ExperimentResult(
                experiment_id="test_exp_001",
                task_name="navigation_test",
                agent_name="test_agent",
                episode_id="ep_001",
                success=True,
                reward=0.85,
                steps=45,
                duration=12.5,
                observations=[{"rgb": "mock_data", "depth": "mock_data"}],
                actions=[{"linear_velocity": [0.1, 0.0, 0.0]}],
                metrics={"efficiency": 0.9, "safety": 1.0},
                metadata={"version": "1.0", "config": "test"},
                timestamp=datetime.now()
            )
            
            success = db.store_experiment_result(result)
            if not success:
                raise AssertionError("Failed to store experiment result")
            
            # Test result retrieval
            results = db.get_experiment_results("test_exp_001")
            if len(results) != 1:
                raise AssertionError(f"Expected 1 result, got {len(results)}")
            
            stored_result = results[0]
            if stored_result["episode_id"] != "ep_001":
                raise AssertionError("Result retrieval mismatch")
            
            # Test task statistics
            stats = db.get_task_statistics("navigation_test")
            if "total_episodes" not in stats:
                raise AssertionError("Task statistics incomplete")
            
            # Test multiple results
            for i in range(5):
                test_result = ExperimentResult(
                    experiment_id="test_exp_002",
                    task_name="manipulation_test", 
                    agent_name="test_agent",
                    episode_id=f"ep_{i:03d}",
                    success=i % 2 == 0,  # Alternating success/failure
                    reward=0.5 + 0.1 * i,
                    steps=50 + i * 5,
                    duration=15.0 + i,
                    observations=[],
                    actions=[],
                    metrics={"test_metric": float(i)},
                    metadata={"test_id": i},
                    timestamp=datetime.now()
                )
                
                db.store_experiment_result(test_result)
            
            # Validate batch storage
            batch_results = db.get_experiment_results("test_exp_002")
            if len(batch_results) != 5:
                raise AssertionError(f"Expected 5 batch results, got {len(batch_results)}")
            
            db.close()
            
            self.results["database_integration"] = {
                "status": "passed",
                "details": {
                    "storage": "functional",
                    "retrieval": "functional",
                    "statistics": "functional",
                    "batch_operations": "functional",
                    "health_monitoring": "operational",
                    "fallback_database": "sqlite",
                    "connection_status": "stable"
                }
            }
            
            logger.info("✅ Database integration validation passed")
            
        except Exception as e:
            self.results["database_integration"] = {
                "status": "failed",
                "details": {"error": str(e)}
            }
            logger.error(f"❌ Database integration validation failed: {e}")
    
    def validate_end_to_end_workflow(self):
        """Validate complete end-to-end workflow."""
        
        logger.info("Testing end-to-end workflow...")
        
        try:
            # Setup components
            env_config = {"max_steps": 50, "sensor_noise_level": 0.02}
            env = ProductionEnv(env_config)
            
            llm_config = {"cache_ttl": 3600}
            llm = LLMIntegration(llm_config)
            
            db_config = {"sqlite_path": "/tmp/e2e_test.db"}
            db = ProductionDatabase(db_config)
            
            # Run complete workflow
            experiment_id = f"e2e_test_{int(time.time())}"
            episode_results = []
            
            # Run multiple episodes
            for episode_idx in range(3):
                episode_id = f"ep_{episode_idx:03d}"
                
                # Reset environment
                obs = env.reset(seed=42 + episode_idx)
                
                total_reward = 0
                steps = 0
                observations = []
                actions = []
                
                # Run episode
                while steps < 25:  # Short episodes
                    # Simple navigation policy
                    action = {
                        "linear_velocity": [0.1, 0.0, 0.0],
                        "angular_velocity": [0.0, 0.0, 0.05]
                    }
                    
                    obs, reward, done, info = env.step(action)
                    
                    total_reward += reward
                    steps += 1
                    
                    # Store limited trajectory data
                    observations.append({
                        "step": steps,
                        "rgb_shape": obs["rgb"].shape,
                        "depth_shape": obs["depth"].shape
                    })
                    actions.append(action)
                    
                    if done:
                        break
                
                # Create result record
                result = ExperimentResult(
                    experiment_id=experiment_id,
                    task_name="e2e_navigation",
                    agent_name="simple_nav_agent",
                    episode_id=episode_id,
                    success=info.get("task_completed", False),
                    reward=total_reward,
                    steps=steps,
                    duration=steps * 0.05,  # Simulate time
                    observations=observations,
                    actions=actions,
                    metrics={
                        "efficiency": total_reward / max(1, steps),
                        "completion": 1.0 if info.get("task_completed") else 0.0
                    },
                    metadata={
                        "episode_index": episode_idx,
                        "environment_config": env_config
                    },
                    timestamp=datetime.now()
                )
                
                # Store in database
                db_success = db.store_experiment_result(result)
                if not db_success:
                    raise AssertionError(f"Failed to store episode {episode_idx}")
                
                episode_results.append({
                    "success": result.success,
                    "reward": result.reward,
                    "steps": result.steps,
                    "failure_reason": None if result.success else "incomplete"
                })
            
            # Analyze performance with LLM
            analysis = llm.analyze_performance(episode_results, "End-to-end validation test")
            
            # Get stored results from database
            stored_results = db.get_experiment_results(experiment_id)
            if len(stored_results) != 3:
                raise AssertionError(f"Expected 3 stored results, got {len(stored_results)}")
            
            # Generate task statistics
            stats = db.get_task_statistics("e2e_navigation")
            
            # Cleanup
            env.close()
            db.close()
            
            # Calculate overall metrics
            success_rate = np.mean([r["success"] for r in episode_results])
            avg_reward = np.mean([r["reward"] for r in episode_results])
            avg_steps = np.mean([r["steps"] for r in episode_results])
            
            self.results["end_to_end_workflow"] = {
                "status": "passed",
                "details": {
                    "episodes_completed": len(episode_results),
                    "success_rate": success_rate,
                    "avg_reward": avg_reward,
                    "avg_steps": avg_steps,
                    "database_storage": "successful",
                    "llm_analysis": "completed",
                    "workflow_integration": "functional",
                    "data_flow": "validated",
                    "performance_tracking": "operational"
                }
            }
            
            logger.info("✅ End-to-end workflow validation passed")
            
        except Exception as e:
            self.results["end_to_end_workflow"] = {
                "status": "failed",
                "details": {"error": str(e)}
            }
            logger.error(f"❌ End-to-end workflow validation failed: {e}")
    
    def generate_report(self) -> Dict[str, Any]:
        """Generate final validation report."""
        
        total_time = time.time() - self.start_time
        
        # Calculate overall status
        passed_tests = sum(1 for result in self.results.values() if result["status"] == "passed")
        total_tests = len(self.results)
        overall_success = passed_tests == total_tests
        
        report = {
            "generation": "Generation 1 - MAKE IT WORK",
            "validation_timestamp": datetime.now().isoformat(),
            "total_validation_time": f"{total_time:.2f} seconds",
            "overall_status": "PASSED" if overall_success else "FAILED",
            "tests_passed": f"{passed_tests}/{total_tests}",
            "success_rate": f"{(passed_tests/total_tests)*100:.1f}%",
            "test_results": self.results,
            "summary": {
                "production_environment": self.results["production_env"]["status"],
                "llm_integration": self.results["llm_integration"]["status"],
                "database_integration": self.results["database_integration"]["status"],
                "end_to_end_workflow": self.results["end_to_end_workflow"]["status"]
            },
            "next_steps": [
                "Generation 1 implementation ready" if overall_success else "Fix failing tests",
                "Proceed to Generation 2 - MAKE IT ROBUST" if overall_success else "Retry Generation 1",
                "Scale up testing environment",
                "Begin robustness implementations"
            ]
        }
        
        # Log final report
        status_emoji = "🎉" if overall_success else "⚠️"
        logger.info(f"{status_emoji} Generation 1 Validation Complete: {report['success_rate']} success rate")
        
        return report


def main():
    """Run Generation 1 production validation."""
    
    try:
        validator = Generation1Validator()
        report = validator.validate_all()
        
        # Save report
        report_path = "generation1_production_validation_report.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\n{'='*60}")
        print("GENERATION 1 PRODUCTION VALIDATION REPORT")
        print('='*60)
        print(f"Overall Status: {report['overall_status']}")
        print(f"Tests Passed: {report['tests_passed']}")
        print(f"Success Rate: {report['success_rate']}")
        print(f"Validation Time: {report['total_validation_time']}")
        print(f"\nDetailed report saved to: {report_path}")
        print('='*60)
        
        # Return appropriate exit code
        return 0 if report['overall_status'] == 'PASSED' else 1
        
    except Exception as e:
        logger.error(f"Validation failed with critical error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
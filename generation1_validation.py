#!/usr/bin/env python3
"""
Generation 1: Basic Functionality Validation
Tests core functionality of the Embodied AI Benchmark++ system.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import json
from datetime import datetime
import traceback

def test_core_imports():
    """Test that all core components can be imported."""
    try:
        from embodied_ai_benchmark import (
            BenchmarkSuite, BaseAgent, RandomAgent, ScriptedAgent,
            MultiAgentBenchmark, LLMCurriculum
        )
        return True, "Core imports successful"
    except Exception as e:
        return False, f"Import error: {str(e)}"

def test_basic_agent_creation():
    """Test creation of basic agents."""
    try:
        from embodied_ai_benchmark import RandomAgent, ScriptedAgent
        
        # Test RandomAgent
        random_config = {
            "agent_id": "test_random",
            "action_space": {
                "type": "continuous", 
                "shape": (7,),
                "low": [-1] * 7,
                "high": [1] * 7
            }
        }
        random_agent = RandomAgent(random_config)
        
        # Test ScriptedAgent
        script_config = {
            "agent_id": "test_scripted",
            "script": [
                {"type": "move", "values": [1, 0, 0]},
                {"type": "grasp", "values": [0.5]},
                {"type": "move", "values": [0, 1, 0]}
            ]
        }
        scripted_agent = ScriptedAgent(script_config)
        
        return True, f"Created agents: {random_agent.agent_id}, {scripted_agent.agent_id}"
    except Exception as e:
        return False, f"Agent creation error: {str(e)}"

def test_benchmark_suite_creation():
    """Test BenchmarkSuite initialization."""
    try:
        from embodied_ai_benchmark import BenchmarkSuite
        
        suite = BenchmarkSuite()
        return True, f"BenchmarkSuite created with {len(suite.available_tasks)} available tasks"
    except Exception as e:
        return False, f"BenchmarkSuite error: {str(e)}"

def test_multi_agent_benchmark():
    """Test MultiAgentBenchmark creation."""
    try:
        from embodied_ai_benchmark import MultiAgentBenchmark
        
        ma_benchmark = MultiAgentBenchmark()
        return True, "MultiAgentBenchmark created successfully"
    except Exception as e:
        return False, f"MultiAgentBenchmark error: {str(e)}"

def test_basic_agent_actions():
    """Test that agents can generate actions."""
    try:
        from embodied_ai_benchmark import RandomAgent
        
        config = {
            "agent_id": "test_agent",
            "action_space": {"type": "continuous", "shape": (3,), "low": [-1]*3, "high": [1]*3}
        }
        agent = RandomAgent(config)
        
        # Reset agent
        agent.reset()
        
        # Generate some actions
        observation = {
            "rgb": "mock_rgb_data",
            "depth": "mock_depth_data", 
            "proprioception": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
        }
        
        actions = []
        for i in range(3):
            action = agent.act(observation)
            actions.append(action)
            agent.update(observation, action, 0.1, observation, False)
        
        return True, f"Generated {len(actions)} actions successfully"
    except Exception as e:
        return False, f"Agent actions error: {str(e)}"

def run_generation1_validation():
    """Run all Generation 1 validation tests."""
    print("🚀 Running Generation 1: Basic Functionality Validation")
    print("=" * 60)
    
    tests = [
        ("Core Imports", test_core_imports),
        ("Agent Creation", test_basic_agent_creation),
        ("BenchmarkSuite", test_benchmark_suite_creation),
        ("MultiAgent", test_multi_agent_benchmark),
        ("Agent Actions", test_basic_agent_actions),
    ]
    
    results = {}
    total_tests = len(tests)
    passed_tests = 0
    
    for test_name, test_func in tests:
        print(f"\n🔍 Testing: {test_name}")
        try:
            success, message = test_func()
            if success:
                print(f"✅ PASS: {message}")
                passed_tests += 1
                results[test_name] = {"status": "PASS", "message": message}
            else:
                print(f"❌ FAIL: {message}")
                results[test_name] = {"status": "FAIL", "message": message}
        except Exception as e:
            error_msg = f"Exception during test: {str(e)}\n{traceback.format_exc()}"
            print(f"❌ ERROR: {error_msg}")
            results[test_name] = {"status": "ERROR", "message": error_msg}
    
    # Summary
    success_rate = passed_tests / total_tests
    print(f"\n{'='*60}")
    print(f"📊 GENERATION 1 VALIDATION SUMMARY:")
    print(f"Total Tests: {total_tests}")
    print(f"Passed: {passed_tests}")
    print(f"Failed: {total_tests - passed_tests}")
    print(f"Success Rate: {success_rate:.2%}")
    
    # Save results
    report = {
        "generation": 1,
        "timestamp": datetime.now().isoformat(),
        "summary": {
            "total_tests": total_tests,
            "passed_tests": passed_tests,
            "failed_tests": total_tests - passed_tests,
            "success_rate": success_rate
        },
        "test_results": results
    }
    
    with open("generation1_validation_results.json", "w") as f:
        json.dump(report, f, indent=2)
    
    if success_rate >= 0.8:
        print("🎉 Generation 1: BASIC FUNCTIONALITY - VALIDATED ✅")
        return True
    else:
        print("⚠️ Generation 1: BASIC FUNCTIONALITY - NEEDS ATTENTION ❌")
        return False

if __name__ == "__main__":
    success = run_generation1_validation()
    sys.exit(0 if success else 1)
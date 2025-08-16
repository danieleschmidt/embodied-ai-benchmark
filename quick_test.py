#!/usr/bin/env python3
"""Quick test to validate basic functionality."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_imports():
    """Test core imports work."""
    try:
        from embodied_ai_benchmark.core.base_env import BaseEnv
        from embodied_ai_benchmark.core.base_agent import BaseAgent
        from embodied_ai_benchmark.core.base_task import BaseTask
        print("✅ Core imports successful")
        return True
    except Exception as e:
        print(f"❌ Core imports failed: {e}")
        return False

def test_basic_instantiation():
    """Test basic class instantiation."""
    try:
        from embodied_ai_benchmark.core.base_agent import RandomAgent
        config = {"action_space": "continuous", "action_dim": 4}
        agent = RandomAgent(config)
        print("✅ Basic instantiation successful")
        return True
    except Exception as e:
        print(f"❌ Basic instantiation failed: {e}")
        return False

if __name__ == "__main__":
    print("🧪 Running Quick Validation Tests")
    
    results = []
    results.append(test_imports())
    results.append(test_basic_instantiation())
    
    if all(results):
        print("🎉 All quick tests passed!")
        sys.exit(0)
    else:
        print("❌ Some tests failed")
        sys.exit(1)
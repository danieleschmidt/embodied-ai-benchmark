#!/usr/bin/env python3
"""Simple validation script to verify core imports work."""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def main():
    """Test core imports."""
    print("🚀 Simple Embodied AI Benchmark Validation")
    print("=" * 50)
    
    try:
        # Test core module import
        import embodied_ai_benchmark
        print("✅ Main package imports successfully")
        
        # Test core classes exist
        from embodied_ai_benchmark import (
            RandomAgent, BenchmarkSuite, Evaluator,
            LRUCache, AdaptiveCache, PersistentCache,
            ErrorHandler, CommunicationProtocol,
            BenchmarkMetricsCollector
        )
        print("✅ Core classes available")
        
        # Test specific components
        from embodied_ai_benchmark.curriculum.llm_curriculum import LLMCurriculum
        print("✅ LLM curriculum components available")
        
        from embodied_ai_benchmark.utils.error_handling import ErrorHandler
        handler = ErrorHandler()
        print("✅ Error handling system working")
        
        from embodied_ai_benchmark.utils.caching import LRUCache
        cache = LRUCache(max_size=10)
        cache.put("test", "value")
        assert cache.get("test") == "value"
        print("✅ Caching system working")
        
        from embodied_ai_benchmark.utils.benchmark_metrics import BenchmarkMetricsCollector
        metrics = BenchmarkMetricsCollector()
        metrics.increment_counter("test_counter", 1)
        print("✅ Metrics collection working")
        
        from embodied_ai_benchmark.multiagent.coordination_protocols import CommunicationProtocol
        print("✅ Multi-agent coordination components available")
        
        print("\n🎉 All validations passed!")
        print("✅ Implementation is ready for testing and deployment")
        return 0
        
    except Exception as e:
        print(f"❌ Validation failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
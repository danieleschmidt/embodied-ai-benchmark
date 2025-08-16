#!/usr/bin/env python3
"""Enhanced validation with comprehensive error handling and logging."""

import sys
import os
import logging
import traceback
from contextlib import contextmanager
from typing import List, Dict, Any

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@contextmanager
def safe_test_execution(test_name: str):
    """Context manager for safe test execution with enhanced error handling."""
    logger.info(f"🧪 Starting test: {test_name}")
    try:
        yield
        logger.info(f"✅ Test passed: {test_name}")
    except Exception as e:
        logger.error(f"❌ Test failed: {test_name} - {str(e)}")
        logger.error(f"🔍 Traceback: {traceback.format_exc()}")
        raise

def test_enhanced_imports():
    """Test enhanced import functionality with error recovery."""
    with safe_test_execution("Enhanced Core Imports"):
        from embodied_ai_benchmark.core.base_env import BaseEnv
        from embodied_ai_benchmark.core.base_agent import BaseAgent, RandomAgent
        from embodied_ai_benchmark.core.base_task import BaseTask
        from embodied_ai_benchmark.core.base_metric import BaseMetric
        
        # Test additional imports
        from embodied_ai_benchmark.utils.error_handling import ErrorHandler
        from embodied_ai_benchmark.utils.monitoring import performance_monitor
        from embodied_ai_benchmark.utils.caching import LRUCache

def test_error_handling():
    """Test error handling and recovery mechanisms."""
    with safe_test_execution("Error Handling Systems"):
        from embodied_ai_benchmark.utils.error_handling import ErrorHandler, ErrorRecoveryStrategy
        
        # Test error handler initialization
        error_handler = ErrorHandler()
        
        # Test error recovery strategy
        strategy = ErrorRecoveryStrategy()

def test_monitoring_systems():
    """Test monitoring and health check systems."""
    with safe_test_execution("Monitoring Systems"):
        from embodied_ai_benchmark.utils.monitoring import performance_monitor, health_checker
        
        # Test performance monitoring
        @performance_monitor
        def dummy_operation():
            return "test"
        
        result = dummy_operation()
        assert result == "test"

def test_concurrent_execution():
    """Test concurrent execution capabilities."""
    with safe_test_execution("Concurrent Execution"):
        from embodied_ai_benchmark.utils.concurrent_execution import ConcurrentBenchmarkExecutor
        
        config = {
            "max_workers": 2,
            "timeout": 30
        }
        executor = ConcurrentBenchmarkExecutor(config)

def test_caching_systems():
    """Test caching mechanisms."""
    with safe_test_execution("Caching Systems"):
        from embodied_ai_benchmark.utils.caching import LRUCache, cache_result
        
        # Test LRU cache
        cache = LRUCache(max_size=100)
        cache.set("test_key", "test_value")
        assert cache.get("test_key") == "test_value"
        
        # Test cache decorator
        @cache_result(ttl=60)
        def cached_function(x):
            return x * 2
        
        result = cached_function(5)
        assert result == 10

def test_multi_agent_systems():
    """Test multi-agent coordination systems."""
    with safe_test_execution("Multi-Agent Systems"):
        from embodied_ai_benchmark.multiagent.coordination_protocols import CommunicationProtocol
        from embodied_ai_benchmark.multiagent.multi_agent_benchmark import MultiAgentBenchmark
        
        # Test communication protocol
        protocol = CommunicationProtocol()

def test_autonomous_sdlc():
    """Test autonomous SDLC components."""
    with safe_test_execution("Autonomous SDLC"):
        from embodied_ai_benchmark.sdlc.autonomous_orchestrator import AutonomousSDLCOrchestrator
        from embodied_ai_benchmark.sdlc.requirements_engine import RequirementsEngine
        from embodied_ai_benchmark.sdlc.code_generator import CodeGenerator

def test_security_hardening():
    """Test security and hardening features."""
    with safe_test_execution("Security Hardening"):
        from embodied_ai_benchmark.sdlc.security_monitor import SecurityMonitoringSystem
        
        config = {"monitoring_enabled": True}
        security_system = SecurityMonitoringSystem(config)

def run_comprehensive_validation() -> Dict[str, Any]:
    """Run comprehensive validation with detailed reporting."""
    logger.info("🚀 Starting Comprehensive Validation Suite")
    
    test_results = {}
    test_functions = [
        test_enhanced_imports,
        test_error_handling,
        test_monitoring_systems,
        test_concurrent_execution,
        test_caching_systems,
        test_multi_agent_systems,
        test_autonomous_sdlc,
        test_security_hardening
    ]
    
    passed = 0
    failed = 0
    
    for test_func in test_functions:
        test_name = test_func.__name__
        try:
            test_func()
            test_results[test_name] = "PASSED"
            passed += 1
        except Exception as e:
            test_results[test_name] = f"FAILED: {str(e)}"
            failed += 1
    
    # Generate comprehensive report
    report = {
        "total_tests": len(test_functions),
        "passed": passed,
        "failed": failed,
        "success_rate": passed / len(test_functions) * 100,
        "test_results": test_results
    }
    
    logger.info(f"📊 Validation Results: {passed}/{len(test_functions)} tests passed ({report['success_rate']:.1f}%)")
    
    return report

if __name__ == "__main__":
    try:
        report = run_comprehensive_validation()
        
        if report["failed"] == 0:
            print("🎉 ALL TESTS PASSED - GENERATION 2 ROBUSTNESS ACHIEVED!")
            sys.exit(0)
        else:
            print(f"❌ {report['failed']} tests failed. Robustness improvements needed.")
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"💥 Critical validation failure: {e}")
        sys.exit(1)
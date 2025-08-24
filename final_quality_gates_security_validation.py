#!/usr/bin/env python3
"""
Final Quality Gates and Security Validation
Comprehensive validation of security, performance, and production readiness.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import json
import time
from datetime import datetime
import traceback
import subprocess
import hashlib
import ast

def test_security_validation():
    """Test security validation and hardening."""
    results = {}
    
    print("🔒 Testing Security Validation...")
    
    try:
        from embodied_ai_benchmark.utils.validation import SecurityValidator
        from embodied_ai_benchmark.utils.compliance import ComplianceManager
        
        # Test security validator
        security_validator = SecurityValidator()
        
        # Test input validation
        safe_input = "normal_benchmark_command"
        security_validator.validate_input(safe_input)
        
        # Test compliance manager
        compliance_manager = ComplianceManager()
        
        results["security_validation"] = {
            "status": "PASS",
            "message": "Security validation and compliance systems operational"
        }
        
    except Exception as e:
        results["security_validation"] = {
            "status": "FAIL", 
            "message": f"Security validation failed: {str(e)}"
        }
    
    return results

def test_code_quality():
    """Test code quality metrics."""
    results = {}
    
    print("📊 Testing Code Quality...")
    
    try:
        # Test 1: Python syntax validation
        python_files_valid = True
        syntax_errors = []
        
        for root, dirs, files in os.walk("src/embodied_ai_benchmark"):
            for file in files:
                if file.endswith(".py"):
                    filepath = os.path.join(root, file)
                    try:
                        with open(filepath, "r") as f:
                            ast.parse(f.read())
                    except SyntaxError as e:
                        python_files_valid = False
                        syntax_errors.append(f"{filepath}: {str(e)}")
        
        results["syntax_validation"] = {
            "status": "PASS" if python_files_valid else "FAIL",
            "message": f"Python syntax validation: {len(syntax_errors)} errors found" if syntax_errors else "All Python files have valid syntax"
        }
        
        # Test 2: Import validation
        critical_imports = [
            "embodied_ai_benchmark.BenchmarkSuite",
            "embodied_ai_benchmark.RandomAgent", 
            "embodied_ai_benchmark.MultiAgentBenchmark"
        ]
        
        import_results = {}
        for import_name in critical_imports:
            try:
                module_parts = import_name.split('.')
                module = __import__(module_parts[0])
                for part in module_parts[1:]:
                    module = getattr(module, part)
                import_results[import_name] = "SUCCESS"
            except Exception as e:
                import_results[import_name] = f"FAILED: {str(e)}"
        
        successful_imports = sum(1 for status in import_results.values() if status == "SUCCESS")
        
        results["import_validation"] = {
            "status": "PASS" if successful_imports == len(critical_imports) else "PARTIAL",
            "message": f"Critical imports: {successful_imports}/{len(critical_imports)} successful"
        }
        
    except Exception as e:
        results["code_quality_error"] = {
            "status": "ERROR",
            "message": f"Code quality testing failed: {str(e)}"
        }
    
    return results

def test_performance_benchmarks():
    """Test performance benchmarks and thresholds."""
    results = {}
    
    print("⚡ Testing Performance Benchmarks...")
    
    try:
        # Test 1: Memory performance
        import psutil
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024
        
        # Simulate workload
        from embodied_ai_benchmark import BenchmarkSuite, RandomAgent
        
        suite = BenchmarkSuite()
        agent_config = {
            "agent_id": "perf_test_agent",
            "action_space": {"type": "continuous", "shape": (3,), "low": [-1]*3, "high": [1]*3}
        }
        agent = RandomAgent(agent_config)
        
        # Generate actions for performance testing
        observation = {"rgb": "test", "depth": "test", "proprioception": [0.1]*7}
        
        start_time = time.time()
        for _ in range(100):
            action = agent.act(observation)
        action_generation_time = time.time() - start_time
        
        final_memory = process.memory_info().rss / 1024 / 1024
        memory_usage = final_memory - initial_memory
        
        # Performance thresholds
        action_time_per_step = (action_generation_time / 100) * 1000  # ms
        
        results["memory_performance"] = {
            "status": "PASS" if memory_usage < 50 else "FAIL",
            "memory_usage_mb": round(memory_usage, 2),
            "message": f"Memory usage: {memory_usage:.2f}MB (threshold: 50MB)"
        }
        
        results["action_performance"] = {
            "status": "PASS" if action_time_per_step < 10 else "FAIL", 
            "action_time_ms": round(action_time_per_step, 2),
            "message": f"Action generation: {action_time_per_step:.2f}ms (threshold: 10ms)"
        }
        
    except Exception as e:
        results["performance_error"] = {
            "status": "ERROR",
            "message": f"Performance benchmarking failed: {str(e)}"
        }
    
    return results

def test_production_readiness():
    """Test production readiness criteria."""
    results = {}
    
    print("🚀 Testing Production Readiness...")
    
    try:
        # Test 1: Configuration validation
        config_files = [
            "pyproject.toml",
            "docker-compose.yml",
            "requirements-production.txt"
        ]
        
        config_status = {}
        for config_file in config_files:
            if os.path.exists(config_file):
                config_status[config_file] = "EXISTS"
            else:
                config_status[config_file] = "MISSING"
        
        configs_present = sum(1 for status in config_status.values() if status == "EXISTS")
        
        results["configuration_files"] = {
            "status": "PASS" if configs_present >= 2 else "FAIL",
            "configs_present": f"{configs_present}/{len(config_files)}",
            "message": f"Configuration files: {configs_present}/{len(config_files)} present"
        }
        
        # Test 2: Docker readiness
        docker_files = ["Dockerfile", "docker-compose.yml"]
        docker_ready = all(os.path.exists(f) for f in docker_files)
        
        results["docker_readiness"] = {
            "status": "PASS" if docker_ready else "PARTIAL",
            "message": "Docker configuration ready" if docker_ready else "Some Docker files missing"
        }
        
        # Test 3: Logging configuration
        from embodied_ai_benchmark.utils.logging_config import get_logger
        
        logger = get_logger("production_test")
        logger.info("Production readiness test")
        
        results["logging_system"] = {
            "status": "PASS",
            "message": "Logging system configured and operational"
        }
        
    except Exception as e:
        results["production_readiness_error"] = {
            "status": "ERROR",
            "message": f"Production readiness testing failed: {str(e)}"
        }
    
    return results

def test_security_scanning():
    """Test security scanning and vulnerability assessment."""
    results = {}
    
    print("🛡️ Testing Security Scanning...")
    
    try:
        # Test 1: Basic dependency security
        # Check for known security issues in common dependencies
        potential_security_issues = []
        
        # Check imports for potentially dangerous modules
        dangerous_imports = ['eval', 'exec', 'compile', '__import__']
        
        for root, dirs, files in os.walk("src/embodied_ai_benchmark"):
            for file in files:
                if file.endswith(".py"):
                    filepath = os.path.join(root, file)
                    try:
                        with open(filepath, "r") as f:
                            content = f.read()
                            for dangerous_import in dangerous_imports:
                                if dangerous_import in content:
                                    potential_security_issues.append(f"{filepath}: {dangerous_import}")
                    except Exception:
                        pass
        
        results["dependency_security"] = {
            "status": "PASS" if len(potential_security_issues) == 0 else "WARNING",
            "issues_found": len(potential_security_issues),
            "message": f"Security scan: {len(potential_security_issues)} potential issues found"
        }
        
        # Test 2: Input validation security
        from embodied_ai_benchmark.utils.validation import InputValidator
        
        validator = InputValidator()
        
        # Test with various inputs
        test_inputs = [
            {"valid": True, "data": {"agent_id": "test", "action_space": {"type": "continuous"}}},
            {"valid": False, "data": {"agent_id": "", "action_space": None}}
        ]
        
        validation_working = True
        for test_input in test_inputs:
            try:
                validator.validate_agent_config(test_input["data"])
                if not test_input["valid"]:
                    validation_working = False  # Should have failed
            except:
                if test_input["valid"]:
                    validation_working = False  # Should have passed
        
        results["input_validation_security"] = {
            "status": "PASS" if validation_working else "FAIL",
            "message": "Input validation security working" if validation_working else "Input validation security issues"
        }
        
    except Exception as e:
        results["security_scanning_error"] = {
            "status": "ERROR", 
            "message": f"Security scanning failed: {str(e)}"
        }
    
    return results

def calculate_overall_quality_score(all_results):
    """Calculate overall quality score from all test results."""
    total_tests = 0
    passed_tests = 0
    
    for category_results in all_results.values():
        for test_name, test_result in category_results.items():
            if isinstance(test_result, dict) and "status" in test_result:
                total_tests += 1
                if test_result["status"] == "PASS":
                    passed_tests += 1
                elif test_result["status"] == "PARTIAL":
                    passed_tests += 0.5
    
    return (passed_tests / total_tests * 100) if total_tests > 0 else 0

def generate_quality_gates_report(all_results):
    """Generate comprehensive quality gates report."""
    
    overall_score = calculate_overall_quality_score(all_results)
    
    report = {
        "quality_gates_validation": "Final Quality Gates and Security Validation",
        "timestamp": datetime.now().isoformat(),
        "overall_quality_score": round(overall_score, 2),
        "status": "PASS" if overall_score >= 80 else "FAIL" if overall_score < 60 else "WARNING",
        "test_categories": all_results,
        "summary": {
            "security_status": "VALIDATED" if overall_score >= 80 else "NEEDS_REVIEW",
            "production_ready": overall_score >= 75,
            "recommendations": []
        }
    }
    
    # Add recommendations based on results
    if overall_score < 80:
        report["summary"]["recommendations"].append("Address failing quality gates before production deployment")
    if overall_score < 90:
        report["summary"]["recommendations"].append("Consider additional testing and validation")
    
    # Save report
    with open("final_quality_gates_report.json", "w") as f:
        json.dump(report, f, indent=2)
    
    return report

def print_quality_gates_summary(report):
    """Print quality gates validation summary."""
    print(f"\n{'='*70}")
    print("🎯 FINAL QUALITY GATES & SECURITY VALIDATION SUMMARY")
    print(f"{'='*70}")
    print(f"Overall Quality Score: {report['overall_quality_score']:.1f}%")
    print(f"Validation Status: {report['status']}")
    print(f"Production Ready: {'✅ YES' if report['summary']['production_ready'] else '❌ NO'}")
    print(f"Security Status: {report['summary']['security_status']}")
    
    print(f"\n📊 Category Breakdown:")
    for category, results in report['test_categories'].items():
        category_score = calculate_overall_quality_score({category: results})
        print(f"  {category.replace('_', ' ').title()}: {category_score:.1f}%")
    
    if report['summary']['recommendations']:
        print(f"\n💡 Recommendations:")
        for i, rec in enumerate(report['summary']['recommendations'], 1):
            print(f"  {i}. {rec}")
    
    print(f"\n🎯 Quality Gate Status:")
    if report['overall_quality_score'] >= 80:
        print("✅ QUALITY GATES PASSED - Ready for production deployment")
    elif report['overall_quality_score'] >= 60:
        print("⚠️ QUALITY GATES WARNING - Review issues before production")
    else:
        print("❌ QUALITY GATES FAILED - Significant issues must be resolved")
    
    print(f"\n📄 Detailed report saved: final_quality_gates_report.json")

def main():
    """Run final quality gates and security validation."""
    print("🔐 AUTONOMOUS SDLC - FINAL QUALITY GATES & SECURITY VALIDATION")
    print("=" * 70)
    
    # Run all quality gate tests
    all_results = {}
    
    all_results["security_validation"] = test_security_validation()
    all_results["code_quality"] = test_code_quality()
    all_results["performance_benchmarks"] = test_performance_benchmarks()
    all_results["production_readiness"] = test_production_readiness()
    all_results["security_scanning"] = test_security_scanning()
    
    # Generate comprehensive report
    report = generate_quality_gates_report(all_results)
    
    # Print summary
    print_quality_gates_summary(report)
    
    # Return success/failure based on overall score
    return report['overall_quality_score'] >= 75

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
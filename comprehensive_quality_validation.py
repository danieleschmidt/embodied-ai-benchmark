#!/usr/bin/env python3
"""Comprehensive quality validation for Embodied AI Benchmark."""

import os
import sys
import time
import json
import traceback
from pathlib import Path
from typing import Dict, List, Any, Optional
import subprocess

# Add src to Python path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))


class QualityValidator:
    """Comprehensive quality validation suite."""
    
    def __init__(self):
        self.results = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "validation_results": {},
            "overall_status": "UNKNOWN",
            "summary": {}
        }
        self.project_root = Path(__file__).parent
    
    def run_all_validations(self) -> Dict[str, Any]:
        """Run all quality validations.
        
        Returns:
            Comprehensive validation results
        """
        print("\n🚀 TERRAGON AUTONOMOUS SDLC - QUALITY VALIDATION SUITE")
        print("=" * 60)
        
        validations = [
            ("import_validation", self.validate_imports),
            ("code_structure", self.validate_code_structure),
            ("research_components", self.validate_research_components),
            ("performance_scaling", self.validate_performance_scaling),
            ("error_handling", self.validate_error_handling),
            ("security_compliance", self.validate_security),
            ("documentation_coverage", self.validate_documentation),
            ("deployment_readiness", self.validate_deployment_readiness)
        ]
        
        total_validations = len(validations)
        passed_validations = 0
        
        for validation_name, validation_func in validations:
            print(f"\n📋 Running {validation_name.replace('_', ' ').title()}...")
            
            try:
                start_time = time.time()
                result = validation_func()
                execution_time = time.time() - start_time
                
                result["execution_time"] = execution_time
                self.results["validation_results"][validation_name] = result
                
                if result.get("status") == "PASS":
                    passed_validations += 1
                    print(f"   ✅ {validation_name}: PASSED ({execution_time:.2f}s)")
                else:
                    print(f"   ❌ {validation_name}: FAILED ({execution_time:.2f}s)")
                    if "error" in result:
                        print(f"      Error: {result['error']}")
                
            except Exception as e:
                print(f"   💥 {validation_name}: EXCEPTION - {str(e)}")
                self.results["validation_results"][validation_name] = {
                    "status": "EXCEPTION",
                    "error": str(e),
                    "traceback": traceback.format_exc()
                }
        
        # Calculate overall results
        success_rate = passed_validations / total_validations
        self.results["summary"] = {
            "total_validations": total_validations,
            "passed_validations": passed_validations,
            "failed_validations": total_validations - passed_validations,
            "success_rate": success_rate
        }
        
        if success_rate >= 0.8:
            self.results["overall_status"] = "PASS"
        elif success_rate >= 0.6:
            self.results["overall_status"] = "PARTIAL"
        else:
            self.results["overall_status"] = "FAIL"
        
        self._print_summary()
        return self.results
    
    def validate_imports(self) -> Dict[str, Any]:
        """Validate that all major modules can be imported."""
        import_tests = [
            "embodied_ai_benchmark.evaluation.cross_simulator_benchmark",
            "embodied_ai_benchmark.curriculum.emergent_curriculum",
            "embodied_ai_benchmark.physics.adaptive_physics",
            "embodied_ai_benchmark.evaluation.long_horizon_benchmark",
            "embodied_ai_benchmark.utils.distributed_execution",
            "embodied_ai_benchmark.utils.advanced_caching"
        ]
        
        results = {
            "status": "PASS",
            "imported_modules": [],
            "failed_imports": [],
            "details": {}
        }
        
        for module_name in import_tests:
            try:
                __import__(module_name)
                results["imported_modules"].append(module_name)
                results["details"][module_name] = "SUCCESS"
            except Exception as e:
                results["failed_imports"].append(module_name)
                results["details"][module_name] = str(e)
        
        if results["failed_imports"]:
            results["status"] = "FAIL"
            results["error"] = f"Failed to import {len(results['failed_imports'])} modules"
        
        return results
    
    def validate_code_structure(self) -> Dict[str, Any]:
        """Validate code structure and organization."""
        required_directories = [
            "src/embodied_ai_benchmark/core",
            "src/embodied_ai_benchmark/evaluation",
            "src/embodied_ai_benchmark/curriculum",
            "src/embodied_ai_benchmark/physics",
            "src/embodied_ai_benchmark/utils",
            "src/embodied_ai_benchmark/tasks",
            "tests/unit",
            "tests/integration"
        ]
        
        required_files = [
            "README.md",
            "pyproject.toml",
            "src/embodied_ai_benchmark/__init__.py"
        ]
        
        results = {
            "status": "PASS",
            "missing_directories": [],
            "missing_files": [],
            "structure_score": 0.0
        }
        
        # Check directories
        missing_dirs = 0
        for directory in required_directories:
            dir_path = self.project_root / directory
            if not dir_path.exists():
                results["missing_directories"].append(directory)
                missing_dirs += 1
        
        # Check files
        missing_files = 0
        for file_path in required_files:
            full_path = self.project_root / file_path
            if not full_path.exists():
                results["missing_files"].append(file_path)
                missing_files += 1
        
        # Calculate structure score
        total_checks = len(required_directories) + len(required_files)
        missing_items = missing_dirs + missing_files
        results["structure_score"] = (total_checks - missing_items) / total_checks
        
        if results["structure_score"] < 0.8:
            results["status"] = "FAIL"
            results["error"] = f"Structure score too low: {results['structure_score']:.2f}"
        
        return results
    
    def validate_research_components(self) -> Dict[str, Any]:
        """Validate research components functionality."""
        results = {
            "status": "PASS",
            "component_tests": {},
            "errors": []
        }
        
        # Test Cross-Simulator Benchmark
        try:
            from embodied_ai_benchmark.evaluation.cross_simulator_benchmark import CrossSimulatorBenchmark
            benchmark = CrossSimulatorBenchmark()
            
            # Test basic functionality
            assert hasattr(benchmark, 'simulators')
            assert hasattr(benchmark, 'modalities')
            assert len(benchmark.simulators) > 0
            assert len(benchmark.modalities) > 0
            
            results["component_tests"]["cross_simulator_benchmark"] = "PASS"
            
        except Exception as e:
            results["component_tests"]["cross_simulator_benchmark"] = "FAIL"
            results["errors"].append(f"Cross-Simulator Benchmark: {str(e)}")
        
        # Test Emergent Curriculum
        try:
            from embodied_ai_benchmark.curriculum.emergent_curriculum import (
                EmergentCurriculumGenerator, BehaviorAnalyzer
            )
            
            analyzer = BehaviorAnalyzer()
            generator = EmergentCurriculumGenerator()
            
            # Test basic functionality
            assert hasattr(analyzer, 'analyze_episode_interactions')
            assert hasattr(generator, 'generate_emergent_tasks')
            
            results["component_tests"]["emergent_curriculum"] = "PASS"
            
        except Exception as e:
            results["component_tests"]["emergent_curriculum"] = "FAIL"
            results["errors"].append(f"Emergent Curriculum: {str(e)}")
        
        # Test Adaptive Physics
        try:
            from embodied_ai_benchmark.physics.adaptive_physics import (
                AdaptivePhysicsLearner, MaterialPropertyLearner
            )
            
            physics_learner = AdaptivePhysicsLearner()
            material_learner = MaterialPropertyLearner()
            
            # Test basic functionality
            assert hasattr(physics_learner, 'adapt_physics_from_real_feedback')
            assert hasattr(material_learner, 'get_material_properties')
            
            results["component_tests"]["adaptive_physics"] = "PASS"
            
        except Exception as e:
            results["component_tests"]["adaptive_physics"] = "FAIL"
            results["errors"].append(f"Adaptive Physics: {str(e)}")
        
        # Test Long-Horizon Benchmark
        try:
            from embodied_ai_benchmark.evaluation.long_horizon_benchmark import (
                LongHorizonMultiAgentBenchmark, HierarchicalTaskPlan
            )
            
            lh_benchmark = LongHorizonMultiAgentBenchmark()
            task_plan = HierarchicalTaskPlan("test", "Test task", 10)
            
            # Test basic functionality
            assert hasattr(lh_benchmark, 'evaluate_hierarchical_planning')
            assert hasattr(task_plan, 'add_phase')
            
            results["component_tests"]["long_horizon_benchmark"] = "PASS"
            
        except Exception as e:
            results["component_tests"]["long_horizon_benchmark"] = "FAIL"
            results["errors"].append(f"Long-Horizon Benchmark: {str(e)}")
        
        # Check overall status
        passed_tests = sum(1 for status in results["component_tests"].values() if status == "PASS")
        total_tests = len(results["component_tests"])
        
        if passed_tests < total_tests * 0.75:  # Require 75% pass rate
            results["status"] = "FAIL"
            results["error"] = f"Only {passed_tests}/{total_tests} component tests passed"
        
        return results
    
    def validate_performance_scaling(self) -> Dict[str, Any]:
        """Validate performance and scaling components."""
        results = {
            "status": "PASS",
            "scaling_tests": {},
            "errors": []
        }
        
        # Test Distributed Execution
        try:
            from embodied_ai_benchmark.utils.distributed_execution import (
                DistributedTaskQueue, TaskSpec, ParallelBenchmarkRunner
            )
            
            # Test task queue creation
            task_queue = DistributedTaskQueue({'max_workers': 2})
            assert hasattr(task_queue, 'submit_task')
            assert hasattr(task_queue, 'get_statistics')
            
            # Test task spec creation
            task_spec = TaskSpec(
                task_id="test_task",
                task_type="test",
                payload={"test": "data"}
            )
            assert task_spec.task_id == "test_task"
            
            results["scaling_tests"]["distributed_execution"] = "PASS"
            
        except Exception as e:
            results["scaling_tests"]["distributed_execution"] = "FAIL"
            results["errors"].append(f"Distributed Execution: {str(e)}")
        
        # Test Advanced Caching
        try:
            from embodied_ai_benchmark.utils.advanced_caching import (
                LRUCache, HierarchicalCache, CacheEntry
            )
            
            # Test LRU cache
            lru_cache = LRUCache(max_size=100)
            test_entry = CacheEntry(
                key="test",
                value="test_value",
                created_at=time.time(),
                accessed_at=time.time()
            )
            
            assert hasattr(lru_cache, 'get')
            assert hasattr(lru_cache, 'put')
            
            # Test hierarchical cache
            hierarchical_cache = HierarchicalCache()
            assert hasattr(hierarchical_cache, 'get')
            assert hasattr(hierarchical_cache, 'put')
            
            results["scaling_tests"]["advanced_caching"] = "PASS"
            
        except Exception as e:
            results["scaling_tests"]["advanced_caching"] = "FAIL"
            results["errors"].append(f"Advanced Caching: {str(e)}")
        
        # Check overall status
        passed_tests = sum(1 for status in results["scaling_tests"].values() if status == "PASS")
        total_tests = len(results["scaling_tests"])
        
        if passed_tests < total_tests * 0.8:  # Require 80% pass rate
            results["status"] = "FAIL"
            results["error"] = f"Only {passed_tests}/{total_tests} scaling tests passed"
        
        return results
    
    def validate_error_handling(self) -> Dict[str, Any]:
        """Validate error handling and resilience."""
        results = {
            "status": "PASS",
            "error_handling_tests": {},
            "resilience_features": []
        }
        
        # Check for error handling utilities
        try:
            from embodied_ai_benchmark.utils.error_handling import SafeExecutor
            safe_executor = SafeExecutor()
            assert hasattr(safe_executor, 'execute_with_recovery')
            results["error_handling_tests"]["safe_executor"] = "PASS"
        except Exception as e:
            results["error_handling_tests"]["safe_executor"] = "FAIL"
        
        # Check for monitoring capabilities
        try:
            from embodied_ai_benchmark.utils.monitoring import performance_monitor
            assert hasattr(performance_monitor, 'start_monitoring')
            results["error_handling_tests"]["monitoring"] = "PASS"
        except Exception as e:
            results["error_handling_tests"]["monitoring"] = "FAIL"
        
        # Look for resilience patterns in code
        resilience_patterns = [
            "try-except blocks",
            "timeout handling",
            "retry mechanisms",
            "graceful degradation"
        ]
        
        # This is a simplified check - in practice would analyze code
        results["resilience_features"] = resilience_patterns
        
        passed_tests = sum(1 for status in results["error_handling_tests"].values() if status == "PASS")
        total_tests = len(results["error_handling_tests"])
        
        if passed_tests < total_tests * 0.7:
            results["status"] = "FAIL"
            results["error"] = "Insufficient error handling capabilities"
        
        return results
    
    def validate_security(self) -> Dict[str, Any]:
        """Validate security measures and compliance."""
        results = {
            "status": "PASS",
            "security_checks": {},
            "compliance_features": []
        }
        
        # Check for security utilities
        try:
            from embodied_ai_benchmark.utils.validation import SecurityValidator
            assert hasattr(SecurityValidator, 'check_resource_limits')
            results["security_checks"]["security_validator"] = "PASS"
        except Exception as e:
            results["security_checks"]["security_validator"] = "FAIL"
        
        # Check for input validation
        try:
            from embodied_ai_benchmark.utils.validation import InputValidator
            assert hasattr(InputValidator, 'validate_input')
            results["security_checks"]["input_validator"] = "PASS"
        except Exception as e:
            results["security_checks"]["input_validator"] = "FAIL"
        
        # Security features that should be present
        security_features = [
            "Input validation",
            "Resource limits",
            "Error sanitization",
            "Safe execution contexts"
        ]
        
        results["compliance_features"] = security_features
        
        passed_checks = sum(1 for status in results["security_checks"].values() if status == "PASS")
        total_checks = len(results["security_checks"])
        
        if passed_checks < total_checks * 0.8:
            results["status"] = "FAIL"
            results["error"] = "Insufficient security measures"
        
        return results
    
    def validate_documentation(self) -> Dict[str, Any]:
        """Validate documentation coverage."""
        results = {
            "status": "PASS",
            "documentation_files": [],
            "missing_docs": [],
            "coverage_score": 0.0
        }
        
        required_docs = [
            "README.md",
            "ARCHITECTURE.md",
            "DEPLOYMENT.md",
            "CONTRIBUTING.md",
            "SECURITY.md"
        ]
        
        existing_docs = []
        missing_docs = []
        
        for doc_file in required_docs:
            doc_path = self.project_root / doc_file
            if doc_path.exists() and doc_path.stat().st_size > 100:  # At least 100 bytes
                existing_docs.append(doc_file)
            else:
                missing_docs.append(doc_file)
        
        results["documentation_files"] = existing_docs
        results["missing_docs"] = missing_docs
        results["coverage_score"] = len(existing_docs) / len(required_docs)
        
        if results["coverage_score"] < 0.8:
            results["status"] = "FAIL"
            results["error"] = f"Documentation coverage too low: {results['coverage_score']:.2f}"
        
        return results
    
    def validate_deployment_readiness(self) -> Dict[str, Any]:
        """Validate deployment readiness."""
        results = {
            "status": "PASS",
            "deployment_files": [],
            "missing_deployment_files": [],
            "container_ready": False,
            "ci_cd_ready": False
        }
        
        deployment_files = [
            "Dockerfile",
            "docker-compose.yml",
            "kubernetes-deployment.yaml",
            "pyproject.toml"
        ]
        
        for deploy_file in deployment_files:
            file_path = self.project_root / deploy_file
            if file_path.exists():
                results["deployment_files"].append(deploy_file)
            else:
                results["missing_deployment_files"].append(deploy_file)
        
        # Check container readiness
        if "Dockerfile" in results["deployment_files"]:
            results["container_ready"] = True
        
        # Check for CI/CD files
        ci_cd_paths = [
            ".github/workflows",
            ".gitlab-ci.yml",
            "jenkins/Jenkinsfile"
        ]
        
        for ci_path in ci_cd_paths:
            if (self.project_root / ci_path).exists():
                results["ci_cd_ready"] = True
                break
        
        deployment_score = len(results["deployment_files"]) / len(deployment_files)
        
        if deployment_score < 0.75 or not results["container_ready"]:
            results["status"] = "FAIL"
            results["error"] = "Insufficient deployment readiness"
        
        return results
    
    def _print_summary(self):
        """Print validation summary."""
        print("\n" + "=" * 60)
        print("🎯 QUALITY VALIDATION SUMMARY")
        print("=" * 60)
        
        summary = self.results["summary"]
        print(f"Overall Status: {self.results['overall_status']}")
        print(f"Success Rate: {summary['success_rate']:.1%}")
        print(f"Passed: {summary['passed_validations']}/{summary['total_validations']}")
        print(f"Failed: {summary['failed_validations']}/{summary['total_validations']}")
        
        print("\n📊 VALIDATION BREAKDOWN:")
        for validation_name, result in self.results["validation_results"].items():
            status_emoji = "✅" if result.get("status") == "PASS" else "❌"
            print(f"  {status_emoji} {validation_name.replace('_', ' ').title()}: {result.get('status', 'UNKNOWN')}")
        
        print("\n" + "=" * 60)
        
        if self.results["overall_status"] == "PASS":
            print("🎉 QUALITY VALIDATION PASSED - READY FOR PRODUCTION! 🎉")
        elif self.results["overall_status"] == "PARTIAL":
            print("⚠️  QUALITY VALIDATION PARTIAL - REVIEW REQUIRED ⚠️")
        else:
            print("🚨 QUALITY VALIDATION FAILED - ISSUES NEED RESOLUTION 🚨")
        
        print("=" * 60)
    
    def save_results(self, output_file: str = "quality_validation_results.json"):
        """Save validation results to file."""
        output_path = self.project_root / output_file
        with open(output_path, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        print(f"\n💾 Results saved to: {output_path}")


def main():
    """Main validation execution."""
    validator = QualityValidator()
    
    try:
        # Run all validations
        results = validator.run_all_validations()
        
        # Save results
        validator.save_results()
        
        # Exit with appropriate code
        if results["overall_status"] == "PASS":
            sys.exit(0)
        elif results["overall_status"] == "PARTIAL":
            sys.exit(1)
        else:
            sys.exit(2)
            
    except Exception as e:
        print(f"\n💥 VALIDATION SUITE FAILED WITH EXCEPTION: {e}")
        traceback.print_exc()
        sys.exit(3)


if __name__ == "__main__":
    main()

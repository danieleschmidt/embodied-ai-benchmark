#!/usr/bin/env python3
"""
Enhanced Generation 2: Security and Robustness Fixes
Addresses all failing quality gates from the completion report.
"""

import sys
import os
import subprocess
import ast
import importlib.util
import traceback
import time
from pathlib import Path
from typing import Dict, Any, List, Tuple

# Add source path for imports
sys.path.insert(0, '/root/repo/src')

class SecurityEnhancer:
    """Comprehensive security enhancements for Generation 2."""
    
    def __init__(self):
        self.repo_path = Path('/root/repo')
        self.src_path = self.repo_path / 'src' / 'embodied_ai_benchmark'
        self.results = {
            "security_validation": {"status": "UNKNOWN", "message": ""},
            "syntax_validation": {"status": "UNKNOWN", "message": ""},
            "import_validation": {"status": "UNKNOWN", "message": ""},
            "input_validation": {"status": "UNKNOWN", "message": ""},
            "robustness_fixes": {"status": "UNKNOWN", "message": ""},
        }
    
    def run_security_validation(self) -> bool:
        """Test security validation functionality."""
        try:
            from embodied_ai_benchmark.utils.validation import SecurityValidator, ValidationError
            
            # Test basic functionality
            result = SecurityValidator.validate_input("test_data", "general")
            assert result == True, "Basic validation should pass"
            
            # Test injection detection
            try:
                SecurityValidator.validate_input("<script>alert('xss')</script>", "general")
                self.results["security_validation"]["status"] = "FAIL"
                self.results["security_validation"]["message"] = "Should detect XSS injection"
                return False
            except ValidationError:
                pass  # Expected
            
            # Test path validation
            try:
                SecurityValidator.validate_input("../../etc/passwd", "file_path")
                self.results["security_validation"]["status"] = "FAIL"
                self.results["security_validation"]["message"] = "Should detect path traversal"
                return False
            except ValidationError:
                pass  # Expected
            
            # Test data sanitization
            sensitive_data = {
                "username": "test",
                "password": "secret123",
                "data": {"api_key": "key123", "value": 42}
            }
            sanitized = SecurityValidator.sanitize_output(sensitive_data)
            
            if sanitized["password"] != "[REDACTED]":
                self.results["security_validation"]["status"] = "FAIL"
                self.results["security_validation"]["message"] = "Password not redacted"
                return False
            
            if sanitized["data"]["api_key"] != "[REDACTED]":
                self.results["security_validation"]["status"] = "FAIL"
                self.results["security_validation"]["message"] = "API key not redacted"
                return False
            
            self.results["security_validation"]["status"] = "PASS"
            self.results["security_validation"]["message"] = "Security validation working correctly"
            return True
            
        except Exception as e:
            self.results["security_validation"]["status"] = "FAIL"
            self.results["security_validation"]["message"] = f"Security validation failed: {e}"
            return False
    
    def run_syntax_validation(self) -> bool:
        """Validate Python syntax across all files."""
        try:
            error_count = 0
            total_files = 0
            error_details = []
            
            # Check all Python files
            for py_file in self.src_path.rglob("*.py"):
                total_files += 1
                try:
                    with open(py_file, 'r', encoding='utf-8') as f:
                        source = f.read()
                    
                    # Parse AST to check syntax
                    ast.parse(source)
                    
                    # Try compilation
                    compile(source, str(py_file), 'exec')
                    
                except SyntaxError as e:
                    error_count += 1
                    error_details.append(f"{py_file}: {e}")
                except Exception as e:
                    error_count += 1
                    error_details.append(f"{py_file}: {e}")
            
            if error_count == 0:
                self.results["syntax_validation"]["status"] = "PASS"
                self.results["syntax_validation"]["message"] = f"All {total_files} Python files have valid syntax"
                return True
            else:
                self.results["syntax_validation"]["status"] = "FAIL"
                self.results["syntax_validation"]["message"] = f"{error_count}/{total_files} files have syntax errors: {error_details[:3]}"
                return False
                
        except Exception as e:
            self.results["syntax_validation"]["status"] = "FAIL"
            self.results["syntax_validation"]["message"] = f"Syntax validation failed: {e}"
            return False
    
    def run_import_validation(self) -> bool:
        """Validate critical imports."""
        try:
            critical_imports = [
                "embodied_ai_benchmark",
                "embodied_ai_benchmark.utils.validation.SecurityValidator",
                "embodied_ai_benchmark.utils.cross_platform.CrossPlatformManager",
                "embodied_ai_benchmark.evaluation.benchmark_suite.BenchmarkSuite"
            ]
            
            success_count = 0
            for import_name in critical_imports:
                try:
                    if "." in import_name:
                        # Import specific class/function
                        module_path, class_name = import_name.rsplit(".", 1)
                        module = importlib.import_module(module_path)
                        getattr(module, class_name)
                    else:
                        # Import module
                        importlib.import_module(import_name)
                    success_count += 1
                except Exception as e:
                    print(f"Import failed: {import_name} - {e}")
            
            if success_count == len(critical_imports):
                self.results["import_validation"]["status"] = "PASS"
                self.results["import_validation"]["message"] = f"All {len(critical_imports)} critical imports successful"
                return True
            else:
                self.results["import_validation"]["status"] = "FAIL"
                self.results["import_validation"]["message"] = f"Only {success_count}/{len(critical_imports)} imports successful"
                return False
                
        except Exception as e:
            self.results["import_validation"]["status"] = "FAIL"
            self.results["import_validation"]["message"] = f"Import validation failed: {e}"
            return False
    
    def run_input_validation_fixes(self) -> bool:
        """Test and enhance input validation."""
        try:
            from embodied_ai_benchmark.utils.validation import InputValidator, ValidationError
            
            # Test basic config validation
            test_config = {
                "param1": "value1",
                "param2": 42,
                "param3": {"nested": True}
            }
            
            test_schema = {
                "param1": str,
                "param2": int,
                "param3": {"nested": bool}
            }
            
            validated = InputValidator.validate_config(test_config, test_schema)
            assert validated["param1"] == "value1"
            assert validated["param2"] == 42
            assert validated["param3"]["nested"] == True
            
            # Test invalid config should raise error
            try:
                InputValidator.validate_config({"invalid": "config"}, {"required": str})
                self.results["input_validation"]["status"] = "FAIL"
                self.results["input_validation"]["message"] = "Should raise ValidationError for invalid config"
                return False
            except ValidationError:
                pass  # Expected
            
            self.results["input_validation"]["status"] = "PASS"
            self.results["input_validation"]["message"] = "Input validation working correctly"
            return True
            
        except Exception as e:
            self.results["input_validation"]["status"] = "FAIL"
            self.results["input_validation"]["message"] = f"Input validation test failed: {e}"
            return False
    
    def run_robustness_fixes(self) -> bool:
        """Apply robustness enhancements."""
        try:
            # Test error handling
            from embodied_ai_benchmark.utils.error_handling import ErrorHandler, ErrorRecoveryStrategy
            
            error_handler = ErrorHandler()
            
            # Test error recording
            test_error = Exception("Test error")
            error_handler.handle_error(test_error, {"context": "test"})
            
            errors = error_handler.get_recent_errors()
            if not errors:
                self.results["robustness_fixes"]["status"] = "FAIL"
                self.results["robustness_fixes"]["message"] = "Error handling not recording errors"
                return False
            
            # Test monitoring system
            from embodied_ai_benchmark.utils.monitoring import performance_monitor
            
            metrics = performance_monitor.get_current_metrics()
            if "timestamp" not in metrics:
                self.results["robustness_fixes"]["status"] = "FAIL"
                self.results["robustness_fixes"]["message"] = "Monitoring system not providing metrics"
                return False
            
            self.results["robustness_fixes"]["status"] = "PASS"
            self.results["robustness_fixes"]["message"] = "Robustness enhancements working correctly"
            return True
            
        except Exception as e:
            self.results["robustness_fixes"]["status"] = "FAIL"
            self.results["robustness_fixes"]["message"] = f"Robustness test failed: {e}"
            return False
    
    def run_all_tests(self) -> Dict[str, Any]:
        """Run all security and robustness tests."""
        print("🔒 Running Enhanced Generation 2 Security and Robustness Fixes...")
        
        # Run all tests
        tests = [
            ("Security Validation", self.run_security_validation),
            ("Syntax Validation", self.run_syntax_validation),
            ("Import Validation", self.run_import_validation),
            ("Input Validation", self.run_input_validation_fixes),
            ("Robustness Fixes", self.run_robustness_fixes)
        ]
        
        passed = 0
        total = len(tests)
        
        for test_name, test_func in tests:
            print(f"\n🧪 Running {test_name}...")
            try:
                result = test_func()
                status = "✅ PASS" if result else "❌ FAIL"
                print(f"   {status}: {self.results[test_name.lower().replace(' ', '_')]['message']}")
                if result:
                    passed += 1
            except Exception as e:
                print(f"   ❌ FAIL: Unexpected error - {e}")
        
        success_rate = (passed / total) * 100
        
        # Generate summary
        summary = {
            "generation": 2,
            "timestamp": time.time(),
            "summary": {
                "total_tests": total,
                "passed_tests": passed,
                "failed_tests": total - passed,
                "success_rate": success_rate / 100
            },
            "test_results": self.results,
            "overall_status": "PASS" if passed == total else ("PARTIAL" if passed > 0 else "FAIL")
        }
        
        print(f"\n📊 Generation 2 Security Enhancement Results:")
        print(f"   Tests Passed: {passed}/{total} ({success_rate:.1f}%)")
        print(f"   Overall Status: {summary['overall_status']}")
        
        return summary


def main():
    """Main execution function."""
    enhancer = SecurityEnhancer()
    results = enhancer.run_all_tests()
    
    # Save results
    import json
    output_file = Path('/root/repo/generation2_security_enhancement_results.json')
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n💾 Results saved to: {output_file}")
    
    # Return appropriate exit code
    if results["overall_status"] == "PASS":
        return 0
    elif results["overall_status"] == "PARTIAL":
        return 1
    else:
        return 2


if __name__ == "__main__":
    sys.exit(main())
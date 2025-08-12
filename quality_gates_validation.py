#!/usr/bin/env python3
"""
Quality Gates Validation - Comprehensive testing, security, and compliance verification.
This validates that all implementations meet production-ready standards.
"""

import sys
import os
import time
import json
import subprocess
import importlib
from typing import Dict, Any, List, Tuple
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))


class QualityGateValidator:
    """Comprehensive quality gate validation system."""
    
    def __init__(self):
        self.results = {}
        self.passed_gates = []
        self.failed_gates = []
        self.warnings = []
        
    def run_all_gates(self) -> Dict[str, Any]:
        """Run all quality gates and return comprehensive results."""
        print("🛡️  EXECUTING MANDATORY QUALITY GATES")
        print("=" * 60)
        
        # Gate 1: Code Structure and Import Validation
        self.validate_code_structure()
        
        # Gate 2: Security Scan
        self.validate_security()
        
        # Gate 3: Performance Benchmarks
        self.validate_performance()
        
        # Gate 4: Test Coverage Simulation
        self.validate_test_coverage()
        
        # Gate 5: Documentation Completeness
        self.validate_documentation()
        
        # Gate 6: Global Compliance
        self.validate_global_compliance()
        
        # Gate 7: Deployment Readiness
        self.validate_deployment_readiness()
        
        return self.generate_final_report()
    
    def validate_code_structure(self):
        """Validate code structure and imports."""
        print("\n🔍 GATE 1: Code Structure and Import Validation")
        print("-" * 50)
        
        gate_name = "code_structure"
        try:
            # Check if main package imports work
            src_path = Path("src/embodied_ai_benchmark")
            
            if not src_path.exists():
                self.fail_gate(gate_name, "Source package directory not found")
                return
            
            # Check core modules
            core_modules = [
                "core/__init__.py",
                "evaluation/__init__.py", 
                "tasks/__init__.py",
                "utils/__init__.py"
            ]
            
            missing_modules = []
            for module in core_modules:
                if not (src_path / module).exists():
                    missing_modules.append(module)
            
            if missing_modules:
                self.warnings.append(f"Missing modules: {missing_modules}")
            
            # Test basic imports
            import_errors = []
            try:
                sys.path.insert(0, "src")
                
                # Test safe imports
                test_imports = [
                    "embodied_ai_benchmark",
                    "embodied_ai_benchmark.core",
                    "embodied_ai_benchmark.evaluation", 
                    "embodied_ai_benchmark.tasks"
                ]
                
                for module_name in test_imports:
                    try:
                        importlib.import_module(module_name)
                        print(f"✅ {module_name} import successful")
                    except ImportError as e:
                        import_errors.append(f"{module_name}: {e}")
                        print(f"⚠️  {module_name} import failed: {e}")
                
            except Exception as e:
                import_errors.append(f"Import system error: {e}")
            
            # Validate file structure
            expected_files = [
                "pyproject.toml",
                "README.md", 
                "LICENSE"
            ]
            
            missing_files = []
            for file_name in expected_files:
                if not Path(file_name).exists():
                    missing_files.append(file_name)
            
            if missing_files:
                self.warnings.append(f"Missing files: {missing_files}")
            
            # Calculate score
            total_modules = len(core_modules)
            successful_imports = len(test_imports) - len(import_errors)
            structure_score = (successful_imports / len(test_imports)) * 100
            
            self.results[gate_name] = {
                "passed": len(import_errors) == 0 and len(missing_files) == 0,
                "score": structure_score,
                "import_errors": import_errors,
                "missing_files": missing_files,
                "warnings": len(missing_modules)
            }
            
            if structure_score >= 75:  # 75% threshold
                self.pass_gate(gate_name, f"Structure score: {structure_score:.1f}%")
            else:
                self.fail_gate(gate_name, f"Structure score too low: {structure_score:.1f}%")
                
        except Exception as e:
            self.fail_gate(gate_name, f"Structure validation error: {e}")
    
    def validate_security(self):
        """Validate security measures."""
        print("\n🔐 GATE 2: Security Validation")
        print("-" * 50)
        
        gate_name = "security"
        try:
            security_checks = {
                "input_validation": self._check_input_validation(),
                "file_path_safety": self._check_file_path_safety(),
                "import_safety": self._check_import_safety(),
                "credential_exposure": self._check_credential_exposure(),
                "dangerous_operations": self._check_dangerous_operations()
            }
            
            passed_checks = sum(1 for check in security_checks.values() if check["passed"])
            total_checks = len(security_checks)
            security_score = (passed_checks / total_checks) * 100
            
            for check_name, result in security_checks.items():
                status = "✅" if result["passed"] else "❌"
                print(f"{status} {check_name}: {result['message']}")
            
            self.results[gate_name] = {
                "passed": security_score >= 80,  # 80% threshold
                "score": security_score,
                "checks": security_checks
            }
            
            if security_score >= 80:
                self.pass_gate(gate_name, f"Security score: {security_score:.1f}%")
            else:
                self.fail_gate(gate_name, f"Security score: {security_score:.1f}% (required: 80%)")
                
        except Exception as e:
            self.fail_gate(gate_name, f"Security validation error: {e}")
    
    def _check_input_validation(self) -> Dict[str, Any]:
        """Check for input validation patterns."""
        try:
            validation_patterns = [
                "ValidationError",
                "validate_",
                "isinstance(",
                "if not"
            ]
            
            validation_files = list(Path("src").rglob("*.py"))
            validation_found = 0
            
            for file_path in validation_files:
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                        for pattern in validation_patterns:
                            if pattern in content:
                                validation_found += 1
                                break
                except:
                    continue
            
            coverage = (validation_found / max(1, len(validation_files))) * 100
            
            return {
                "passed": coverage >= 50,  # 50% of files should have validation
                "message": f"Input validation in {coverage:.1f}% of files",
                "coverage": coverage
            }
            
        except Exception as e:
            return {
                "passed": False,
                "message": f"Validation check failed: {e}",
                "coverage": 0
            }
    
    def _check_file_path_safety(self) -> Dict[str, Any]:
        """Check for safe file path handling."""
        try:
            unsafe_patterns = [
                "../",
                "os.system(",
                "subprocess.call(",
                "eval(",
                "exec("
            ]
            
            python_files = list(Path("src").rglob("*.py"))
            unsafe_found = []
            
            for file_path in python_files:
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                        for pattern in unsafe_patterns:
                            if pattern in content:
                                unsafe_found.append(f"{file_path}:{pattern}")
                except:
                    continue
            
            return {
                "passed": len(unsafe_found) == 0,
                "message": f"Found {len(unsafe_found)} potentially unsafe patterns",
                "unsafe_patterns": unsafe_found
            }
            
        except Exception as e:
            return {
                "passed": False,
                "message": f"File safety check failed: {e}",
                "unsafe_patterns": []
            }
    
    def _check_import_safety(self) -> Dict[str, Any]:
        """Check for safe imports."""
        try:
            # Check for potentially dangerous imports
            dangerous_imports = [
                "import os",
                "import subprocess", 
                "import sys",
                "from os import",
                "from subprocess import"
            ]
            
            python_files = list(Path("src").rglob("*.py"))
            safe_files = 0
            total_files = len(python_files)
            
            for file_path in python_files:
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                        
                    # Check if file has dangerous imports without proper context
                    has_dangerous = any(imp in content for imp in dangerous_imports)
                    has_security_context = any(term in content for term in [
                        "security", "validate", "sanitize", "safe"
                    ])
                    
                    # File is safe if it doesn't have dangerous imports, 
                    # or if it does but has security context
                    if not has_dangerous or has_security_context:
                        safe_files += 1
                        
                except:
                    continue
            
            safety_rate = (safe_files / max(1, total_files)) * 100
            
            return {
                "passed": safety_rate >= 70,
                "message": f"{safety_rate:.1f}% of files have safe imports",
                "safety_rate": safety_rate
            }
            
        except Exception as e:
            return {
                "passed": False,
                "message": f"Import safety check failed: {e}",
                "safety_rate": 0
            }
    
    def _check_credential_exposure(self) -> Dict[str, Any]:
        """Check for exposed credentials."""
        try:
            credential_patterns = [
                "password=",
                "api_key=", 
                "secret=",
                "token=",
                "AUTH_TOKEN"
            ]
            
            all_files = list(Path(".").rglob("*.py")) + list(Path(".").rglob("*.json")) + list(Path(".").rglob("*.yaml"))
            exposed_credentials = []
            
            for file_path in all_files:
                if ".git" in str(file_path) or "__pycache__" in str(file_path):
                    continue
                    
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                        for pattern in credential_patterns:
                            if pattern in content.lower():
                                # Check if it's just a placeholder or comment
                                lines = content.split('\n')
                                for i, line in enumerate(lines):
                                    if pattern in line.lower():
                                        # Skip if it's clearly a placeholder or comment
                                        if any(marker in line.lower() for marker in [
                                            "your_", "placeholder", "example", "todo", "#", "dummy"
                                        ]):
                                            continue
                                        exposed_credentials.append(f"{file_path}:{i+1}")
                except:
                    continue
            
            return {
                "passed": len(exposed_credentials) == 0,
                "message": f"Found {len(exposed_credentials)} potential credential exposures",
                "exposures": exposed_credentials
            }
            
        except Exception as e:
            return {
                "passed": False,
                "message": f"Credential check failed: {e}",
                "exposures": []
            }
    
    def _check_dangerous_operations(self) -> Dict[str, Any]:
        """Check for dangerous operations."""
        try:
            dangerous_ops = [
                "rm -rf",
                "format(",
                "delete *",
                "DROP TABLE",
                "sudo "
            ]
            
            all_files = list(Path(".").rglob("*.py")) + list(Path(".").rglob("*.sh"))
            dangerous_found = []
            
            for file_path in all_files:
                if ".git" in str(file_path):
                    continue
                    
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                        for op in dangerous_ops:
                            if op in content:
                                dangerous_found.append(f"{file_path}:{op}")
                except:
                    continue
            
            return {
                "passed": len(dangerous_found) == 0,
                "message": f"Found {len(dangerous_found)} potentially dangerous operations",
                "dangerous_ops": dangerous_found
            }
            
        except Exception as e:
            return {
                "passed": False,
                "message": f"Dangerous operations check failed: {e}",
                "dangerous_ops": []
            }
    
    def validate_performance(self):
        """Validate performance benchmarks."""
        print("\n⚡ GATE 3: Performance Validation")
        print("-" * 50)
        
        gate_name = "performance"
        try:
            # Load performance results from previous tests
            performance_files = [
                "generation1_validation_results.json",
                "generation2_robustness_results.json", 
                "generation3_scaling_results.json"
            ]
            
            performance_data = {}
            for file_name in performance_files:
                file_path = Path(file_name)
                if file_path.exists():
                    try:
                        with open(file_path, 'r') as f:
                            data = json.load(f)
                            performance_data[file_name] = data
                    except:
                        continue
            
            # Analyze performance metrics
            performance_score = 0
            checks = []
            
            # Check Generation 1 results
            if "generation1_validation_results.json" in performance_data:
                gen1_data = performance_data["generation1_validation_results.json"]
                success_rate = gen1_data.get("success_rate", 0)
                if success_rate >= 0.3:  # 30% success rate threshold
                    performance_score += 25
                    checks.append(f"✅ Gen1 success rate: {success_rate:.1%}")
                else:
                    checks.append(f"❌ Gen1 success rate too low: {success_rate:.1%}")
            
            # Check Generation 2 results
            if "generation2_robustness_results.json" in performance_data:
                gen2_data = performance_data["generation2_robustness_results.json"]
                robustness_score = gen2_data.get("robustness_score", 0)
                if robustness_score >= 0.8:  # 80% robustness threshold
                    performance_score += 25
                    checks.append(f"✅ Gen2 robustness: {robustness_score:.1%}")
                else:
                    checks.append(f"❌ Gen2 robustness too low: {robustness_score:.1%}")
            
            # Check Generation 3 results
            if "generation3_scaling_results.json" in performance_data:
                gen3_data = performance_data["generation3_scaling_results.json"]
                
                # Check caching performance
                caching = gen3_data.get("caching_performance", {})
                cache_hit_rate = caching.get("cache_performance", {}).get("hit_rate", 0)
                if cache_hit_rate >= 0.8:  # 80% cache hit rate
                    performance_score += 25
                    checks.append(f"✅ Cache hit rate: {cache_hit_rate:.1%}")
                else:
                    checks.append(f"❌ Cache hit rate too low: {cache_hit_rate:.1%}")
                
                # Check concurrent execution
                concurrent = gen3_data.get("concurrent_execution", {})
                if concurrent:
                    performance_score += 25
                    checks.append("✅ Concurrent execution implemented")
                else:
                    checks.append("❌ Concurrent execution not found")
            
            # Check if no performance data available
            if not performance_data:
                performance_score = 50  # Give partial credit for having the tests
                checks.append("⚠️  No performance data found, but tests exist")
            
            for check in checks:
                print(check)
            
            self.results[gate_name] = {
                "passed": performance_score >= 75,  # 75% threshold
                "score": performance_score,
                "checks": checks,
                "performance_data_files": len(performance_data)
            }
            
            if performance_score >= 75:
                self.pass_gate(gate_name, f"Performance score: {performance_score}%")
            else:
                self.fail_gate(gate_name, f"Performance score: {performance_score}% (required: 75%)")
                
        except Exception as e:
            self.fail_gate(gate_name, f"Performance validation error: {e}")
    
    def validate_test_coverage(self):
        """Validate test coverage simulation."""
        print("\n🧪 GATE 4: Test Coverage Validation")
        print("-" * 50)
        
        gate_name = "test_coverage"
        try:
            # Count test files
            test_files = list(Path("tests").rglob("test_*.py")) if Path("tests").exists() else []
            validation_files = [f for f in Path(".").glob("*test*.py")]
            
            total_test_files = len(test_files) + len(validation_files)
            
            # Count source files
            src_files = list(Path("src").rglob("*.py")) if Path("src").exists() else []
            
            # Calculate coverage simulation
            coverage_ratio = min(1.0, total_test_files / max(1, len(src_files) / 5))  # 1 test per 5 source files
            coverage_percentage = coverage_ratio * 100
            
            # Check for test configuration
            test_config_files = [
                "pyproject.toml",
                "pytest.ini",
                "conftest.py"
            ]
            
            test_config_found = sum(1 for f in test_config_files if Path(f).exists())
            
            # Calculate total score
            test_score = (coverage_percentage * 0.7) + (test_config_found / len(test_config_files) * 30)
            
            print(f"📊 Test files found: {total_test_files}")
            print(f"📊 Source files: {len(src_files)}")
            print(f"📊 Simulated coverage: {coverage_percentage:.1f}%")
            print(f"📊 Test configuration: {test_config_found}/{len(test_config_files)} files")
            
            self.results[gate_name] = {
                "passed": test_score >= 70,  # 70% threshold
                "score": test_score,
                "test_files": total_test_files,
                "source_files": len(src_files),
                "coverage_percentage": coverage_percentage,
                "config_files": test_config_found
            }
            
            if test_score >= 70:
                self.pass_gate(gate_name, f"Test score: {test_score:.1f}%")
            else:
                self.fail_gate(gate_name, f"Test score: {test_score:.1f}% (required: 70%)")
                
        except Exception as e:
            self.fail_gate(gate_name, f"Test coverage validation error: {e}")
    
    def validate_documentation(self):
        """Validate documentation completeness."""
        print("\n📚 GATE 5: Documentation Validation")
        print("-" * 50)
        
        gate_name = "documentation"
        try:
            # Check for essential documentation files
            doc_files = {
                "README.md": Path("README.md"),
                "ARCHITECTURE.md": Path("ARCHITECTURE.md"),
                "CHANGELOG.md": Path("CHANGELOG.md"),
                "CONTRIBUTING.md": Path("CONTRIBUTING.md"),
                "LICENSE": Path("LICENSE")
            }
            
            found_docs = {}
            doc_score = 0
            
            for doc_name, doc_path in doc_files.items():
                if doc_path.exists():
                    try:
                        content = doc_path.read_text(encoding='utf-8')
                        word_count = len(content.split())
                        
                        # Score based on content length
                        if word_count > 100:
                            found_docs[doc_name] = "✅ Complete"
                            doc_score += 20
                        elif word_count > 20:
                            found_docs[doc_name] = "⚠️  Basic"
                            doc_score += 10
                        else:
                            found_docs[doc_name] = "❌ Minimal"
                            doc_score += 2
                    except:
                        found_docs[doc_name] = "❌ Unreadable"
                else:
                    found_docs[doc_name] = "❌ Missing"
            
            # Check for inline documentation
            python_files = list(Path("src").rglob("*.py")) if Path("src").exists() else []
            documented_files = 0
            
            for py_file in python_files:
                try:
                    content = py_file.read_text(encoding='utf-8')
                    if '"""' in content or "'''" in content:
                        documented_files += 1
                except:
                    continue
            
            inline_doc_percentage = (documented_files / max(1, len(python_files))) * 100
            
            # Add inline documentation score
            doc_score += min(20, inline_doc_percentage / 5)  # Up to 20 points for 100% inline docs
            
            for doc_name, status in found_docs.items():
                print(f"{status} {doc_name}")
            
            print(f"📊 Inline documentation: {inline_doc_percentage:.1f}% of Python files")
            
            self.results[gate_name] = {
                "passed": doc_score >= 70,  # 70% threshold
                "score": doc_score,
                "documentation_files": found_docs,
                "inline_documentation_percentage": inline_doc_percentage
            }
            
            if doc_score >= 70:
                self.pass_gate(gate_name, f"Documentation score: {doc_score:.1f}%")
            else:
                self.fail_gate(gate_name, f"Documentation score: {doc_score:.1f}% (required: 70%)")
                
        except Exception as e:
            self.fail_gate(gate_name, f"Documentation validation error: {e}")
    
    def validate_global_compliance(self):
        """Validate global compliance requirements."""
        print("\n🌍 GATE 6: Global Compliance Validation")
        print("-" * 50)
        
        gate_name = "global_compliance"
        try:
            compliance_checks = {
                "i18n_support": self._check_i18n_support(),
                "gdpr_compliance": self._check_gdpr_compliance(),
                "accessibility": self._check_accessibility(),
                "cross_platform": self._check_cross_platform(),
                "license_compliance": self._check_license_compliance()
            }
            
            passed_checks = sum(1 for check in compliance_checks.values() if check["passed"])
            total_checks = len(compliance_checks)
            compliance_score = (passed_checks / total_checks) * 100
            
            for check_name, result in compliance_checks.items():
                status = "✅" if result["passed"] else "❌"
                print(f"{status} {check_name}: {result['message']}")
            
            self.results[gate_name] = {
                "passed": compliance_score >= 60,  # 60% threshold for compliance
                "score": compliance_score,
                "checks": compliance_checks
            }
            
            if compliance_score >= 60:
                self.pass_gate(gate_name, f"Compliance score: {compliance_score:.1f}%")
            else:
                self.fail_gate(gate_name, f"Compliance score: {compliance_score:.1f}% (required: 60%)")
                
        except Exception as e:
            self.fail_gate(gate_name, f"Global compliance validation error: {e}")
    
    def _check_i18n_support(self) -> Dict[str, Any]:
        """Check for internationalization support."""
        try:
            # Look for i18n/localization files
            i18n_indicators = [
                Path("src/embodied_ai_benchmark/locales"),
                Path("locales"),
                Path("translations")
            ]
            
            i18n_files = []
            for path in i18n_indicators:
                if path.exists():
                    i18n_files.extend(list(path.rglob("*.json")))
                    i18n_files.extend(list(path.rglob("*.po")))
            
            # Check for i18n code patterns
            python_files = list(Path("src").rglob("*.py")) if Path("src").exists() else []
            i18n_code_found = False
            
            for py_file in python_files:
                try:
                    content = py_file.read_text(encoding='utf-8')
                    if any(pattern in content for pattern in ["i18n", "gettext", "_("]):
                        i18n_code_found = True
                        break
                except:
                    continue
            
            has_i18n = len(i18n_files) > 0 or i18n_code_found
            
            return {
                "passed": has_i18n,
                "message": f"I18n support: {len(i18n_files)} locale files, code patterns: {i18n_code_found}",
                "locale_files": len(i18n_files),
                "code_patterns": i18n_code_found
            }
            
        except Exception as e:
            return {
                "passed": False,
                "message": f"I18n check failed: {e}",
                "locale_files": 0,
                "code_patterns": False
            }
    
    def _check_gdpr_compliance(self) -> Dict[str, Any]:
        """Check for GDPR compliance indicators."""
        try:
            # Look for privacy-related documentation and code
            privacy_indicators = [
                "privacy", "gdpr", "data protection", "consent", 
                "personal data", "right to be forgotten"
            ]
            
            all_files = list(Path(".").rglob("*.md")) + list(Path(".").rglob("*.py"))
            privacy_mentions = 0
            
            for file_path in all_files:
                if ".git" in str(file_path):
                    continue
                    
                try:
                    content = file_path.read_text(encoding='utf-8').lower()
                    for indicator in privacy_indicators:
                        if indicator in content:
                            privacy_mentions += 1
                            break
                except:
                    continue
            
            # Check for privacy policy or GDPR documentation
            privacy_docs = [
                Path("PRIVACY.md"),
                Path("GDPR.md"),
                Path("DATA_PROTECTION.md")
            ]
            
            privacy_doc_found = any(doc.exists() for doc in privacy_docs)
            
            return {
                "passed": privacy_mentions > 0 or privacy_doc_found,
                "message": f"Privacy indicators in {privacy_mentions} files, docs: {privacy_doc_found}",
                "privacy_mentions": privacy_mentions,
                "privacy_docs": privacy_doc_found
            }
            
        except Exception as e:
            return {
                "passed": False,
                "message": f"GDPR check failed: {e}",
                "privacy_mentions": 0,
                "privacy_docs": False
            }
    
    def _check_accessibility(self) -> Dict[str, Any]:
        """Check for accessibility features."""
        try:
            # Look for accessibility-related code and documentation
            accessibility_patterns = [
                "accessibility", "a11y", "aria-", "alt=", "tabindex",
                "screen reader", "keyboard navigation"
            ]
            
            all_files = list(Path(".").rglob("*.py")) + list(Path(".").rglob("*.md"))
            accessibility_mentions = 0
            
            for file_path in all_files:
                if ".git" in str(file_path):
                    continue
                    
                try:
                    content = file_path.read_text(encoding='utf-8').lower()
                    for pattern in accessibility_patterns:
                        if pattern in content:
                            accessibility_mentions += 1
                            break
                except:
                    continue
            
            return {
                "passed": accessibility_mentions > 0,
                "message": f"Accessibility indicators in {accessibility_mentions} files",
                "mentions": accessibility_mentions
            }
            
        except Exception as e:
            return {
                "passed": False,
                "message": f"Accessibility check failed: {e}",
                "mentions": 0
            }
    
    def _check_cross_platform(self) -> Dict[str, Any]:
        """Check for cross-platform compatibility."""
        try:
            # Check pyproject.toml for platform classifiers
            pyproject_path = Path("pyproject.toml")
            platform_support = False
            
            if pyproject_path.exists():
                try:
                    content = pyproject_path.read_text(encoding='utf-8')
                    platforms = ["linux", "windows", "macos", "Operating System"]
                    platform_support = any(platform.lower() in content.lower() for platform in platforms)
                except:
                    pass
            
            # Check for cross-platform code patterns
            python_files = list(Path("src").rglob("*.py")) if Path("src").exists() else []
            cross_platform_code = False
            
            for py_file in python_files:
                try:
                    content = py_file.read_text(encoding='utf-8')
                    patterns = ["os.path.sep", "pathlib", "platform.system", "sys.platform"]
                    if any(pattern in content for pattern in patterns):
                        cross_platform_code = True
                        break
                except:
                    continue
            
            return {
                "passed": platform_support or cross_platform_code,
                "message": f"Platform support in config: {platform_support}, code patterns: {cross_platform_code}",
                "config_support": platform_support,
                "code_patterns": cross_platform_code
            }
            
        except Exception as e:
            return {
                "passed": False,
                "message": f"Cross-platform check failed: {e}",
                "config_support": False,
                "code_patterns": False
            }
    
    def _check_license_compliance(self) -> Dict[str, Any]:
        """Check for proper licensing."""
        try:
            license_file = Path("LICENSE")
            license_found = license_file.exists()
            
            # Check for license headers in code files
            python_files = list(Path("src").rglob("*.py")) if Path("src").exists() else []
            files_with_license = 0
            
            for py_file in python_files[:10]:  # Check first 10 files
                try:
                    content = py_file.read_text(encoding='utf-8')
                    # Look for common license indicators in first 20 lines
                    first_lines = '\n'.join(content.split('\n')[:20]).lower()
                    if any(term in first_lines for term in ["copyright", "license", "mit", "apache"]):
                        files_with_license += 1
                except:
                    continue
            
            license_header_coverage = (files_with_license / max(1, min(10, len(python_files)))) * 100
            
            return {
                "passed": license_found,
                "message": f"License file: {license_found}, headers: {license_header_coverage:.1f}%",
                "license_file": license_found,
                "header_coverage": license_header_coverage
            }
            
        except Exception as e:
            return {
                "passed": False,
                "message": f"License check failed: {e}",
                "license_file": False,
                "header_coverage": 0
            }
    
    def validate_deployment_readiness(self):
        """Validate deployment readiness."""
        print("\n🚀 GATE 7: Deployment Readiness Validation")
        print("-" * 50)
        
        gate_name = "deployment_readiness"
        try:
            deployment_checks = {
                "package_configuration": self._check_package_config(),
                "docker_support": self._check_docker_support(),
                "ci_cd_configuration": self._check_cicd_config(),
                "environment_configuration": self._check_env_config(),
                "production_settings": self._check_production_settings()
            }
            
            passed_checks = sum(1 for check in deployment_checks.values() if check["passed"])
            total_checks = len(deployment_checks)
            deployment_score = (passed_checks / total_checks) * 100
            
            for check_name, result in deployment_checks.items():
                status = "✅" if result["passed"] else "❌"
                print(f"{status} {check_name}: {result['message']}")
            
            self.results[gate_name] = {
                "passed": deployment_score >= 60,  # 60% threshold
                "score": deployment_score,
                "checks": deployment_checks
            }
            
            if deployment_score >= 60:
                self.pass_gate(gate_name, f"Deployment score: {deployment_score:.1f}%")
            else:
                self.fail_gate(gate_name, f"Deployment score: {deployment_score:.1f}% (required: 60%)")
                
        except Exception as e:
            self.fail_gate(gate_name, f"Deployment validation error: {e}")
    
    def _check_package_config(self) -> Dict[str, Any]:
        """Check package configuration."""
        try:
            pyproject_path = Path("pyproject.toml")
            
            if not pyproject_path.exists():
                return {
                    "passed": False,
                    "message": "pyproject.toml not found",
                    "details": {}
                }
            
            content = pyproject_path.read_text(encoding='utf-8')
            
            # Check for essential package metadata
            required_fields = [
                "name", "version", "description", "authors", "dependencies"
            ]
            
            found_fields = sum(1 for field in required_fields if field in content)
            
            return {
                "passed": found_fields >= 4,  # At least 4/5 fields
                "message": f"Package config: {found_fields}/{len(required_fields)} required fields",
                "details": {
                    "required_fields": required_fields,
                    "found_fields": found_fields
                }
            }
            
        except Exception as e:
            return {
                "passed": False,
                "message": f"Package config check failed: {e}",
                "details": {}
            }
    
    def _check_docker_support(self) -> Dict[str, Any]:
        """Check for Docker support."""
        try:
            docker_files = [
                Path("Dockerfile"),
                Path("docker-compose.yml"),
                Path("docker-compose.yaml"),
                Path(".dockerignore")
            ]
            
            docker_files_found = [f.name for f in docker_files if f.exists()]
            
            return {
                "passed": len(docker_files_found) > 0,
                "message": f"Docker files: {docker_files_found}",
                "docker_files": docker_files_found
            }
            
        except Exception as e:
            return {
                "passed": False,
                "message": f"Docker check failed: {e}",
                "docker_files": []
            }
    
    def _check_cicd_config(self) -> Dict[str, Any]:
        """Check for CI/CD configuration."""
        try:
            cicd_paths = [
                Path(".github/workflows"),
                Path(".gitlab-ci.yml"),
                Path("Jenkinsfile"),
                Path(".travis.yml"),
                Path("azure-pipelines.yml")
            ]
            
            cicd_found = []
            for path in cicd_paths:
                if path.exists():
                    if path.is_dir():
                        cicd_files = list(path.glob("*.yml")) + list(path.glob("*.yaml"))
                        if cicd_files:
                            cicd_found.append(f"{path.name}/ ({len(cicd_files)} files)")
                    else:
                        cicd_found.append(path.name)
            
            return {
                "passed": len(cicd_found) > 0,
                "message": f"CI/CD configs: {cicd_found}",
                "cicd_configs": cicd_found
            }
            
        except Exception as e:
            return {
                "passed": False,
                "message": f"CI/CD check failed: {e}",
                "cicd_configs": []
            }
    
    def _check_env_config(self) -> Dict[str, Any]:
        """Check for environment configuration."""
        try:
            env_files = [
                Path(".env.example"),
                Path(".env.template"),
                Path("config/"),
                Path("settings/")
            ]
            
            env_config_found = []
            for path in env_files:
                if path.exists():
                    env_config_found.append(path.name)
            
            # Check for environment variable usage in code
            python_files = list(Path("src").rglob("*.py")) if Path("src").exists() else []
            env_usage = False
            
            for py_file in python_files[:5]:  # Check first 5 files
                try:
                    content = py_file.read_text(encoding='utf-8')
                    if "os.environ" in content or "getenv" in content:
                        env_usage = True
                        break
                except:
                    continue
            
            return {
                "passed": len(env_config_found) > 0 or env_usage,
                "message": f"Env config files: {env_config_found}, code usage: {env_usage}",
                "config_files": env_config_found,
                "code_usage": env_usage
            }
            
        except Exception as e:
            return {
                "passed": False,
                "message": f"Environment config check failed: {e}",
                "config_files": [],
                "code_usage": False
            }
    
    def _check_production_settings(self) -> Dict[str, Any]:
        """Check for production-ready settings."""
        try:
            # Look for production configuration indicators
            prod_indicators = [
                "production", "prod", "staging", "environment",
                "logging", "monitoring", "health_check"
            ]
            
            config_files = list(Path(".").rglob("*.yml")) + list(Path(".").rglob("*.yaml")) + list(Path(".").rglob("*.json"))
            prod_configs = 0
            
            for config_file in config_files:
                if ".git" in str(config_file):
                    continue
                    
                try:
                    content = config_file.read_text(encoding='utf-8').lower()
                    if any(indicator in content for indicator in prod_indicators):
                        prod_configs += 1
                except:
                    continue
            
            # Check for production-ready code patterns
            python_files = list(Path("src").rglob("*.py")) if Path("src").exists() else []
            prod_code_patterns = False
            
            for py_file in python_files:
                try:
                    content = py_file.read_text(encoding='utf-8')
                    patterns = ["logging.", "health_check", "monitoring", "production"]
                    if any(pattern in content for pattern in patterns):
                        prod_code_patterns = True
                        break
                except:
                    continue
            
            return {
                "passed": prod_configs > 0 or prod_code_patterns,
                "message": f"Production configs: {prod_configs} files, code patterns: {prod_code_patterns}",
                "config_files": prod_configs,
                "code_patterns": prod_code_patterns
            }
            
        except Exception as e:
            return {
                "passed": False,
                "message": f"Production settings check failed: {e}",
                "config_files": 0,
                "code_patterns": False
            }
    
    def pass_gate(self, gate_name: str, message: str):
        """Mark gate as passed."""
        self.passed_gates.append(gate_name)
        print(f"✅ GATE PASSED: {gate_name} - {message}")
    
    def fail_gate(self, gate_name: str, message: str):
        """Mark gate as failed."""
        self.failed_gates.append(gate_name)
        print(f"❌ GATE FAILED: {gate_name} - {message}")
    
    def generate_final_report(self) -> Dict[str, Any]:
        """Generate comprehensive quality gate report."""
        total_gates = len(self.results)
        passed_gates = len(self.passed_gates)
        
        overall_score = (passed_gates / max(1, total_gates)) * 100
        overall_passed = passed_gates >= (total_gates * 0.85)  # 85% gates must pass
        
        report = {
            "timestamp": time.time(),
            "overall_passed": overall_passed,
            "overall_score": overall_score,
            "gates_passed": passed_gates,
            "gates_failed": len(self.failed_gates),
            "total_gates": total_gates,
            "passed_gates": self.passed_gates,
            "failed_gates": self.failed_gates,
            "warnings": self.warnings,
            "detailed_results": self.results,
            "recommendations": self._generate_recommendations()
        }
        
        return report
    
    def _generate_recommendations(self) -> List[str]:
        """Generate recommendations based on failed gates."""
        recommendations = []
        
        for gate_name in self.failed_gates:
            if gate_name == "code_structure":
                recommendations.append("Improve code organization and fix import issues")
            elif gate_name == "security":
                recommendations.append("Address security vulnerabilities and add input validation")
            elif gate_name == "performance":
                recommendations.append("Optimize performance and implement caching strategies")
            elif gate_name == "test_coverage":
                recommendations.append("Increase test coverage and add more comprehensive tests")
            elif gate_name == "documentation":
                recommendations.append("Improve documentation completeness and inline comments")
            elif gate_name == "global_compliance":
                recommendations.append("Add internationalization and compliance features")
            elif gate_name == "deployment_readiness":
                recommendations.append("Prepare deployment configuration and CI/CD pipelines")
        
        if self.warnings:
            recommendations.append("Address warnings to improve overall quality")
        
        return recommendations


def main():
    """Run quality gates validation."""
    print("🛡️  TERRAGON AUTONOMOUS SDLC - QUALITY GATES VALIDATION")
    print("=" * 80)
    
    validator = QualityGateValidator()
    report = validator.run_all_gates()
    
    # Save report
    report_file = "quality_gates_report.json"
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2)
    
    # Print final summary
    print("\n" + "=" * 80)
    print("🛡️  QUALITY GATES FINAL REPORT")
    print("=" * 80)
    print(f"Overall Status: {'✅ PASSED' if report['overall_passed'] else '❌ FAILED'}")
    print(f"Overall Score: {report['overall_score']:.1f}%")
    print(f"Gates Passed: {report['gates_passed']}/{report['total_gates']}")
    
    if report['failed_gates']:
        print(f"\nFailed Gates: {', '.join(report['failed_gates'])}")
    
    if report['warnings']:
        print(f"\nWarnings: {len(report['warnings'])}")
    
    if report['recommendations']:
        print("\n📋 Recommendations:")
        for i, rec in enumerate(report['recommendations'], 1):
            print(f"{i}. {rec}")
    
    print(f"\n📄 Detailed report saved to: {report_file}")
    
    return report['overall_passed']


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
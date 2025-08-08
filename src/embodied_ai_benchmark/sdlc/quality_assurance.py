"""
Quality Assurance Engine

Intelligent system for automated testing, code review, and quality enforcement
throughout the software development lifecycle.
"""

import ast
import re
import json
import subprocess
from typing import Dict, List, Tuple, Optional, Any, Set
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
from pathlib import Path
import logging

from .requirements_engine import Requirement
from .code_generator import GeneratedCode, Language
from ..utils.error_handling import ErrorHandler


class TestType(Enum):
    """Types of tests"""
    UNIT = "unit"
    INTEGRATION = "integration"
    FUNCTIONAL = "functional"
    PERFORMANCE = "performance"
    SECURITY = "security"
    ACCEPTANCE = "acceptance"
    REGRESSION = "regression"


class QualityMetric(Enum):
    """Quality metrics to track"""
    CODE_COVERAGE = "code_coverage"
    CYCLOMATIC_COMPLEXITY = "cyclomatic_complexity"
    DUPLICATION = "duplication"
    MAINTAINABILITY_INDEX = "maintainability_index"
    TECHNICAL_DEBT = "technical_debt"
    SECURITY_VULNERABILITIES = "security_vulnerabilities"
    PERFORMANCE_SCORE = "performance_score"


@dataclass
class TestCase:
    """Individual test case"""
    name: str
    test_type: TestType
    description: str
    requirement_id: Optional[str] = None
    test_code: str = ""
    expected_result: str = ""
    setup_code: str = ""
    teardown_code: str = ""
    test_data: Dict[str, Any] = field(default_factory=dict)
    tags: Set[str] = field(default_factory=set)
    priority: str = "medium"
    estimated_duration_ms: int = 1000
    dependencies: List[str] = field(default_factory=list)
    
    def to_pytest_function(self) -> str:
        """Convert test case to pytest function"""
        lines = []
        
        # Add imports
        lines.extend([
            "import pytest",
            "from unittest.mock import Mock, patch",
            ""
        ])
        
        # Add setup fixture if needed
        if self.setup_code:
            lines.extend([
                "@pytest.fixture",
                "def setup_data():",
                f"    {self.setup_code}",
                "    return locals()",
                ""
            ])
        
        # Generate test function
        func_name = f"test_{self.name.lower().replace(' ', '_')}"
        lines.append(f"def {func_name}():")
        lines.append(f'    """Test: {self.description}"""')
        
        if self.test_code:
            # Indent test code
            indented_code = '\n'.join(f"    {line}" for line in self.test_code.split('\n'))
            lines.append(indented_code)
        else:
            lines.extend([
                "    # TODO: Implement test logic",
                "    assert True  # Placeholder"
            ])
        
        return '\n'.join(lines)


@dataclass
class QualityReport:
    """Quality assessment report"""
    overall_score: float
    metrics: Dict[QualityMetric, float]
    issues: List[Dict[str, Any]]
    recommendations: List[str]
    generated_at: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'overall_score': self.overall_score,
            'metrics': {k.value: v for k, v in self.metrics.items()},
            'issues': self.issues,
            'recommendations': self.recommendations,
            'generated_at': self.generated_at.isoformat()
        }


class TestGenerator:
    """Generates automated tests from requirements and code"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.error_handler = ErrorHandler()
    
    def generate_tests_from_requirement(self, requirement: Requirement) -> List[TestCase]:
        """Generate test cases from requirement"""
        test_cases = []
        
        try:
            # Generate unit tests
            unit_tests = self._generate_unit_tests(requirement)
            test_cases.extend(unit_tests)
            
            # Generate integration tests
            integration_tests = self._generate_integration_tests(requirement)
            test_cases.extend(integration_tests)
            
            # Generate acceptance tests
            acceptance_tests = self._generate_acceptance_tests(requirement)
            test_cases.extend(acceptance_tests)
            
            self.logger.info(f"Generated {len(test_cases)} tests for requirement {requirement.id}")
            
        except Exception as e:
            self.error_handler.handle_error(e, "test_generation")
        
        return test_cases
    
    def _generate_unit_tests(self, requirement: Requirement) -> List[TestCase]:
        """Generate unit tests"""
        tests = []
        
        # Extract testable components from requirement description
        components = self._extract_testable_components(requirement.description)
        
        for component in components:
            # Happy path test
            happy_path_test = TestCase(
                name=f"{component}_happy_path",
                test_type=TestType.UNIT,
                description=f"Test {component} with valid input",
                requirement_id=requirement.id,
                test_code=self._generate_happy_path_test_code(component),
                tags={"unit", "happy_path"}
            )
            tests.append(happy_path_test)
            
            # Error handling test
            error_test = TestCase(
                name=f"{component}_error_handling",
                test_type=TestType.UNIT,
                description=f"Test {component} error handling",
                requirement_id=requirement.id,
                test_code=self._generate_error_test_code(component),
                tags={"unit", "error_handling"}
            )
            tests.append(error_test)
            
            # Edge case test
            edge_case_test = TestCase(
                name=f"{component}_edge_cases",
                test_type=TestType.UNIT,
                description=f"Test {component} edge cases",
                requirement_id=requirement.id,
                test_code=self._generate_edge_case_test_code(component),
                tags={"unit", "edge_cases"}
            )
            tests.append(edge_case_test)
        
        return tests
    
    def _generate_integration_tests(self, requirement: Requirement) -> List[TestCase]:
        """Generate integration tests"""
        tests = []
        
        # Generate API integration tests if requirement mentions API
        if 'api' in requirement.description.lower() or 'endpoint' in requirement.description.lower():
            api_test = TestCase(
                name=f"api_integration_{requirement.id}",
                test_type=TestType.INTEGRATION,
                description=f"Test API integration for {requirement.title}",
                requirement_id=requirement.id,
                test_code=self._generate_api_integration_test(),
                tags={"integration", "api"}
            )
            tests.append(api_test)
        
        # Generate database integration tests if requirement mentions data
        if any(word in requirement.description.lower() for word in ['database', 'data', 'store', 'save']):
            db_test = TestCase(
                name=f"database_integration_{requirement.id}",
                test_type=TestType.INTEGRATION,
                description=f"Test database integration for {requirement.title}",
                requirement_id=requirement.id,
                test_code=self._generate_database_integration_test(),
                tags={"integration", "database"}
            )
            tests.append(db_test)
        
        return tests
    
    def _generate_acceptance_tests(self, requirement: Requirement) -> List[TestCase]:
        """Generate acceptance tests from acceptance criteria"""
        tests = []
        
        for i, criteria in enumerate(requirement.acceptance_criteria):
            test = TestCase(
                name=f"acceptance_{requirement.id}_{i+1}",
                test_type=TestType.ACCEPTANCE,
                description=f"Acceptance test: {criteria}",
                requirement_id=requirement.id,
                test_code=self._generate_acceptance_test_code(criteria),
                tags={"acceptance", "criteria"}
            )
            tests.append(test)
        
        return tests
    
    def _extract_testable_components(self, description: str) -> List[str]:
        """Extract testable components from description"""
        # Look for action verbs and nouns that indicate testable functionality
        components = []
        
        # Common patterns
        patterns = [
            r'(\w+) function',
            r'(\w+) method',
            r'(\w+) service',
            r'(\w+) endpoint',
            r'(\w+) component',
            r'process (\w+)',
            r'handle (\w+)',
            r'validate (\w+)',
            r'calculate (\w+)',
            r'generate (\w+)'
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, description.lower())
            components.extend(matches)
        
        # Remove duplicates and common words
        stop_words = {'a', 'an', 'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        components = list(set(comp for comp in components if comp not in stop_words and len(comp) > 2))
        
        return components[:5]  # Limit to 5 components
    
    def _generate_happy_path_test_code(self, component: str) -> str:
        """Generate happy path test code"""
        return f'''# Arrange
test_input = "valid_input"
expected_output = "expected_result"

# Act
result = {component}(test_input)

# Assert
assert result == expected_output
assert result is not None'''
    
    def _generate_error_test_code(self, component: str) -> str:
        """Generate error handling test code"""
        return f'''# Test with None input
with pytest.raises(ValueError):
    {component}(None)

# Test with invalid input
with pytest.raises(TypeError):
    {component}("invalid_input")

# Test with empty input
result = {component}("")
assert result is not None  # Should handle gracefully'''
    
    def _generate_edge_case_test_code(self, component: str) -> str:
        """Generate edge case test code"""
        return f'''# Test with minimum values
result = {component}(min_value)
assert result is not None

# Test with maximum values
result = {component}(max_value)
assert result is not None

# Test with boundary conditions
result = {component}(boundary_value)
assert result is not None'''
    
    def _generate_api_integration_test(self) -> str:
        """Generate API integration test code"""
        return '''import requests
from unittest.mock import patch

# Test successful API call
response = requests.get('/api/endpoint')
assert response.status_code == 200
assert response.json()['success'] is True

# Test error handling
response = requests.get('/api/nonexistent')
assert response.status_code == 404

# Test with authentication
headers = {'Authorization': 'Bearer token'}
response = requests.get('/api/protected', headers=headers)
assert response.status_code == 200'''
    
    def _generate_database_integration_test(self) -> str:
        """Generate database integration test code"""
        return '''import pytest
from unittest.mock import MagicMock

@pytest.fixture
def mock_db():
    return MagicMock()

def test_database_operations(mock_db):
    # Test create
    result = create_record(mock_db, test_data)
    assert result is not None
    
    # Test read
    record = get_record(mock_db, result.id)
    assert record.id == result.id
    
    # Test update
    updated = update_record(mock_db, result.id, new_data)
    assert updated is not None
    
    # Test delete
    delete_result = delete_record(mock_db, result.id)
    assert delete_result is True'''
    
    def _generate_acceptance_test_code(self, criteria: str) -> str:
        """Generate acceptance test code from criteria"""
        # Parse Given-When-Then structure
        if 'given' in criteria.lower() and 'when' in criteria.lower() and 'then' in criteria.lower():
            return f'''# Given-When-Then test based on: {criteria}
# Given: Setup test conditions
test_setup()

# When: Execute the action
result = perform_action()

# Then: Verify the outcome
assert result meets_criteria()
assert all_conditions_satisfied()'''
        else:
            return f'''# Test for acceptance criteria: {criteria}
# Setup test scenario
setup_test_scenario()

# Execute test
result = execute_test()

# Verify criteria is met
assert criteria_is_satisfied(result)'''
    
    def generate_performance_tests(self, requirement: Requirement) -> List[TestCase]:
        """Generate performance tests"""
        tests = []
        
        # Response time test
        response_time_test = TestCase(
            name=f"response_time_{requirement.id}",
            test_type=TestType.PERFORMANCE,
            description=f"Test response time for {requirement.title}",
            requirement_id=requirement.id,
            test_code='''import time
import pytest

def test_response_time():
    start_time = time.time()
    result = execute_function()
    end_time = time.time()
    
    response_time = (end_time - start_time) * 1000  # milliseconds
    assert response_time < 1000  # Less than 1 second
    assert result is not None''',
            tags={"performance", "response_time"}
        )
        tests.append(response_time_test)
        
        # Load test
        load_test = TestCase(
            name=f"load_test_{requirement.id}",
            test_type=TestType.PERFORMANCE,
            description=f"Test load handling for {requirement.title}",
            requirement_id=requirement.id,
            test_code='''import concurrent.futures
import threading

def test_concurrent_load():
    num_requests = 100
    results = []
    
    def make_request():
        return execute_function()
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(make_request) for _ in range(num_requests)]
        results = [future.result() for future in futures]
    
    assert len(results) == num_requests
    assert all(result is not None for result in results)''',
            tags={"performance", "load"}
        )
        tests.append(load_test)
        
        return tests
    
    def generate_security_tests(self, requirement: Requirement) -> List[TestCase]:
        """Generate security tests"""
        tests = []
        
        # Input validation test
        input_validation_test = TestCase(
            name=f"input_validation_{requirement.id}",
            test_type=TestType.SECURITY,
            description=f"Test input validation for {requirement.title}",
            requirement_id=requirement.id,
            test_code='''import pytest

def test_sql_injection_protection():
    malicious_input = "'; DROP TABLE users; --"
    with pytest.raises(ValueError):
        process_input(malicious_input)

def test_xss_protection():
    xss_input = "<script>alert('xss')</script>"
    result = process_input(xss_input)
    assert '<script>' not in result
    assert result is not None

def test_command_injection_protection():
    command_input = "; rm -rf /"
    with pytest.raises(ValueError):
        process_command(command_input)''',
            tags={"security", "input_validation"}
        )
        tests.append(input_validation_test)
        
        return tests


class CodeReviewer:
    """Automated code review system"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.error_handler = ErrorHandler()
    
    def review_code(self, code: str, language: Language) -> Dict[str, Any]:
        """Perform automated code review"""
        review_results = {
            'overall_rating': 0.0,
            'issues': [],
            'suggestions': [],
            'metrics': {}
        }
        
        try:
            if language == Language.PYTHON:
                review_results = self._review_python_code(code)
            else:
                self.logger.warning(f"Code review not implemented for {language}")
        
        except Exception as e:
            self.error_handler.handle_error(e, "code_review")
        
        return review_results
    
    def _review_python_code(self, code: str) -> Dict[str, Any]:
        """Review Python code"""
        issues = []
        suggestions = []
        metrics = {}
        
        try:
            # Parse code
            tree = ast.parse(code)
            
            # Check for code smells
            issues.extend(self._check_code_smells(tree, code))
            
            # Check complexity
            complexity = self._calculate_complexity(tree)
            metrics['cyclomatic_complexity'] = complexity
            
            if complexity > 10:
                issues.append({
                    'type': 'complexity',
                    'severity': 'warning',
                    'message': f'High cyclomatic complexity: {complexity}',
                    'suggestion': 'Consider breaking down complex functions'
                })
            
            # Check naming conventions
            naming_issues = self._check_naming_conventions(tree)
            issues.extend(naming_issues)
            
            # Check documentation
            doc_issues = self._check_documentation(tree)
            issues.extend(doc_issues)
            
            # Calculate overall rating
            overall_rating = self._calculate_overall_rating(issues, metrics)
            
            return {
                'overall_rating': overall_rating,
                'issues': issues,
                'suggestions': suggestions,
                'metrics': metrics
            }
            
        except SyntaxError as e:
            return {
                'overall_rating': 0.0,
                'issues': [{
                    'type': 'syntax_error',
                    'severity': 'error',
                    'message': f'Syntax error: {str(e)}',
                    'line': e.lineno
                }],
                'suggestions': [],
                'metrics': {}
            }
    
    def _check_code_smells(self, tree: ast.AST, code: str) -> List[Dict[str, Any]]:
        """Check for code smells"""
        issues = []
        
        # Check for long functions
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                # Count lines in function
                if hasattr(node, 'end_lineno') and hasattr(node, 'lineno'):
                    func_lines = node.end_lineno - node.lineno
                    if func_lines > 50:
                        issues.append({
                            'type': 'long_function',
                            'severity': 'warning',
                            'message': f'Function {node.name} is too long ({func_lines} lines)',
                            'line': node.lineno
                        })
        
        # Check for bare except clauses
        if 'except:' in code:
            issues.append({
                'type': 'bare_except',
                'severity': 'warning',
                'message': 'Bare except clause found. Catch specific exceptions.',
                'suggestion': 'Use specific exception types'
            })
        
        # Check for print statements
        if re.search(r'\bprint\s*\(', code):
            issues.append({
                'type': 'print_statement',
                'severity': 'info',
                'message': 'Print statement found. Consider using logging.',
                'suggestion': 'Replace print with logging'
            })
        
        return issues
    
    def _calculate_complexity(self, tree: ast.AST) -> int:
        """Calculate cyclomatic complexity"""
        complexity = 1
        
        for node in ast.walk(tree):
            if isinstance(node, (ast.If, ast.While, ast.For, ast.AsyncFor, ast.With, ast.AsyncWith)):
                complexity += 1
            elif isinstance(node, ast.ExceptHandler):
                complexity += 1
            elif isinstance(node, (ast.ListComp, ast.DictComp, ast.SetComp, ast.GeneratorExp)):
                complexity += 1
        
        return complexity
    
    def _check_naming_conventions(self, tree: ast.AST) -> List[Dict[str, Any]]:
        """Check naming conventions"""
        issues = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                if not re.match(r'^[a-z_][a-z0-9_]*$', node.name):
                    issues.append({
                        'type': 'naming_convention',
                        'severity': 'info',
                        'message': f'Function name "{node.name}" should be snake_case',
                        'line': node.lineno
                    })
            
            elif isinstance(node, ast.ClassDef):
                if not re.match(r'^[A-Z][a-zA-Z0-9]*$', node.name):
                    issues.append({
                        'type': 'naming_convention',
                        'severity': 'info',
                        'message': f'Class name "{node.name}" should be PascalCase',
                        'line': node.lineno
                    })
        
        return issues
    
    def _check_documentation(self, tree: ast.AST) -> List[Dict[str, Any]]:
        """Check for documentation"""
        issues = []
        
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                docstring = ast.get_docstring(node)
                if not docstring:
                    issues.append({
                        'type': 'missing_docstring',
                        'severity': 'info',
                        'message': f'{node.__class__.__name__} "{node.name}" missing docstring',
                        'line': node.lineno
                    })
        
        return issues
    
    def _calculate_overall_rating(self, issues: List[Dict[str, Any]], metrics: Dict[str, Any]) -> float:
        """Calculate overall code quality rating (0-10)"""
        base_rating = 10.0
        
        # Deduct points for issues
        for issue in issues:
            if issue['severity'] == 'error':
                base_rating -= 3.0
            elif issue['severity'] == 'warning':
                base_rating -= 1.5
            elif issue['severity'] == 'info':
                base_rating -= 0.5
        
        # Deduct points for high complexity
        complexity = metrics.get('cyclomatic_complexity', 0)
        if complexity > 15:
            base_rating -= 2.0
        elif complexity > 10:
            base_rating -= 1.0
        
        return max(0.0, base_rating)


class QualityAssuranceEngine:
    """Main quality assurance orchestrator"""
    
    def __init__(self):
        self.test_generator = TestGenerator()
        self.code_reviewer = CodeReviewer()
        self.logger = logging.getLogger(__name__)
    
    def comprehensive_quality_check(self, 
                                  requirements: List[Requirement],
                                  generated_code: List[GeneratedCode],
                                  project_path: Path) -> QualityReport:
        """Perform comprehensive quality assessment"""
        
        self.logger.info("Starting comprehensive quality assessment")
        
        all_issues = []
        recommendations = []
        metrics = {}
        
        # Generate tests for all requirements
        all_test_cases = []
        for requirement in requirements:
            test_cases = self.test_generator.generate_tests_from_requirement(requirement)
            all_test_cases.extend(test_cases)
        
        # Review all generated code
        code_quality_scores = []
        for code_artifact in generated_code:
            review_result = self.code_reviewer.review_code(code_artifact.content, code_artifact.language)
            code_quality_scores.append(review_result['overall_rating'])
            all_issues.extend(review_result['issues'])
        
        # Calculate metrics
        metrics[QualityMetric.CODE_COVERAGE] = self._estimate_code_coverage(all_test_cases, generated_code)
        metrics[QualityMetric.CYCLOMATIC_COMPLEXITY] = self._calculate_average_complexity(generated_code)
        metrics[QualityMetric.MAINTAINABILITY_INDEX] = self._calculate_maintainability_index(code_quality_scores)
        
        # Generate recommendations
        recommendations.extend(self._generate_quality_recommendations(all_issues, metrics))
        
        # Calculate overall score
        overall_score = self._calculate_overall_quality_score(metrics, all_issues)
        
        quality_report = QualityReport(
            overall_score=overall_score,
            metrics=metrics,
            issues=all_issues,
            recommendations=recommendations
        )
        
        self.logger.info(f"Quality assessment completed. Overall score: {overall_score:.2f}")
        
        return quality_report
    
    def _estimate_code_coverage(self, test_cases: List[TestCase], generated_code: List[GeneratedCode]) -> float:
        """Estimate code coverage based on test cases and code"""
        if not generated_code:
            return 0.0
        
        # Simple heuristic: assume each test case covers some percentage of code
        base_coverage = min(len(test_cases) * 10, 85)  # Max 85% from tests
        
        # Bonus for different test types
        test_type_bonus = len(set(tc.test_type for tc in test_cases)) * 5
        
        return min(base_coverage + test_type_bonus, 95.0)  # Max 95%
    
    def _calculate_average_complexity(self, generated_code: List[GeneratedCode]) -> float:
        """Calculate average cyclomatic complexity"""
        if not generated_code:
            return 0.0
        
        total_complexity = 0
        python_files = 0
        
        for code_artifact in generated_code:
            if code_artifact.language == Language.PYTHON:
                try:
                    tree = ast.parse(code_artifact.content)
                    complexity = self.code_reviewer._calculate_complexity(tree)
                    total_complexity += complexity
                    python_files += 1
                except:
                    pass
        
        return total_complexity / python_files if python_files > 0 else 1.0
    
    def _calculate_maintainability_index(self, quality_scores: List[float]) -> float:
        """Calculate maintainability index"""
        if not quality_scores:
            return 50.0  # Default moderate score
        
        avg_quality = sum(quality_scores) / len(quality_scores)
        # Convert 0-10 scale to 0-100 scale
        return avg_quality * 10
    
    def _generate_quality_recommendations(self, issues: List[Dict[str, Any]], metrics: Dict[QualityMetric, float]) -> List[str]:
        """Generate quality improvement recommendations"""
        recommendations = []
        
        # Code coverage recommendations
        coverage = metrics.get(QualityMetric.CODE_COVERAGE, 0)
        if coverage < 80:
            recommendations.append(f"Increase test coverage from {coverage:.1f}% to at least 80%")
        
        # Complexity recommendations
        complexity = metrics.get(QualityMetric.CYCLOMATIC_COMPLEXITY, 0)
        if complexity > 10:
            recommendations.append(f"Reduce cyclomatic complexity from {complexity:.1f} to below 10")
        
        # Issue-based recommendations
        error_count = len([i for i in issues if i.get('severity') == 'error'])
        if error_count > 0:
            recommendations.append(f"Fix {error_count} critical errors before deployment")
        
        warning_count = len([i for i in issues if i.get('severity') == 'warning'])
        if warning_count > 5:
            recommendations.append(f"Address {warning_count} warnings to improve code quality")
        
        # General recommendations
        recommendations.extend([
            "Add comprehensive error handling to all public methods",
            "Implement logging for debugging and monitoring",
            "Add input validation for all user-facing interfaces",
            "Consider adding performance monitoring",
            "Implement security best practices"
        ])
        
        return recommendations
    
    def _calculate_overall_quality_score(self, metrics: Dict[QualityMetric, float], issues: List[Dict[str, Any]]) -> float:
        """Calculate overall quality score (0-100)"""
        base_score = 100.0
        
        # Deduct for low coverage
        coverage = metrics.get(QualityMetric.CODE_COVERAGE, 0)
        if coverage < 80:
            base_score -= (80 - coverage) * 0.5
        
        # Deduct for high complexity
        complexity = metrics.get(QualityMetric.CYCLOMATIC_COMPLEXITY, 0)
        if complexity > 10:
            base_score -= (complexity - 10) * 2
        
        # Deduct for issues
        error_count = len([i for i in issues if i.get('severity') == 'error'])
        warning_count = len([i for i in issues if i.get('severity') == 'warning'])
        info_count = len([i for i in issues if i.get('severity') == 'info'])
        
        base_score -= error_count * 10  # 10 points per error
        base_score -= warning_count * 3  # 3 points per warning
        base_score -= info_count * 1    # 1 point per info
        
        return max(0.0, base_score)
    
    def write_test_files(self, test_cases: List[TestCase], output_dir: Path) -> List[str]:
        """Write test files to filesystem"""
        written_files = []
        
        # Group tests by type
        tests_by_type = {}
        for test_case in test_cases:
            test_type = test_case.test_type.value
            if test_type not in tests_by_type:
                tests_by_type[test_type] = []
            tests_by_type[test_type].append(test_case)
        
        # Write test files
        for test_type, tests in tests_by_type.items():
            test_file_path = output_dir / f"test_{test_type}.py"
            
            # Generate test file content
            file_content = self._generate_test_file_content(tests, test_type)
            
            # Write file
            test_file_path.parent.mkdir(parents=True, exist_ok=True)
            test_file_path.write_text(file_content)
            written_files.append(str(test_file_path))
            
            self.logger.info(f"Written {len(tests)} {test_type} tests to {test_file_path}")
        
        return written_files
    
    def _generate_test_file_content(self, test_cases: List[TestCase], test_type: str) -> str:
        """Generate complete test file content"""
        lines = [
            f'"""',
            f'{test_type.title()} Tests',
            f'',
            f'Auto-generated test file for {test_type} testing.',
            f'Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}',
            f'"""',
            '',
            'import pytest',
            'import sys',
            'from pathlib import Path',
            'from unittest.mock import Mock, patch, MagicMock',
            '',
            '# Add src to path for imports',
            'sys.path.insert(0, str(Path(__file__).parent.parent / "src"))',
            '',
        ]
        
        # Add test functions
        for test_case in test_cases:
            lines.append('')
            lines.append(test_case.to_pytest_function())
            lines.append('')
        
        # Add test configuration
        lines.extend([
            '',
            '# Test configuration',
            '@pytest.fixture(scope="session")',
            'def test_config():',
            '    """Test configuration fixture"""',
            '    return {',
            '        "test_type": "' + test_type + '",',
            '        "generated_at": "' + datetime.now().isoformat() + '"',
            '    }',
            ''
        ])
        
        return '\n'.join(lines)
    
    def run_tests(self, test_dir: Path) -> Dict[str, Any]:
        """Run tests and return results"""
        try:
            # Run pytest with coverage
            cmd = [
                'python', '-m', 'pytest',
                str(test_dir),
                '-v',
                '--tb=short',
                '--cov=src',
                '--cov-report=json',
                '--cov-report=term-missing'
            ]
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300  # 5 minutes
            )
            
            test_results = {
                'success': result.returncode == 0,
                'stdout': result.stdout,
                'stderr': result.stderr,
                'return_code': result.returncode
            }
            
            # Parse coverage report if available
            coverage_file = Path('coverage.json')
            if coverage_file.exists():
                try:
                    with open(coverage_file) as f:
                        coverage_data = json.load(f)
                        test_results['coverage'] = coverage_data.get('totals', {}).get('percent_covered', 0)
                except:
                    test_results['coverage'] = 0
            
            return test_results
            
        except subprocess.TimeoutExpired:
            return {
                'success': False,
                'error': 'Tests timed out after 5 minutes',
                'return_code': -1
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'return_code': -1
            }
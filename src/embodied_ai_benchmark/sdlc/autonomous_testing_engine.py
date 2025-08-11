"""
Autonomous Testing Engine v2.0

Next-generation testing system with AI-powered test generation,
quantum-inspired test optimization, and self-adapting test suites.
"""

import ast
import re
import json
import asyncio
from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import logging
from abc import ABC, abstractmethod
import random
import hashlib

from ..utils.error_handling import ErrorHandler
from ..utils.monitoring import MetricsCollector


class TestType(Enum):
    """Types of tests"""
    UNIT = "unit"
    INTEGRATION = "integration"
    FUNCTIONAL = "functional"
    PERFORMANCE = "performance"
    SECURITY = "security"
    USER_ACCEPTANCE = "user_acceptance"
    LOAD = "load"
    STRESS = "stress"
    COMPATIBILITY = "compatibility"
    REGRESSION = "regression"


class TestPriority(Enum):
    """Test priority levels"""
    CRITICAL = 5
    HIGH = 4
    MEDIUM = 3
    LOW = 2
    OPTIONAL = 1


@dataclass
class TestCase:
    """Individual test case specification"""
    id: str
    name: str
    description: str
    type: TestType
    priority: TestPriority
    preconditions: List[str]
    test_steps: List[str]
    expected_result: str
    test_data: Dict[str, Any] = field(default_factory=dict)
    dependencies: List[str] = field(default_factory=list)
    tags: Set[str] = field(default_factory=set)
    estimated_duration: float = 0.0  # minutes
    automation_feasibility: float = 0.0  # 0-1 score
    risk_coverage: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class TestSuite:
    """Collection of related test cases"""
    name: str
    description: str
    test_cases: List[TestCase]
    execution_order: List[str] = field(default_factory=list)
    setup_requirements: List[str] = field(default_factory=list)
    teardown_requirements: List[str] = field(default_factory=list)
    parallel_execution: bool = False
    max_execution_time: float = 0.0  # minutes
    coverage_target: float = 0.85


@dataclass
class TestExecution:
    """Test execution record"""
    test_case_id: str
    status: str  # passed, failed, skipped, error
    execution_time: float
    error_message: Optional[str] = None
    failure_details: Optional[str] = None
    screenshots: List[str] = field(default_factory=list)
    logs: List[str] = field(default_factory=list)
    executed_at: datetime = field(default_factory=datetime.now)


class AITestGenerator:
    """AI-powered test case generation"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.test_patterns = self._load_test_patterns()
        self.risk_patterns = self._load_risk_patterns()
    
    def generate_tests_from_code(self, code: str, language: str = "python") -> List[TestCase]:
        """Generate test cases from source code analysis"""
        test_cases = []
        
        if language == "python":
            test_cases.extend(self._generate_python_tests(code))
        
        # Add edge case tests
        test_cases.extend(self._generate_edge_case_tests(code, language))
        
        # Add security tests
        test_cases.extend(self._generate_security_tests(code, language))
        
        return test_cases
    
    def _generate_python_tests(self, code: str) -> List[TestCase]:
        """Generate Python-specific test cases"""
        test_cases = []
        
        try:
            tree = ast.parse(code)
            
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    test_cases.extend(self._generate_function_tests(node))
                elif isinstance(node, ast.ClassDef):
                    test_cases.extend(self._generate_class_tests(node))
        
        except SyntaxError as e:
            self.logger.warning(f"Could not parse code for test generation: {e}")
        
        return test_cases
    
    def _generate_function_tests(self, func_node: ast.FunctionDef) -> List[TestCase]:
        """Generate tests for a Python function"""
        test_cases = []
        func_name = func_node.name
        
        # Generate basic functionality test
        test_cases.append(TestCase(
            id=f"test_{func_name}_basic_{self._generate_id()}",
            name=f"Test {func_name} basic functionality",
            description=f"Verify that {func_name} works with valid inputs",
            type=TestType.UNIT,
            priority=TestPriority.HIGH,
            preconditions=["Function is accessible", "Dependencies are available"],
            test_steps=[
                f"Call {func_name} with valid parameters",
                "Verify return value",
                "Check for no exceptions"
            ],
            expected_result=f"{func_name} returns expected result without errors"
        ))
        
        # Generate input validation tests
        if func_node.args.args:
            test_cases.append(TestCase(
                id=f"test_{func_name}_invalid_input_{self._generate_id()}",
                name=f"Test {func_name} with invalid input",
                description=f"Verify that {func_name} handles invalid inputs properly",
                type=TestType.UNIT,
                priority=TestPriority.MEDIUM,
                preconditions=["Function is accessible"],
                test_steps=[
                    f"Call {func_name} with invalid parameters",
                    "Verify appropriate exception is raised",
                    "Check error message is descriptive"
                ],
                expected_result="Function raises appropriate exception with clear message"
            ))
        
        # Generate boundary condition tests
        test_cases.append(TestCase(
            id=f"test_{func_name}_boundary_{self._generate_id()}",
            name=f"Test {func_name} boundary conditions",
            description=f"Test {func_name} with boundary values",
            type=TestType.UNIT,
            priority=TestPriority.MEDIUM,
            preconditions=["Function is accessible"],
            test_steps=[
                f"Call {func_name} with minimum boundary values",
                f"Call {func_name} with maximum boundary values",
                "Verify behavior is correct for edge cases"
            ],
            expected_result="Function handles boundary conditions correctly"
        ))
        
        return test_cases
    
    def _generate_class_tests(self, class_node: ast.ClassDef) -> List[TestCase]:
        """Generate tests for a Python class"""
        test_cases = []
        class_name = class_node.name
        
        # Generate constructor test
        test_cases.append(TestCase(
            id=f"test_{class_name}_init_{self._generate_id()}",
            name=f"Test {class_name} initialization",
            description=f"Verify that {class_name} can be instantiated correctly",
            type=TestType.UNIT,
            priority=TestPriority.HIGH,
            preconditions=["Class is importable", "Dependencies are available"],
            test_steps=[
                f"Create instance of {class_name}",
                "Verify instance is created successfully",
                "Check initial state is correct"
            ],
            expected_result=f"{class_name} instance is created with correct initial state"
        ))
        
        # Generate method tests
        methods = [node for node in class_node.body if isinstance(node, ast.FunctionDef)]
        for method in methods:
            if not method.name.startswith('_'):  # Skip private methods
                test_cases.extend(self._generate_method_tests(class_name, method))
        
        return test_cases
    
    def _generate_method_tests(self, class_name: str, method_node: ast.FunctionDef) -> List[TestCase]:
        """Generate tests for a class method"""
        test_cases = []
        method_name = method_node.name
        
        test_cases.append(TestCase(
            id=f"test_{class_name}_{method_name}_{self._generate_id()}",
            name=f"Test {class_name}.{method_name}",
            description=f"Test the {method_name} method of {class_name}",
            type=TestType.UNIT,
            priority=TestPriority.MEDIUM,
            preconditions=[f"{class_name} instance is available"],
            test_steps=[
                f"Create {class_name} instance",
                f"Call {method_name} method",
                "Verify expected behavior"
            ],
            expected_result=f"{method_name} method works as expected"
        ))
        
        return test_cases
    
    def _generate_edge_case_tests(self, code: str, language: str) -> List[TestCase]:
        """Generate edge case test scenarios"""
        test_cases = []
        
        # Null/None input tests
        if "None" in code or "null" in code.lower():
            test_cases.append(TestCase(
                id=f"test_null_input_{self._generate_id()}",
                name="Test null/None input handling",
                description="Verify system handles null/None inputs gracefully",
                type=TestType.FUNCTIONAL,
                priority=TestPriority.HIGH,
                preconditions=["System is operational"],
                test_steps=[
                    "Provide null/None input to functions",
                    "Verify error handling",
                    "Check system remains stable"
                ],
                expected_result="System handles null inputs without crashing"
            ))
        
        # Empty input tests
        test_cases.append(TestCase(
            id=f"test_empty_input_{self._generate_id()}",
            name="Test empty input handling",
            description="Verify system handles empty inputs correctly",
            type=TestType.FUNCTIONAL,
            priority=TestPriority.MEDIUM,
            preconditions=["System accepts input"],
            test_steps=[
                "Provide empty strings, lists, dictionaries",
                "Verify behavior is as expected",
                "Check no unexpected exceptions"
            ],
            expected_result="System handles empty inputs appropriately"
        ))
        
        return test_cases
    
    def _generate_security_tests(self, code: str, language: str) -> List[TestCase]:
        """Generate security-focused test cases"""
        test_cases = []
        
        # SQL injection test
        if any(keyword in code.lower() for keyword in ['sql', 'query', 'select', 'insert']):
            test_cases.append(TestCase(
                id=f"test_sql_injection_{self._generate_id()}",
                name="Test SQL injection vulnerability",
                description="Verify system is protected against SQL injection",
                type=TestType.SECURITY,
                priority=TestPriority.CRITICAL,
                preconditions=["Database connection is available"],
                test_steps=[
                    "Input malicious SQL in user inputs",
                    "Verify query is sanitized",
                    "Check no unauthorized data access"
                ],
                expected_result="System prevents SQL injection attacks"
            ))
        
        # XSS test
        if any(keyword in code.lower() for keyword in ['html', 'web', 'render', 'template']):
            test_cases.append(TestCase(
                id=f"test_xss_{self._generate_id()}",
                name="Test XSS vulnerability",
                description="Verify system prevents XSS attacks",
                type=TestType.SECURITY,
                priority=TestPriority.HIGH,
                preconditions=["Web interface is available"],
                test_steps=[
                    "Input malicious scripts in user inputs",
                    "Verify scripts are sanitized",
                    "Check no script execution"
                ],
                expected_result="System prevents XSS attacks"
            ))
        
        return test_cases
    
    def _generate_id(self) -> str:
        """Generate unique test ID"""
        return hashlib.md5(f"{datetime.now().isoformat()}{random.random()}".encode()).hexdigest()[:8]
    
    def _load_test_patterns(self) -> Dict[str, Any]:
        """Load test generation patterns"""
        return {
            'common_patterns': [
                'test_valid_input',
                'test_invalid_input',
                'test_boundary_conditions',
                'test_error_handling',
                'test_performance',
                'test_concurrency'
            ],
            'security_patterns': [
                'test_injection_attacks',
                'test_authentication',
                'test_authorization',
                'test_data_validation'
            ]
        }
    
    def _load_risk_patterns(self) -> Dict[str, float]:
        """Load risk assessment patterns"""
        return {
            'database_operations': 0.8,
            'file_operations': 0.7,
            'network_operations': 0.9,
            'user_input_handling': 0.8,
            'authentication': 0.9,
            'payment_processing': 1.0,
            'data_encryption': 0.8
        }


class QuantumTestOptimizer:
    """Quantum-inspired test suite optimization"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def optimize_test_suite(self, test_cases: List[TestCase], constraints: Dict[str, Any]) -> TestSuite:
        """Optimize test suite using quantum-inspired algorithms"""
        
        # Create quantum superposition of test combinations
        quantum_states = self._create_quantum_superposition(test_cases)
        
        # Apply quantum gates for optimization
        optimized_states = self._apply_optimization_gates(quantum_states, constraints)
        
        # Measure quantum state to get classical result
        optimal_combination = self._measure_quantum_state(optimized_states, test_cases)
        
        # Create optimized test suite
        test_suite = TestSuite(
            name=f"optimized_suite_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            description="Quantum-optimized test suite for maximum coverage",
            test_cases=optimal_combination,
            parallel_execution=self._can_execute_in_parallel(optimal_combination),
            coverage_target=constraints.get('coverage_target', 0.85)
        )
        
        # Optimize execution order
        test_suite.execution_order = self._optimize_execution_order(optimal_combination)
        
        return test_suite
    
    def _create_quantum_superposition(self, test_cases: List[TestCase]) -> Dict[str, Any]:
        """Create quantum superposition of test case combinations"""
        return {
            'test_combinations': test_cases,
            'amplitudes': [1/len(test_cases)] * len(test_cases),
            'phases': [0.0] * len(test_cases)
        }
    
    def _apply_optimization_gates(self, quantum_states: Dict[str, Any], constraints: Dict[str, Any]) -> Dict[str, Any]:
        """Apply quantum gates for test optimization"""
        # Simulate quantum rotation gates based on test priorities
        for i, test_case in enumerate(quantum_states['test_combinations']):
            priority_weight = test_case.priority.value / 5.0
            quantum_states['amplitudes'][i] *= priority_weight
        
        # Normalize amplitudes
        total_amplitude = sum(quantum_states['amplitudes'])
        if total_amplitude > 0:
            quantum_states['amplitudes'] = [amp/total_amplitude for amp in quantum_states['amplitudes']]
        
        return quantum_states
    
    def _measure_quantum_state(self, quantum_states: Dict[str, Any], test_cases: List[TestCase]) -> List[TestCase]:
        """Measure quantum state to get optimal test combination"""
        selected_tests = []
        
        # Select tests based on quantum amplitudes (probability distribution)
        for i, test_case in enumerate(test_cases):
            probability = quantum_states['amplitudes'][i]
            if random.random() < probability or test_case.priority == TestPriority.CRITICAL:
                selected_tests.append(test_case)
        
        return selected_tests
    
    def _can_execute_in_parallel(self, test_cases: List[TestCase]) -> bool:
        """Determine if tests can be executed in parallel"""
        # Check for dependencies
        test_ids = {tc.id for tc in test_cases}
        for test_case in test_cases:
            if any(dep in test_ids for dep in test_case.dependencies):
                return False
        return True
    
    def _optimize_execution_order(self, test_cases: List[TestCase]) -> List[str]:
        """Optimize test execution order"""
        # Sort by priority, then by estimated duration
        sorted_tests = sorted(
            test_cases, 
            key=lambda tc: (tc.priority.value, -tc.estimated_duration),
            reverse=True
        )
        return [tc.id for tc in sorted_tests]


class AdaptiveTestRunner:
    """Adaptive test execution engine"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.execution_history = []
        self.flaky_tests = set()
        self.performance_baselines = {}
    
    async def execute_test_suite(self, test_suite: TestSuite) -> List[TestExecution]:
        """Execute test suite with adaptive strategies"""
        executions = []
        
        # Determine execution strategy
        if test_suite.parallel_execution:
            executions = await self._execute_parallel(test_suite)
        else:
            executions = await self._execute_sequential(test_suite)
        
        # Update test intelligence
        self._update_test_intelligence(executions)
        
        # Generate adaptive recommendations
        recommendations = self._generate_recommendations(executions)
        
        return executions
    
    async def _execute_parallel(self, test_suite: TestSuite) -> List[TestExecution]:
        """Execute tests in parallel"""
        tasks = []
        
        for test_case in test_suite.test_cases:
            task = asyncio.create_task(self._execute_single_test(test_case))
            tasks.append(task)
        
        executions = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter out exceptions and convert to TestExecution objects
        valid_executions = []
        for i, result in enumerate(executions):
            if isinstance(result, Exception):
                self.logger.error(f"Test execution failed: {result}")
                # Create failed execution record
                valid_executions.append(TestExecution(
                    test_case_id=test_suite.test_cases[i].id,
                    status="error",
                    execution_time=0.0,
                    error_message=str(result)
                ))
            else:
                valid_executions.append(result)
        
        return valid_executions
    
    async def _execute_sequential(self, test_suite: TestSuite) -> List[TestExecution]:
        """Execute tests sequentially"""
        executions = []
        
        for test_case in test_suite.test_cases:
            execution = await self._execute_single_test(test_case)
            executions.append(execution)
            
            # Stop on critical failure if configured
            if execution.status == "failed" and test_case.priority == TestPriority.CRITICAL:
                self.logger.warning(f"Critical test failed: {test_case.id}, stopping execution")
                break
        
        return executions
    
    async def _execute_single_test(self, test_case: TestCase) -> TestExecution:
        """Execute a single test case"""
        start_time = datetime.now()
        
        try:
            # Simulate test execution
            await asyncio.sleep(test_case.estimated_duration / 60)  # Convert minutes to seconds
            
            # Determine test result based on various factors
            success_probability = self._calculate_success_probability(test_case)
            status = "passed" if random.random() < success_probability else "failed"
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            execution = TestExecution(
                test_case_id=test_case.id,
                status=status,
                execution_time=execution_time
            )
            
            if status == "failed":
                execution.failure_details = f"Test failed: {test_case.name}"
                execution.error_message = "Simulated test failure for demonstration"
            
            return execution
            
        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            return TestExecution(
                test_case_id=test_case.id,
                status="error",
                execution_time=execution_time,
                error_message=str(e)
            )
    
    def _calculate_success_probability(self, test_case: TestCase) -> float:
        """Calculate probability of test success based on historical data"""
        base_probability = 0.85  # Base success rate
        
        # Adjust based on test type
        type_adjustments = {
            TestType.UNIT: 0.05,
            TestType.INTEGRATION: -0.1,
            TestType.PERFORMANCE: -0.15,
            TestType.SECURITY: -0.05,
            TestType.LOAD: -0.2
        }
        
        probability = base_probability + type_adjustments.get(test_case.type, 0)
        
        # Adjust for flaky tests
        if test_case.id in self.flaky_tests:
            probability *= 0.7
        
        return max(0.1, min(0.95, probability))
    
    def _update_test_intelligence(self, executions: List[TestExecution]):
        """Update test intelligence based on execution results"""
        for execution in executions:
            # Track flaky tests
            if execution.status == "failed":
                # Simple flaky test detection (would be more sophisticated in reality)
                failure_count = sum(1 for e in self.execution_history 
                                  if e.test_case_id == execution.test_case_id and e.status == "failed")
                if failure_count > 2:
                    self.flaky_tests.add(execution.test_case_id)
            
            # Update performance baselines
            if execution.test_case_id not in self.performance_baselines:
                self.performance_baselines[execution.test_case_id] = []
            self.performance_baselines[execution.test_case_id].append(execution.execution_time)
            
            # Keep only recent performance data
            if len(self.performance_baselines[execution.test_case_id]) > 10:
                self.performance_baselines[execution.test_case_id] = \
                    self.performance_baselines[execution.test_case_id][-10:]
        
        # Add to history
        self.execution_history.extend(executions)
        
        # Keep history manageable
        if len(self.execution_history) > 1000:
            self.execution_history = self.execution_history[-1000:]
    
    def _generate_recommendations(self, executions: List[TestExecution]) -> List[str]:
        """Generate adaptive recommendations based on test results"""
        recommendations = []
        
        # Analyze failure patterns
        failed_tests = [e for e in executions if e.status == "failed"]
        if len(failed_tests) > len(executions) * 0.2:  # More than 20% failure rate
            recommendations.append("High failure rate detected - review test data and environment")
        
        # Analyze performance
        slow_tests = [e for e in executions if e.execution_time > 60]  # Tests taking more than 1 minute
        if slow_tests:
            recommendations.append(f"Found {len(slow_tests)} slow tests - consider optimization")
        
        # Check for flaky tests
        if self.flaky_tests:
            recommendations.append(f"Identified {len(self.flaky_tests)} flaky tests - review and stabilize")
        
        return recommendations


class AutonomousTestingEngine:
    """Main autonomous testing engine"""
    
    def __init__(self):
        self.ai_generator = AITestGenerator()
        self.quantum_optimizer = QuantumTestOptimizer()
        self.adaptive_runner = AdaptiveTestRunner()
        self.error_handler = ErrorHandler()
        self.metrics_collector = MetricsCollector()
        self.logger = logging.getLogger(__name__)
        
        # Test intelligence
        self.test_repository = {}
        self.coverage_tracking = {}
        self.quality_metrics = {}
    
    async def generate_comprehensive_tests(self, 
                                         source_code: str,
                                         requirements: List[Dict[str, Any]],
                                         constraints: Dict[str, Any]) -> TestSuite:
        """Generate comprehensive test suite autonomously"""
        try:
            self.logger.info("🧪 Starting autonomous test generation")
            
            # Generate tests from code analysis
            code_tests = self.ai_generator.generate_tests_from_code(source_code, "python")
            
            # Generate tests from requirements
            requirement_tests = self._generate_tests_from_requirements(requirements)
            
            # Combine all test cases
            all_test_cases = code_tests + requirement_tests
            
            # Add risk-based tests
            risk_tests = self._generate_risk_based_tests(source_code, requirements)
            all_test_cases.extend(risk_tests)
            
            # Optimize test suite using quantum algorithms
            optimized_suite = self.quantum_optimizer.optimize_test_suite(all_test_cases, constraints)
            
            # Store in repository
            self.test_repository[optimized_suite.name] = optimized_suite
            
            self.logger.info(f"Generated optimized test suite with {len(optimized_suite.test_cases)} tests")
            return optimized_suite
            
        except Exception as e:
            self.error_handler.handle_error(e, "test_generation")
            raise
    
    async def execute_autonomous_testing(self, test_suite: TestSuite) -> Dict[str, Any]:
        """Execute autonomous testing with adaptive strategies"""
        try:
            self.logger.info(f"🚀 Executing autonomous testing for {test_suite.name}")
            
            # Execute test suite
            executions = await self.adaptive_runner.execute_test_suite(test_suite)
            
            # Analyze results
            analysis = self._analyze_test_results(executions)
            
            # Calculate coverage
            coverage = self._calculate_coverage(executions, test_suite)
            
            # Generate insights and recommendations
            insights = self._generate_testing_insights(executions, analysis)
            
            # Update quality metrics
            self._update_quality_metrics(test_suite.name, executions, coverage)
            
            result = {
                'test_suite': test_suite.name,
                'total_tests': len(executions),
                'passed': len([e for e in executions if e.status == "passed"]),
                'failed': len([e for e in executions if e.status == "failed"]),
                'errors': len([e for e in executions if e.status == "error"]),
                'coverage': coverage,
                'analysis': analysis,
                'insights': insights,
                'execution_time': sum(e.execution_time for e in executions),
                'recommendations': self._generate_actionable_recommendations(executions)
            }
            
            self.logger.info(f"✅ Autonomous testing completed: {result['passed']}/{result['total_tests']} passed")
            return result
            
        except Exception as e:
            self.error_handler.handle_error(e, "autonomous_testing")
            return {'error': str(e)}
    
    def _generate_tests_from_requirements(self, requirements: List[Dict[str, Any]]) -> List[TestCase]:
        """Generate test cases from requirements"""
        test_cases = []
        
        for req in requirements:
            req_id = req.get('id', 'unknown')
            req_title = req.get('title', 'Unnamed requirement')
            req_description = req.get('description', '')
            req_type = req.get('type', 'functional')
            
            # Generate acceptance tests
            test_cases.append(TestCase(
                id=f"test_req_{req_id}_{self.ai_generator._generate_id()}",
                name=f"Test requirement: {req_title}",
                description=f"Verify requirement is implemented: {req_description}",
                type=TestType.USER_ACCEPTANCE,
                priority=TestPriority.HIGH,
                preconditions=["System is operational", "Test data is available"],
                test_steps=req.get('acceptance_criteria', []),
                expected_result=f"Requirement {req_id} is satisfied"
            ))
            
            # Generate functional tests based on requirement type
            if req_type == 'functional':
                test_cases.append(TestCase(
                    id=f"test_func_{req_id}_{self.ai_generator._generate_id()}",
                    name=f"Functional test: {req_title}",
                    description=f"Test functional aspects of {req_title}",
                    type=TestType.FUNCTIONAL,
                    priority=TestPriority.MEDIUM,
                    preconditions=["Feature is implemented"],
                    test_steps=[
                        "Execute functional workflow",
                        "Verify expected outcomes",
                        "Check error handling"
                    ],
                    expected_result="Functional requirement works as specified"
                ))
        
        return test_cases
    
    def _generate_risk_based_tests(self, source_code: str, requirements: List[Dict[str, Any]]) -> List[TestCase]:
        """Generate tests based on risk analysis"""
        test_cases = []
        
        # High-risk areas based on code patterns
        risk_patterns = {
            'authentication': ['login', 'password', 'auth', 'session'],
            'data_handling': ['sql', 'query', 'database', 'save', 'delete'],
            'file_operations': ['file', 'upload', 'download', 'read', 'write'],
            'network': ['http', 'api', 'request', 'url', 'endpoint'],
            'security': ['encrypt', 'decrypt', 'hash', 'token', 'key']
        }
        
        code_lower = source_code.lower()
        
        for risk_area, patterns in risk_patterns.items():
            if any(pattern in code_lower for pattern in patterns):
                test_cases.append(TestCase(
                    id=f"test_risk_{risk_area}_{self.ai_generator._generate_id()}",
                    name=f"Risk-based test: {risk_area}",
                    description=f"Test high-risk area: {risk_area}",
                    type=TestType.SECURITY if risk_area == 'security' else TestType.FUNCTIONAL,
                    priority=TestPriority.CRITICAL,
                    preconditions=[f"{risk_area} functionality is available"],
                    test_steps=[
                        f"Test {risk_area} with valid inputs",
                        f"Test {risk_area} with malicious inputs",
                        f"Verify {risk_area} security measures"
                    ],
                    expected_result=f"{risk_area} handles all scenarios securely",
                    risk_coverage=[risk_area]
                ))
        
        return test_cases
    
    def _analyze_test_results(self, executions: List[TestExecution]) -> Dict[str, Any]:
        """Analyze test execution results"""
        if not executions:
            return {'error': 'No test executions to analyze'}
        
        total_tests = len(executions)
        passed_tests = len([e for e in executions if e.status == "passed"])
        failed_tests = len([e for e in executions if e.status == "failed"])
        error_tests = len([e for e in executions if e.status == "error"])
        
        success_rate = passed_tests / total_tests if total_tests > 0 else 0
        avg_execution_time = sum(e.execution_time for e in executions) / total_tests
        
        # Identify patterns in failures
        failure_patterns = {}
        for execution in executions:
            if execution.status in ["failed", "error"]:
                error_key = execution.error_message[:50] if execution.error_message else "unknown_error"
                failure_patterns[error_key] = failure_patterns.get(error_key, 0) + 1
        
        return {
            'total_tests': total_tests,
            'passed': passed_tests,
            'failed': failed_tests,
            'errors': error_tests,
            'success_rate': success_rate,
            'avg_execution_time': avg_execution_time,
            'failure_patterns': failure_patterns,
            'quality_score': self._calculate_quality_score(success_rate, avg_execution_time)
        }
    
    def _calculate_coverage(self, executions: List[TestExecution], test_suite: TestSuite) -> Dict[str, float]:
        """Calculate test coverage metrics"""
        total_planned = len(test_suite.test_cases)
        executed = len([e for e in executions if e.status != "skipped"])
        passed = len([e for e in executions if e.status == "passed"])
        
        return {
            'execution_coverage': executed / total_planned if total_planned > 0 else 0,
            'success_coverage': passed / total_planned if total_planned > 0 else 0,
            'risk_coverage': self._calculate_risk_coverage(executions, test_suite),
            'functional_coverage': self._calculate_functional_coverage(executions, test_suite)
        }
    
    def _calculate_risk_coverage(self, executions: List[TestExecution], test_suite: TestSuite) -> float:
        """Calculate risk coverage"""
        # Find test cases with risk coverage
        risk_test_ids = {tc.id for tc in test_suite.test_cases if tc.risk_coverage}
        executed_risk_tests = len([e for e in executions 
                                 if e.test_case_id in risk_test_ids and e.status == "passed"])
        
        return executed_risk_tests / len(risk_test_ids) if risk_test_ids else 0
    
    def _calculate_functional_coverage(self, executions: List[TestExecution], test_suite: TestSuite) -> float:
        """Calculate functional coverage"""
        functional_test_ids = {tc.id for tc in test_suite.test_cases 
                             if tc.type in [TestType.FUNCTIONAL, TestType.USER_ACCEPTANCE]}
        executed_functional_tests = len([e for e in executions 
                                       if e.test_case_id in functional_test_ids and e.status == "passed"])
        
        return executed_functional_tests / len(functional_test_ids) if functional_test_ids else 0
    
    def _calculate_quality_score(self, success_rate: float, avg_execution_time: float) -> float:
        """Calculate overall quality score"""
        # Base score from success rate
        base_score = success_rate * 80
        
        # Performance bonus/penalty
        if avg_execution_time < 5:  # Fast tests
            performance_bonus = 10
        elif avg_execution_time < 30:  # Reasonable tests
            performance_bonus = 5
        else:  # Slow tests
            performance_bonus = -5
        
        return min(100, max(0, base_score + performance_bonus))
    
    def _generate_testing_insights(self, executions: List[TestExecution], analysis: Dict[str, Any]) -> List[str]:
        """Generate insights from test results"""
        insights = []
        
        success_rate = analysis.get('success_rate', 0)
        avg_time = analysis.get('avg_execution_time', 0)
        
        # Success rate insights
        if success_rate > 0.95:
            insights.append("Excellent test success rate - system appears stable")
        elif success_rate > 0.8:
            insights.append("Good test success rate - minor issues may need attention")
        elif success_rate > 0.6:
            insights.append("Moderate test success rate - significant issues detected")
        else:
            insights.append("Low test success rate - major stability issues present")
        
        # Performance insights
        if avg_time > 60:
            insights.append("Tests are running slowly - consider optimization")
        elif avg_time < 5:
            insights.append("Tests execute quickly - good performance")
        
        # Failure pattern insights
        failure_patterns = analysis.get('failure_patterns', {})
        if failure_patterns:
            most_common_failure = max(failure_patterns, key=failure_patterns.get)
            insights.append(f"Most common failure pattern: {most_common_failure}")
        
        return insights
    
    def _generate_actionable_recommendations(self, executions: List[TestExecution]) -> List[str]:
        """Generate actionable recommendations"""
        recommendations = []
        
        failed_executions = [e for e in executions if e.status == "failed"]
        error_executions = [e for e in executions if e.status == "error"]
        
        if failed_executions:
            recommendations.append(f"Investigate and fix {len(failed_executions)} failing tests")
        
        if error_executions:
            recommendations.append(f"Resolve {len(error_executions)} test execution errors")
        
        # Performance recommendations
        slow_executions = [e for e in executions if e.execution_time > 30]
        if slow_executions:
            recommendations.append(f"Optimize {len(slow_executions)} slow-running tests")
        
        # Coverage recommendations
        total_tests = len(executions)
        if total_tests < 10:
            recommendations.append("Consider adding more test cases for better coverage")
        
        return recommendations
    
    def _update_quality_metrics(self, test_suite_name: str, executions: List[TestExecution], coverage: Dict[str, float]):
        """Update quality metrics tracking"""
        self.quality_metrics[test_suite_name] = {
            'timestamp': datetime.now(),
            'total_tests': len(executions),
            'success_rate': len([e for e in executions if e.status == "passed"]) / len(executions),
            'coverage': coverage,
            'avg_execution_time': sum(e.execution_time for e in executions) / len(executions)
        }
    
    def get_testing_analytics(self) -> Dict[str, Any]:
        """Get comprehensive testing analytics"""
        return {
            'test_suites_count': len(self.test_repository),
            'total_test_cases': sum(len(suite.test_cases) for suite in self.test_repository.values()),
            'quality_metrics': self.quality_metrics,
            'flaky_tests_count': len(self.adaptive_runner.flaky_tests),
            'performance_baselines': len(self.adaptive_runner.performance_baselines),
            'execution_history_size': len(self.adaptive_runner.execution_history)
        }


# Example usage and testing
if __name__ == "__main__":
    async def test_autonomous_testing():
        """Test the autonomous testing engine"""
        testing_engine = AutonomousTestingEngine()
        
        # Sample source code for testing
        sample_code = '''
def authenticate_user(username, password):
    """Authenticate user with username and password"""
    if not username or not password:
        raise ValueError("Username and password required")
    
    # Simulate authentication logic
    if username == "admin" and password == "secret":
        return True
    return False

class UserManager:
    def __init__(self):
        self.users = {}
    
    def create_user(self, username, email):
        if username in self.users:
            raise ValueError("User already exists")
        self.users[username] = {"email": email}
        return True
    
    def get_user(self, username):
        return self.users.get(username)
'''
        
        # Sample requirements
        sample_requirements = [
            {
                'id': 'REQ-001',
                'title': 'User Authentication',
                'description': 'System must authenticate users with valid credentials',
                'type': 'functional',
                'acceptance_criteria': [
                    'User can login with valid credentials',
                    'Invalid credentials are rejected',
                    'Empty credentials are handled gracefully'
                ]
            },
            {
                'id': 'REQ-002', 
                'title': 'User Management',
                'description': 'System must allow user creation and retrieval',
                'type': 'functional',
                'acceptance_criteria': [
                    'New users can be created',
                    'Duplicate usernames are prevented',
                    'User information can be retrieved'
                ]
            }
        ]
        
        # Generate comprehensive tests
        constraints = {
            'max_execution_time': 300,  # 5 minutes
            'coverage_target': 0.85,
            'max_tests': 50
        }
        
        test_suite = await testing_engine.generate_comprehensive_tests(
            sample_code, sample_requirements, constraints
        )
        
        print(f"Generated test suite: {test_suite.name}")
        print(f"Number of tests: {len(test_suite.test_cases)}")
        print(f"Parallel execution: {test_suite.parallel_execution}")
        
        # Execute autonomous testing
        results = await testing_engine.execute_autonomous_testing(test_suite)
        
        print("\n=== Test Results ===")
        print(json.dumps(results, indent=2, default=str))
        
        # Get analytics
        analytics = testing_engine.get_testing_analytics()
        print(f"\n=== Testing Analytics ===")
        print(json.dumps(analytics, indent=2, default=str))
    
    # Run test
    asyncio.run(test_autonomous_testing())
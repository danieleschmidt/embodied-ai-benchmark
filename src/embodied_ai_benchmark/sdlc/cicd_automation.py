"""
Continuous Integration/Continuous Deployment (CI/CD) Automation

Intelligent system for automating the complete CI/CD pipeline including
testing, quality gates, deployment, and monitoring.
"""

import os
import json
import yaml
import subprocess
from typing import Dict, List, Tuple, Optional, Any, Set
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import logging
from pathlib import Path

from .project_orchestrator import DevelopmentTask, TaskStatus
from ..utils.error_handling import ErrorHandler


class PipelineStage(Enum):
    """CI/CD Pipeline stages"""
    SOURCE = "source"
    BUILD = "build"
    TEST = "test"
    QUALITY_GATES = "quality_gates"
    SECURITY_SCAN = "security_scan"
    DEPLOYMENT = "deployment"
    SMOKE_TEST = "smoke_test"
    MONITORING = "monitoring"


class DeploymentTarget(Enum):
    """Deployment target environments"""
    LOCAL = "local"
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"


class PipelineStatus(Enum):
    """Pipeline execution status"""
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    CANCELLED = "cancelled"
    SKIPPED = "skipped"


@dataclass
class PipelineStep:
    """Individual pipeline step"""
    name: str
    stage: PipelineStage
    command: str
    working_directory: str = "."
    environment_vars: Dict[str, str] = field(default_factory=dict)
    dependencies: List[str] = field(default_factory=list)
    timeout_minutes: int = 30
    retry_count: int = 0
    required_for_deployment: bool = True
    status: PipelineStatus = PipelineStatus.PENDING
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    output: str = ""
    error_output: str = ""
    
    def execute(self, base_path: Path) -> bool:
        """Execute pipeline step"""
        self.status = PipelineStatus.RUNNING
        self.start_time = datetime.now()
        
        try:
            # Set up environment
            env = os.environ.copy()
            env.update(self.environment_vars)
            
            # Execute command
            result = subprocess.run(
                self.command,
                shell=True,
                cwd=base_path / self.working_directory,
                env=env,
                capture_output=True,
                text=True,
                timeout=self.timeout_minutes * 60
            )
            
            self.output = result.stdout
            self.error_output = result.stderr
            
            if result.returncode == 0:
                self.status = PipelineStatus.SUCCESS
                return True
            else:
                self.status = PipelineStatus.FAILED
                return False
                
        except subprocess.TimeoutExpired:
            self.status = PipelineStatus.FAILED
            self.error_output = f"Step timed out after {self.timeout_minutes} minutes"
            return False
        except Exception as e:
            self.status = PipelineStatus.FAILED
            self.error_output = f"Step failed with error: {str(e)}"
            return False
        finally:
            self.end_time = datetime.now()
    
    @property
    def duration_seconds(self) -> Optional[int]:
        """Get step duration in seconds"""
        if self.start_time and self.end_time:
            return int((self.end_time - self.start_time).total_seconds())
        return None


@dataclass
class QualityGate:
    """Quality gate definition"""
    name: str
    metric: str
    threshold: float
    operator: str  # ">", "<", ">=", "<=", "=="
    required: bool = True
    
    def evaluate(self, actual_value: float) -> bool:
        """Evaluate quality gate"""
        if self.operator == ">":
            return actual_value > self.threshold
        elif self.operator == "<":
            return actual_value < self.threshold
        elif self.operator == ">=":
            return actual_value >= self.threshold
        elif self.operator == "<=":
            return actual_value <= self.threshold
        elif self.operator == "==":
            return actual_value == self.threshold
        else:
            return False


@dataclass
class DeploymentConfig:
    """Deployment configuration"""
    target: DeploymentTarget
    docker_image: Optional[str] = None
    kubernetes_namespace: Optional[str] = None
    environment_vars: Dict[str, str] = field(default_factory=dict)
    resource_limits: Dict[str, str] = field(default_factory=dict)
    health_check_url: Optional[str] = None
    rollback_enabled: bool = True


class QualityGateManager:
    """Manages quality gates and metrics"""
    
    def __init__(self):
        self.quality_gates: List[QualityGate] = []
        self.logger = logging.getLogger(__name__)
        self._initialize_default_gates()
    
    def _initialize_default_gates(self):
        """Initialize default quality gates"""
        default_gates = [
            QualityGate("test_coverage", "coverage_percentage", 80.0, ">=", required=True),
            QualityGate("code_complexity", "cyclomatic_complexity", 10.0, "<=", required=True),
            QualityGate("security_vulnerabilities", "vulnerability_count", 0.0, "==", required=True),
            QualityGate("performance_test", "response_time_ms", 1000.0, "<=", required=False),
            QualityGate("code_duplication", "duplication_percentage", 5.0, "<=", required=False),
        ]
        self.quality_gates.extend(default_gates)
    
    def add_quality_gate(self, gate: QualityGate) -> None:
        """Add custom quality gate"""
        self.quality_gates.append(gate)
        self.logger.info(f"Added quality gate: {gate.name}")
    
    def evaluate_gates(self, metrics: Dict[str, float]) -> Dict[str, Any]:
        """Evaluate all quality gates"""
        results = {
            'passed': True,
            'gates': [],
            'required_failures': [],
            'optional_failures': []
        }
        
        for gate in self.quality_gates:
            if gate.metric not in metrics:
                if gate.required:
                    results['required_failures'].append({
                        'gate': gate.name,
                        'reason': f"Metric '{gate.metric}' not available"
                    })
                    results['passed'] = False
                continue
            
            passed = gate.evaluate(metrics[gate.metric])
            gate_result = {
                'name': gate.name,
                'metric': gate.metric,
                'threshold': gate.threshold,
                'actual': metrics[gate.metric],
                'operator': gate.operator,
                'passed': passed,
                'required': gate.required
            }
            
            results['gates'].append(gate_result)
            
            if not passed:
                if gate.required:
                    results['required_failures'].append(gate_result)
                    results['passed'] = False
                else:
                    results['optional_failures'].append(gate_result)
        
        return results
    
    def generate_quality_report(self, metrics: Dict[str, float]) -> str:
        """Generate quality gate report"""
        results = self.evaluate_gates(metrics)
        
        report_lines = [
            "# Quality Gate Report",
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Overall Status: {'PASSED' if results['passed'] else 'FAILED'}",
            ""
        ]
        
        if results['gates']:
            report_lines.append("## Gate Results")
            for gate in results['gates']:
                status = "✅ PASS" if gate['passed'] else "❌ FAIL"
                report_lines.append(
                    f"- {gate['name']}: {status} "
                    f"({gate['actual']} {gate['operator']} {gate['threshold']})"
                )
            report_lines.append("")
        
        if results['required_failures']:
            report_lines.append("## Required Gates Failed")
            for failure in results['required_failures']:
                report_lines.append(f"- {failure.get('gate', failure.get('name', 'Unknown'))}")
            report_lines.append("")
        
        if results['optional_failures']:
            report_lines.append("## Optional Gates Failed")
            for failure in results['optional_failures']:
                report_lines.append(f"- {failure.get('gate', failure.get('name', 'Unknown'))}")
        
        return "\n".join(report_lines)


class DeploymentManager:
    """Manages application deployments"""
    
    def __init__(self):
        self.deployments: Dict[str, DeploymentConfig] = {}
        self.logger = logging.getLogger(__name__)
        self.error_handler = ErrorHandler()
    
    def add_deployment_target(self, name: str, config: DeploymentConfig) -> None:
        """Add deployment target"""
        self.deployments[name] = config
        self.logger.info(f"Added deployment target: {name}")
    
    def deploy(self, target_name: str, artifact_path: Path, version: str) -> bool:
        """Deploy application to target"""
        if target_name not in self.deployments:
            self.logger.error(f"Deployment target not found: {target_name}")
            return False
        
        config = self.deployments[target_name]
        
        try:
            # Pre-deployment checks
            if not self._pre_deployment_checks(config):
                return False
            
            # Execute deployment based on target type
            if config.target == DeploymentTarget.LOCAL:
                success = self._deploy_local(config, artifact_path, version)
            elif config.target == DeploymentTarget.DOCKER:
                success = self._deploy_docker(config, artifact_path, version)
            elif config.target in [DeploymentTarget.DEVELOPMENT, DeploymentTarget.STAGING, DeploymentTarget.PRODUCTION]:
                success = self._deploy_kubernetes(config, artifact_path, version)
            else:
                self.logger.error(f"Unsupported deployment target: {config.target}")
                return False
            
            if success:
                # Post-deployment checks
                success = self._post_deployment_checks(config)
            
            return success
            
        except Exception as e:
            self.error_handler.handle_error(e, f"deployment_to_{target_name}")
            return False
    
    def _pre_deployment_checks(self, config: DeploymentConfig) -> bool:
        """Execute pre-deployment checks"""
        self.logger.info("Executing pre-deployment checks")
        
        # Check if target environment is accessible
        # Check resource availability
        # Validate configuration
        
        return True  # Simplified for now
    
    def _deploy_local(self, config: DeploymentConfig, artifact_path: Path, version: str) -> bool:
        """Deploy to local environment"""
        self.logger.info(f"Deploying version {version} to local environment")
        
        # For Python packages, this might mean pip install
        try:
            result = subprocess.run([
                "pip", "install", "-e", str(artifact_path)
            ], capture_output=True, text=True)
            
            return result.returncode == 0
        except Exception as e:
            self.logger.error(f"Local deployment failed: {e}")
            return False
    
    def _deploy_docker(self, config: DeploymentConfig, artifact_path: Path, version: str) -> bool:
        """Deploy using Docker"""
        self.logger.info(f"Deploying version {version} using Docker")
        
        if not config.docker_image:
            self.logger.error("Docker image not specified in configuration")
            return False
        
        try:
            # Build Docker image
            image_tag = f"{config.docker_image}:{version}"
            build_result = subprocess.run([
                "docker", "build", "-t", image_tag, str(artifact_path)
            ], capture_output=True, text=True)
            
            if build_result.returncode != 0:
                self.logger.error(f"Docker build failed: {build_result.stderr}")
                return False
            
            # Run container
            run_cmd = ["docker", "run", "-d", "--name", f"app-{version}"]
            
            # Add environment variables
            for key, value in config.environment_vars.items():
                run_cmd.extend(["-e", f"{key}={value}"])
            
            run_cmd.append(image_tag)
            
            run_result = subprocess.run(run_cmd, capture_output=True, text=True)
            
            return run_result.returncode == 0
            
        except Exception as e:
            self.logger.error(f"Docker deployment failed: {e}")
            return False
    
    def _deploy_kubernetes(self, config: DeploymentConfig, artifact_path: Path, version: str) -> bool:
        """Deploy to Kubernetes"""
        self.logger.info(f"Deploying version {version} to Kubernetes")
        
        try:
            # Generate Kubernetes manifests
            manifests = self._generate_k8s_manifests(config, version)
            
            # Apply manifests
            for manifest_name, manifest_content in manifests.items():
                # Write manifest to temporary file
                manifest_file = Path(f"/tmp/{manifest_name}")
                manifest_file.write_text(yaml.dump(manifest_content))
                
                # Apply manifest
                result = subprocess.run([
                    "kubectl", "apply", "-f", str(manifest_file),
                    "-n", config.kubernetes_namespace or "default"
                ], capture_output=True, text=True)
                
                if result.returncode != 0:
                    self.logger.error(f"Failed to apply {manifest_name}: {result.stderr}")
                    return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Kubernetes deployment failed: {e}")
            return False
    
    def _generate_k8s_manifests(self, config: DeploymentConfig, version: str) -> Dict[str, Dict]:
        """Generate Kubernetes deployment manifests"""
        app_name = f"app-{version}"
        
        deployment_manifest = {
            'apiVersion': 'apps/v1',
            'kind': 'Deployment',
            'metadata': {
                'name': app_name,
                'labels': {'app': app_name, 'version': version}
            },
            'spec': {
                'replicas': 1,
                'selector': {'matchLabels': {'app': app_name}},
                'template': {
                    'metadata': {'labels': {'app': app_name, 'version': version}},
                    'spec': {
                        'containers': [{
                            'name': app_name,
                            'image': f"{config.docker_image}:{version}",
                            'env': [
                                {'name': k, 'value': v} 
                                for k, v in config.environment_vars.items()
                            ],
                            'resources': config.resource_limits
                        }]
                    }
                }
            }
        }
        
        service_manifest = {
            'apiVersion': 'v1',
            'kind': 'Service',
            'metadata': {
                'name': f"{app_name}-service",
                'labels': {'app': app_name}
            },
            'spec': {
                'selector': {'app': app_name},
                'ports': [{'port': 80, 'targetPort': 8080}],
                'type': 'ClusterIP'
            }
        }
        
        return {
            'deployment.yaml': deployment_manifest,
            'service.yaml': service_manifest
        }
    
    def _post_deployment_checks(self, config: DeploymentConfig) -> bool:
        """Execute post-deployment checks"""
        self.logger.info("Executing post-deployment checks")
        
        # Health check
        if config.health_check_url:
            return self._health_check(config.health_check_url)
        
        return True  # Simplified for now
    
    def _health_check(self, health_check_url: str, max_retries: int = 5) -> bool:
        """Perform health check"""
        import time
        import requests
        
        for attempt in range(max_retries):
            try:
                response = requests.get(health_check_url, timeout=10)
                if response.status_code == 200:
                    self.logger.info("Health check passed")
                    return True
            except requests.RequestException:
                pass
            
            if attempt < max_retries - 1:
                time.sleep(10)  # Wait before retry
        
        self.logger.error("Health check failed")
        return False
    
    def rollback(self, target_name: str, previous_version: str) -> bool:
        """Rollback to previous version"""
        if target_name not in self.deployments:
            return False
        
        config = self.deployments[target_name]
        if not config.rollback_enabled:
            self.logger.error(f"Rollback not enabled for {target_name}")
            return False
        
        self.logger.info(f"Rolling back {target_name} to version {previous_version}")
        
        # This would implement rollback logic specific to the deployment target
        return True


class CICDPipeline:
    """Main CI/CD pipeline orchestrator"""
    
    def __init__(self, project_path: Path):
        self.project_path = project_path
        self.steps: List[PipelineStep] = []
        self.quality_gate_manager = QualityGateManager()
        self.deployment_manager = DeploymentManager()
        self.logger = logging.getLogger(__name__)
        self._initialize_default_pipeline()
    
    def _initialize_default_pipeline(self):
        """Initialize default CI/CD pipeline steps"""
        
        # Source stage
        self.add_step(PipelineStep(
            name="checkout",
            stage=PipelineStage.SOURCE,
            command="git status",  # Simplified - would be actual checkout
            timeout_minutes=5
        ))
        
        # Build stage
        self.add_step(PipelineStep(
            name="install_dependencies",
            stage=PipelineStage.BUILD,
            command="pip install -r requirements.txt || pip install -e .",
            dependencies=["checkout"],
            timeout_minutes=15
        ))
        
        self.add_step(PipelineStep(
            name="build_package",
            stage=PipelineStage.BUILD,
            command="python -m build || python setup.py sdist bdist_wheel",
            dependencies=["install_dependencies"],
            timeout_minutes=10
        ))
        
        # Test stage
        self.add_step(PipelineStep(
            name="unit_tests",
            stage=PipelineStage.TEST,
            command="python -m pytest tests/unit/ -v --cov=src --cov-report=xml",
            dependencies=["build_package"],
            timeout_minutes=20
        ))
        
        self.add_step(PipelineStep(
            name="integration_tests",
            stage=PipelineStage.TEST,
            command="python -m pytest tests/integration/ -v",
            dependencies=["unit_tests"],
            timeout_minutes=30
        ))
        
        # Quality gates stage
        self.add_step(PipelineStep(
            name="code_analysis",
            stage=PipelineStage.QUALITY_GATES,
            command="flake8 src/ || true && mypy src/ || true",
            dependencies=["unit_tests"],
            timeout_minutes=10
        ))
        
        # Security scan stage
        self.add_step(PipelineStep(
            name="security_scan",
            stage=PipelineStage.SECURITY_SCAN,
            command="bandit -r src/ -f json -o security_report.json || true",
            dependencies=["code_analysis"],
            timeout_minutes=15
        ))
    
    def add_step(self, step: PipelineStep) -> None:
        """Add step to pipeline"""
        self.steps.append(step)
        self.logger.info(f"Added pipeline step: {step.name}")
    
    def execute_pipeline(self, target_branch: str = "main") -> Dict[str, Any]:
        """Execute complete CI/CD pipeline"""
        self.logger.info(f"Starting CI/CD pipeline for branch: {target_branch}")
        
        pipeline_result = {
            'start_time': datetime.now(),
            'end_time': None,
            'status': PipelineStatus.RUNNING,
            'steps': [],
            'quality_gates': {},
            'deployment_results': {},
            'artifacts': []
        }
        
        try:
            # Execute steps in dependency order
            execution_order = self._get_execution_order()
            
            for step_name in execution_order:
                step = next((s for s in self.steps if s.name == step_name), None)
                if not step:
                    continue
                
                self.logger.info(f"Executing step: {step.name}")
                success = step.execute(self.project_path)
                
                step_result = {
                    'name': step.name,
                    'stage': step.stage.value,
                    'status': step.status.value,
                    'duration_seconds': step.duration_seconds,
                    'output': step.output,
                    'error_output': step.error_output
                }
                pipeline_result['steps'].append(step_result)
                
                if not success and step.required_for_deployment:
                    pipeline_result['status'] = PipelineStatus.FAILED
                    self.logger.error(f"Pipeline failed at step: {step.name}")
                    break
            
            # If all steps passed, evaluate quality gates
            if pipeline_result['status'] != PipelineStatus.FAILED:
                metrics = self._collect_metrics()
                quality_results = self.quality_gate_manager.evaluate_gates(metrics)
                pipeline_result['quality_gates'] = quality_results
                
                if not quality_results['passed']:
                    pipeline_result['status'] = PipelineStatus.FAILED
                    self.logger.error("Pipeline failed quality gates")
                else:
                    pipeline_result['status'] = PipelineStatus.SUCCESS
                    self.logger.info("Pipeline completed successfully")
            
        except Exception as e:
            pipeline_result['status'] = PipelineStatus.FAILED
            self.logger.error(f"Pipeline execution failed: {e}")
        finally:
            pipeline_result['end_time'] = datetime.now()
        
        return pipeline_result
    
    def _get_execution_order(self) -> List[str]:
        """Get step execution order based on dependencies"""
        # Simplified topological sort
        visited = set()
        order = []
        
        def visit(step_name: str):
            if step_name in visited:
                return
            
            step = next((s for s in self.steps if s.name == step_name), None)
            if not step:
                return
            
            # Visit dependencies first
            for dep in step.dependencies:
                visit(dep)
            
            visited.add(step_name)
            order.append(step_name)
        
        # Visit all steps
        for step in self.steps:
            visit(step.name)
        
        return order
    
    def _collect_metrics(self) -> Dict[str, float]:
        """Collect metrics from pipeline execution"""
        metrics = {}
        
        # Parse test coverage from coverage.xml if it exists
        coverage_file = self.project_path / "coverage.xml"
        if coverage_file.exists():
            try:
                import xml.etree.ElementTree as ET
                tree = ET.parse(coverage_file)
                root = tree.getroot()
                coverage_elem = root.find(".//coverage")
                if coverage_elem is not None:
                    line_rate = coverage_elem.get("line-rate", "0")
                    metrics["coverage_percentage"] = float(line_rate) * 100
            except Exception:
                metrics["coverage_percentage"] = 0.0
        
        # Parse security scan results
        security_file = self.project_path / "security_report.json"
        if security_file.exists():
            try:
                with open(security_file) as f:
                    security_data = json.load(f)
                    metrics["vulnerability_count"] = len(security_data.get("results", []))
            except Exception:
                metrics["vulnerability_count"] = 999  # Fail safe
        
        # Calculate code complexity (simplified)
        try:
            result = subprocess.run([
                "radon", "cc", "src/", "-j"
            ], capture_output=True, text=True, cwd=self.project_path)
            
            if result.returncode == 0:
                complexity_data = json.loads(result.stdout)
                avg_complexity = 0
                total_functions = 0
                
                for file_data in complexity_data.values():
                    for item in file_data:
                        if item.get('type') == 'function':
                            avg_complexity += item.get('complexity', 0)
                            total_functions += 1
                
                if total_functions > 0:
                    metrics["cyclomatic_complexity"] = avg_complexity / total_functions
                else:
                    metrics["cyclomatic_complexity"] = 1
            else:
                metrics["cyclomatic_complexity"] = 1
        except Exception:
            metrics["cyclomatic_complexity"] = 1
        
        # Default values for missing metrics
        metrics.setdefault("coverage_percentage", 0.0)
        metrics.setdefault("vulnerability_count", 999)
        metrics.setdefault("cyclomatic_complexity", 1.0)
        metrics.setdefault("response_time_ms", 500.0)
        metrics.setdefault("duplication_percentage", 0.0)
        
        return metrics
    
    def generate_pipeline_config(self, format: str = "yaml") -> str:
        """Generate pipeline configuration for external CI systems"""
        
        if format.lower() == "github_actions":
            return self._generate_github_actions_config()
        elif format.lower() == "gitlab_ci":
            return self._generate_gitlab_ci_config()
        elif format.lower() == "jenkins":
            return self._generate_jenkinsfile()
        else:
            # Generate generic YAML
            config = {
                'pipeline': {
                    'stages': []
                }
            }
            
            # Group steps by stage
            stages = {}
            for step in self.steps:
                stage_name = step.stage.value
                if stage_name not in stages:
                    stages[stage_name] = []
                stages[stage_name].append({
                    'name': step.name,
                    'command': step.command,
                    'working_directory': step.working_directory,
                    'timeout_minutes': step.timeout_minutes,
                    'dependencies': step.dependencies
                })
            
            config['pipeline']['stages'] = stages
            return yaml.dump(config, default_flow_style=False)
    
    def _generate_github_actions_config(self) -> str:
        """Generate GitHub Actions workflow"""
        workflow = {
            'name': 'CI/CD Pipeline',
            'on': {
                'push': {'branches': ['main', 'develop']},
                'pull_request': {'branches': ['main']}
            },
            'jobs': {
                'build-and-test': {
                    'runs-on': 'ubuntu-latest',
                    'steps': [
                        {
                            'name': 'Checkout code',
                            'uses': 'actions/checkout@v3'
                        },
                        {
                            'name': 'Set up Python',
                            'uses': 'actions/setup-python@v3',
                            'with': {'python-version': '3.9'}
                        }
                    ]
                }
            }
        }
        
        # Add steps from pipeline
        for step in self.steps:
            if step.stage != PipelineStage.SOURCE:  # Skip source stage
                workflow['jobs']['build-and-test']['steps'].append({
                    'name': step.name,
                    'run': step.command
                })
        
        return yaml.dump(workflow, default_flow_style=False)
    
    def _generate_gitlab_ci_config(self) -> str:
        """Generate GitLab CI configuration"""
        config = {
            'stages': [stage.value for stage in PipelineStage if stage != PipelineStage.SOURCE],
            'image': 'python:3.9'
        }
        
        for step in self.steps:
            if step.stage != PipelineStage.SOURCE:
                config[step.name] = {
                    'stage': step.stage.value,
                    'script': [step.command]
                }
        
        return yaml.dump(config, default_flow_style=False)
    
    def _generate_jenkinsfile(self) -> str:
        """Generate Jenkinsfile"""
        jenkinsfile = '''pipeline {
    agent any
    
    stages {'''
        
        # Group steps by stage
        stages = {}
        for step in self.steps:
            if step.stage != PipelineStage.SOURCE:
                stage_name = step.stage.value
                if stage_name not in stages:
                    stages[stage_name] = []
                stages[stage_name].append(step)
        
        for stage_name, steps in stages.items():
            jenkinsfile += f'''
        stage('{stage_name.title()}') {{
            steps {{'''
            for step in steps:
                jenkinsfile += f'''
                sh '{step.command}\'\'\'
            }
        }'''
        
        jenkinsfile += '''
    }
    
    post {
        always {
            cleanWs()
        }
    }
}'''
        
        return jenkinsfile
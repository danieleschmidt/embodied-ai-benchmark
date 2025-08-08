"""
Autonomous SDLC Orchestrator

Master orchestrator that coordinates all SDLC components for fully autonomous
software development lifecycle execution.
"""

import asyncio
import json
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
import logging

from .requirements_engine import RequirementsEngine, Requirement
from .code_generator import CodeGenerator, GeneratedCode, Language
from .project_orchestrator import ProjectOrchestrator, DevelopmentTask
from .cicd_automation import CICDPipeline, DeploymentManager
from .doc_generator import DocumentationGenerator
from .quality_assurance import QualityAssuranceEngine
from .security_monitor import SecurityMonitoringSystem
from ..utils.error_handling import ErrorHandler
from ..utils.monitoring import MetricsCollector
from ..utils.caching import AdaptiveCache


@dataclass
class AutonomousProject:
    """Autonomous project configuration"""
    name: str
    description: str
    target_language: Language = Language.PYTHON
    output_directory: Path = Path("./generated_project")
    requirements_input: str = ""
    stakeholders: List[str] = field(default_factory=list)
    compliance_standards: List[str] = field(default_factory=lambda: ["gdpr"])
    deployment_targets: List[str] = field(default_factory=lambda: ["local"])
    created_at: datetime = field(default_factory=datetime.now)


class AutonomousSDLCOrchestrator:
    """
    Master orchestrator for autonomous software development lifecycle.
    
    Coordinates all SDLC phases from requirements to deployment without human intervention.
    """
    
    def __init__(self, project_config: AutonomousProject):
        self.project = project_config
        self.logger = logging.getLogger(__name__)
        self.error_handler = ErrorHandler()
        self.metrics = MetricsCollector()
        self.cache = AdaptiveCache(max_size=1000)
        
        # Initialize SDLC components
        self.requirements_engine = RequirementsEngine()
        self.code_generator = CodeGenerator()
        self.project_orchestrator = ProjectOrchestrator()
        self.cicd_pipeline = CICDPipeline(project_config.output_directory)
        self.doc_generator = DocumentationGenerator()
        self.qa_engine = QualityAssuranceEngine()
        self.security_system = SecurityMonitoringSystem()
        
        # Execution state
        self.current_phase = "initialization"
        self.execution_log: List[Dict[str, Any]] = []
        self.project_artifacts: Dict[str, Any] = {}
        
    async def execute_autonomous_sdlc(self) -> Dict[str, Any]:
        """Execute complete autonomous SDLC process"""
        self.logger.info("🚀 Starting Autonomous SDLC Execution")
        
        execution_result = {
            'project_name': self.project.name,
            'start_time': datetime.now(),
            'phases_completed': [],
            'artifacts_generated': {},
            'quality_metrics': {},
            'deployment_status': {},
            'success': False
        }
        
        try:
            # Phase 1: Requirements Analysis & Generation
            requirements = await self._phase_requirements_analysis()
            execution_result['phases_completed'].append('requirements_analysis')
            
            # Phase 2: Project Planning & Task Decomposition
            project_plan = await self._phase_project_planning(requirements)
            execution_result['phases_completed'].append('project_planning')
            
            # Phase 3: Automated Code Generation
            generated_code = await self._phase_code_generation(requirements)
            execution_result['phases_completed'].append('code_generation')
            execution_result['artifacts_generated']['code_files'] = len(generated_code)
            
            # Phase 4: Quality Assurance & Testing
            qa_results = await self._phase_quality_assurance(requirements, generated_code)
            execution_result['phases_completed'].append('quality_assurance')
            execution_result['quality_metrics'] = qa_results
            
            # Phase 5: Security Analysis & Compliance
            security_results = await self._phase_security_analysis(generated_code)
            execution_result['phases_completed'].append('security_analysis')
            
            # Phase 6: Documentation Generation
            documentation = await self._phase_documentation_generation(requirements, generated_code)
            execution_result['phases_completed'].append('documentation_generation')
            execution_result['artifacts_generated']['docs_files'] = len(documentation)
            
            # Phase 7: CI/CD Pipeline Setup & Execution
            cicd_results = await self._phase_cicd_execution()
            execution_result['phases_completed'].append('cicd_execution')
            
            # Phase 8: Deployment & Monitoring Setup
            deployment_results = await self._phase_deployment()
            execution_result['phases_completed'].append('deployment')
            execution_result['deployment_status'] = deployment_results
            
            execution_result['success'] = True
            execution_result['end_time'] = datetime.now()
            execution_result['total_duration'] = str(execution_result['end_time'] - execution_result['start_time'])
            
            self.logger.info("✅ Autonomous SDLC execution completed successfully")
            
        except Exception as e:
            self.error_handler.handle_error(e, "autonomous_sdlc_execution")
            execution_result['error'] = str(e)
            execution_result['end_time'] = datetime.now()
            
        finally:
            # Generate final execution report
            await self._generate_execution_report(execution_result)
        
        return execution_result
    
    async def _phase_requirements_analysis(self) -> List[Requirement]:
        """Phase 1: Autonomous Requirements Analysis & Generation"""
        self.current_phase = "requirements_analysis"
        self.logger.info("📋 Phase 1: Requirements Analysis & Generation")
        
        # Generate requirements from input
        requirements = self.requirements_engine.analyze_user_input(
            self.project.requirements_input,
            context={'project_name': self.project.name}
        )
        
        # Add stakeholders if provided
        for stakeholder_name in self.project.stakeholders:
            stakeholder = self.requirements_engine.stakeholder_analyzer.stakeholders.get(stakeholder_name)
            if stakeholder:
                stakeholder_requirements = self.requirements_engine.stakeholder_analyzer.analyze_stakeholder_needs(stakeholder_name)
                for need in stakeholder_requirements:
                    req_list = self.requirements_engine.analyze_user_input(need)
                    requirements.extend(req_list)
        
        # Prioritize requirements
        if requirements:
            prioritized_requirements = self.requirements_engine.prioritizer.prioritize_requirements(
                requirements, 
                self.requirements_engine.stakeholder_analyzer.stakeholders
            )
            requirements = prioritized_requirements
        
        self.project_artifacts['requirements'] = requirements
        self.logger.info(f"Generated {len(requirements)} requirements")
        
        return requirements
    
    async def _phase_project_planning(self, requirements: List[Requirement]) -> Dict[str, Any]:
        """Phase 2: Autonomous Project Planning & Task Decomposition"""
        self.current_phase = "project_planning"
        self.logger.info("📊 Phase 2: Project Planning & Task Decomposition")
        
        # Create comprehensive project plan
        project_plan = self.project_orchestrator.create_project_plan(requirements)
        
        # Simulate project execution to identify risks
        simulation_results = self.project_orchestrator.simulate_project_execution()
        project_plan['simulation_results'] = simulation_results
        
        self.project_artifacts['project_plan'] = project_plan
        self.logger.info(f"Created project plan with {project_plan['summary']['total_tasks']} tasks")
        
        return project_plan
    
    async def _phase_code_generation(self, requirements: List[Requirement]) -> List[GeneratedCode]:
        """Phase 3: Autonomous Code Generation"""
        self.current_phase = "code_generation"
        self.logger.info("💻 Phase 3: Automated Code Generation")
        
        all_generated_code = []
        
        # Generate code for each requirement
        for requirement in requirements:
            code_artifacts = self.code_generator.generate_from_requirement(
                requirement, 
                self.project.target_language
            )
            
            # Optimize generated code
            for artifact in code_artifacts:
                optimized_artifact = self.code_generator.optimize_generated_code(artifact)
                all_generated_code.append(optimized_artifact)
        
        # Generate complete project structure
        project_structure = self.code_generator.generate_project_structure(
            requirements, 
            self.project.target_language
        )
        
        # Write generated code to filesystem
        await self._write_generated_code(all_generated_code, project_structure)
        
        self.project_artifacts['generated_code'] = all_generated_code
        self.project_artifacts['project_structure'] = project_structure
        self.logger.info(f"Generated {len(all_generated_code)} code artifacts")
        
        return all_generated_code
    
    async def _phase_quality_assurance(self, requirements: List[Requirement], generated_code: List[GeneratedCode]) -> Dict[str, Any]:
        """Phase 4: Autonomous Quality Assurance & Testing"""
        self.current_phase = "quality_assurance"
        self.logger.info("🔍 Phase 4: Quality Assurance & Testing")
        
        # Generate comprehensive quality report
        quality_report = self.qa_engine.comprehensive_quality_check(
            requirements,
            generated_code,
            self.project.output_directory
        )
        
        # Generate test cases for all requirements
        all_test_cases = []
        for requirement in requirements:
            test_cases = self.qa_engine.test_generator.generate_tests_from_requirement(requirement)
            all_test_cases.extend(test_cases)
        
        # Write test files
        test_files = self.qa_engine.write_test_files(
            all_test_cases, 
            self.project.output_directory / "tests"
        )
        
        # Run tests if possible
        test_results = self.qa_engine.run_tests(self.project.output_directory / "tests")
        
        qa_results = {
            'quality_report': quality_report.to_dict(),
            'test_cases_generated': len(all_test_cases),
            'test_files_written': len(test_files),
            'test_execution_results': test_results
        }
        
        self.project_artifacts['qa_results'] = qa_results
        self.logger.info(f"Quality assessment completed with score: {quality_report.overall_score:.2f}")
        
        return qa_results
    
    async def _phase_security_analysis(self, generated_code: List[GeneratedCode]) -> Dict[str, Any]:
        """Phase 5: Autonomous Security Analysis & Compliance"""
        self.current_phase = "security_analysis"
        self.logger.info("🔒 Phase 5: Security Analysis & Compliance")
        
        # Prepare code files for scanning
        code_files = []
        for artifact in generated_code:
            file_path = self.project.output_directory / artifact.filename
            if file_path.exists():
                code_files.append(str(file_path))
        
        # Perform comprehensive security scan
        security_results = self.security_system.comprehensive_security_scan(code_files)
        
        # Check compliance with specified standards
        compliance_results = []
        for code_artifact in generated_code:
            compliance_issues = self.security_system.compliance_checker.check_compliance(
                code_artifact.content, 
                self.project.compliance_standards
            )
            compliance_results.extend(compliance_issues)
        
        security_analysis = {
            'vulnerabilities_found': len(security_results['vulnerabilities']),
            'security_score': security_results['security_score'],
            'compliance_violations': len(compliance_results),
            'recommendations': security_results['recommendations']
        }
        
        self.project_artifacts['security_analysis'] = security_analysis
        self.logger.info(f"Security analysis completed. Score: {security_results['security_score']:.1f}")
        
        return security_analysis
    
    async def _phase_documentation_generation(self, requirements: List[Requirement], generated_code: List[GeneratedCode]) -> Dict[str, str]:
        """Phase 6: Autonomous Documentation Generation"""
        self.current_phase = "documentation_generation"
        self.logger.info("📚 Phase 6: Documentation Generation")
        
        # Generate comprehensive documentation
        documentation = self.doc_generator.generate_comprehensive_docs(
            self.project.output_directory,
            requirements,
            generated_code
        )
        
        # Write documentation files
        for doc_path, doc_content in documentation.items():
            full_path = self.project.output_directory / doc_path
            full_path.parent.mkdir(parents=True, exist_ok=True)
            full_path.write_text(doc_content)
        
        self.project_artifacts['documentation'] = documentation
        self.logger.info(f"Generated {len(documentation)} documentation files")
        
        return documentation
    
    async def _phase_cicd_execution(self) -> Dict[str, Any]:
        """Phase 7: Autonomous CI/CD Pipeline Setup & Execution"""
        self.current_phase = "cicd_execution"
        self.logger.info("🔄 Phase 7: CI/CD Pipeline Execution")
        
        # Execute CI/CD pipeline
        pipeline_results = self.cicd_pipeline.execute_pipeline("main")
        
        # Generate pipeline configs for different platforms
        github_config = self.cicd_pipeline.generate_pipeline_config("github_actions")
        gitlab_config = self.cicd_pipeline.generate_pipeline_config("gitlab_ci")
        
        # Write pipeline configurations
        (self.project.output_directory / ".github" / "workflows").mkdir(parents=True, exist_ok=True)
        (self.project.output_directory / ".github" / "workflows" / "ci.yml").write_text(github_config)
        (self.project.output_directory / ".gitlab-ci.yml").write_text(gitlab_config)
        
        cicd_results = {
            'pipeline_status': pipeline_results['status'].value,
            'steps_completed': len(pipeline_results['steps']),
            'quality_gates_passed': pipeline_results.get('quality_gates', {}).get('passed', False),
            'configs_generated': ['github_actions', 'gitlab_ci']
        }
        
        self.project_artifacts['cicd_results'] = cicd_results
        self.logger.info(f"CI/CD pipeline executed with status: {pipeline_results['status'].value}")
        
        return cicd_results
    
    async def _phase_deployment(self) -> Dict[str, Any]:
        """Phase 8: Autonomous Deployment & Monitoring Setup"""
        self.current_phase = "deployment"
        self.logger.info("🚀 Phase 8: Deployment & Monitoring Setup")
        
        deployment_results = {}
        
        # Deploy to specified targets
        for target in self.project.deployment_targets:
            if target == "local":
                # Local deployment
                success = await self._deploy_local()
                deployment_results[target] = {'success': success, 'type': 'local'}
            
            # Add other deployment targets as needed
        
        # Set up monitoring and alerting
        monitoring_config = self._setup_monitoring()
        deployment_results['monitoring'] = monitoring_config
        
        self.project_artifacts['deployment_results'] = deployment_results
        self.logger.info("Deployment phase completed")
        
        return deployment_results
    
    async def _write_generated_code(self, generated_code: List[GeneratedCode], project_structure: Dict[str, Any]):
        """Write generated code to filesystem"""
        self.project.output_directory.mkdir(parents=True, exist_ok=True)
        
        # Write individual code files
        for artifact in generated_code:
            file_path = self.project.output_directory / artifact.filename
            file_path.parent.mkdir(parents=True, exist_ok=True)
            file_path.write_text(artifact.content)
        
        # Write project structure files
        for file_path, content in project_structure.get('files', {}).items():
            full_path = self.project.output_directory / file_path
            full_path.parent.mkdir(parents=True, exist_ok=True)
            full_path.write_text(content)
        
        self.logger.info(f"Written {len(generated_code)} code files to {self.project.output_directory}")
    
    async def _deploy_local(self) -> bool:
        """Deploy to local environment"""
        try:
            # Create virtual environment and install dependencies
            import subprocess
            
            venv_path = self.project.output_directory / "venv"
            subprocess.run(
                ["python", "-m", "venv", str(venv_path)],
                check=True,
                capture_output=True
            )
            
            # Install project in development mode
            pip_path = venv_path / "bin" / "pip" if not venv_path.joinpath("Scripts").exists() else venv_path / "Scripts" / "pip"
            subprocess.run(
                [str(pip_path), "install", "-e", str(self.project.output_directory)],
                check=True,
                capture_output=True
            )
            
            return True
            
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Local deployment failed: {e}")
            return False
    
    def _setup_monitoring(self) -> Dict[str, Any]:
        """Setup monitoring and alerting"""
        monitoring_config = {
            'metrics_collection': True,
            'log_aggregation': True,
            'health_checks': True,
            'alerting_rules': [
                {'metric': 'error_rate', 'threshold': 0.05, 'action': 'alert'},
                {'metric': 'response_time', 'threshold': 1000, 'action': 'alert'},
                {'metric': 'memory_usage', 'threshold': 0.9, 'action': 'scale'}
            ]
        }
        
        # Write monitoring configuration
        monitoring_file = self.project.output_directory / "monitoring.json"
        monitoring_file.write_text(json.dumps(monitoring_config, indent=2))
        
        return monitoring_config
    
    async def _generate_execution_report(self, execution_result: Dict[str, Any]):
        """Generate comprehensive execution report"""
        report_lines = [
            "# Autonomous SDLC Execution Report",
            "",
            f"**Project:** {self.project.name}",
            f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"**Status:** {'SUCCESS' if execution_result['success'] else 'FAILED'}",
            "",
            "## Execution Summary",
            "",
            f"- **Start Time:** {execution_result['start_time'].strftime('%Y-%m-%d %H:%M:%S')}",
            f"- **End Time:** {execution_result.get('end_time', 'N/A')}",
            f"- **Duration:** {execution_result.get('total_duration', 'N/A')}",
            f"- **Phases Completed:** {len(execution_result['phases_completed'])}/8",
            "",
            "## Phases Executed",
            ""
        ]
        
        phase_names = {
            'requirements_analysis': 'Requirements Analysis & Generation',
            'project_planning': 'Project Planning & Task Decomposition', 
            'code_generation': 'Automated Code Generation',
            'quality_assurance': 'Quality Assurance & Testing',
            'security_analysis': 'Security Analysis & Compliance',
            'documentation_generation': 'Documentation Generation',
            'cicd_execution': 'CI/CD Pipeline Execution',
            'deployment': 'Deployment & Monitoring Setup'
        }
        
        for phase in execution_result['phases_completed']:
            phase_name = phase_names.get(phase, phase)
            report_lines.append(f"- ✅ {phase_name}")
        
        # Add artifacts summary
        artifacts = execution_result.get('artifacts_generated', {})
        if artifacts:
            report_lines.extend([
                "",
                "## Artifacts Generated",
                ""
            ])
            for artifact_type, count in artifacts.items():
                report_lines.append(f"- **{artifact_type.replace('_', ' ').title()}:** {count}")
        
        # Add quality metrics
        quality_metrics = execution_result.get('quality_metrics', {})
        if quality_metrics and quality_metrics.get('quality_report'):
            quality_report = quality_metrics['quality_report']
            report_lines.extend([
                "",
                "## Quality Metrics",
                "",
                f"- **Overall Quality Score:** {quality_report['overall_score']:.2f}/100",
                f"- **Test Cases Generated:** {quality_metrics.get('test_cases_generated', 0)}",
                f"- **Test Files Written:** {quality_metrics.get('test_files_written', 0)}"
            ])
        
        # Add deployment status
        deployment_status = execution_result.get('deployment_status', {})
        if deployment_status:
            report_lines.extend([
                "",
                "## Deployment Status",
                ""
            ])
            for target, status in deployment_status.items():
                if isinstance(status, dict) and 'success' in status:
                    status_icon = "✅" if status['success'] else "❌"
                    report_lines.append(f"- **{target.title()}:** {status_icon}")
        
        # Add error information if failed
        if not execution_result['success'] and 'error' in execution_result:
            report_lines.extend([
                "",
                "## Error Information",
                "",
                f"```",
                f"{execution_result['error']}",
                f"```"
            ])
        
        # Write report
        report_content = "\n".join(report_lines)
        report_file = self.project.output_directory / "EXECUTION_REPORT.md"
        report_file.write_text(report_content)
        
        self.logger.info(f"Execution report written to {report_file}")
    
    def get_current_status(self) -> Dict[str, Any]:
        """Get current execution status"""
        return {
            'project_name': self.project.name,
            'current_phase': self.current_phase,
            'artifacts_count': len(self.project_artifacts),
            'execution_log_entries': len(self.execution_log),
            'last_updated': datetime.now().isoformat()
        }
    
    def export_project_data(self) -> Dict[str, Any]:
        """Export all project data for analysis"""
        return {
            'project_config': {
                'name': self.project.name,
                'description': self.project.description,
                'target_language': self.project.target_language.value,
                'created_at': self.project.created_at.isoformat()
            },
            'artifacts': self.project_artifacts,
            'execution_log': self.execution_log,
            'current_status': self.get_current_status()
        }


# Main execution function
async def run_autonomous_sdlc(project_config: AutonomousProject) -> Dict[str, Any]:
    """
    Main entry point for autonomous SDLC execution.
    
    This function orchestrates the complete software development lifecycle
    from requirements analysis to deployment without human intervention.
    """
    orchestrator = AutonomousSDLCOrchestrator(project_config)
    return await orchestrator.execute_autonomous_sdlc()


# Example usage
if __name__ == "__main__":
    # Example project configuration
    example_project = AutonomousProject(
        name="AI-Powered Task Manager",
        description="An intelligent task management system with natural language processing",
        requirements_input="""
        Create a task management system that can:
        1. Accept tasks in natural language
        2. Automatically categorize and prioritize tasks
        3. Send intelligent reminders
        4. Generate productivity reports
        5. Integrate with calendar systems
        """,
        target_language=Language.PYTHON,
        deployment_targets=["local"]
    )
    
    # Run autonomous SDLC
    result = asyncio.run(run_autonomous_sdlc(example_project))
    print(f"Autonomous SDLC completed with status: {result['success']}")
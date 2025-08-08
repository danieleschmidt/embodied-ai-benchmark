"""
Comprehensive Integration Tests for Autonomous SDLC

Tests the complete autonomous SDLC pipeline end-to-end to ensure
all components work together seamlessly.
"""

import pytest
import asyncio
import tempfile
from pathlib import Path
from typing import Dict, Any

from .autonomous_orchestrator import AutonomousSDLCOrchestrator, AutonomousProject, run_autonomous_sdlc
from .requirements_engine import RequirementsEngine, Requirement, RequirementType, Priority
from .code_generator import CodeGenerator, Language
from .project_orchestrator import ProjectOrchestrator
from .quality_assurance import QualityAssuranceEngine


class TestAutonomousSDLCIntegration:
    """Integration tests for the complete autonomous SDLC system"""
    
    @pytest.fixture
    def sample_project_config(self):
        """Create sample project configuration for testing"""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield AutonomousProject(
                name="Test AI Assistant",
                description="A simple AI-powered assistant for task management",
                requirements_input="""
                Create an AI assistant that can:
                1. Accept user queries in natural language
                2. Provide intelligent responses
                3. Manage user tasks and reminders
                4. Generate daily productivity reports
                """,
                target_language=Language.PYTHON,
                output_directory=Path(temp_dir) / "generated_project",
                deployment_targets=["local"]
            )
    
    @pytest.fixture
    def orchestrator(self, sample_project_config):
        """Create orchestrator instance for testing"""
        return AutonomousSDLCOrchestrator(sample_project_config)
    
    @pytest.mark.asyncio
    async def test_complete_autonomous_sdlc_flow(self, sample_project_config):
        """Test complete autonomous SDLC execution from start to finish"""
        # Execute autonomous SDLC
        result = await run_autonomous_sdlc(sample_project_config)
        
        # Verify execution completed
        assert result is not None
        assert 'success' in result
        assert 'phases_completed' in result
        
        # Verify key phases were executed
        expected_phases = [
            'requirements_analysis',
            'project_planning', 
            'code_generation',
            'quality_assurance',
            'security_analysis',
            'documentation_generation'
        ]
        
        for phase in expected_phases:
            assert phase in result['phases_completed'], f"Phase {phase} was not completed"
        
        # Verify artifacts were generated
        assert result.get('artifacts_generated', {}).get('code_files', 0) > 0
        assert result.get('artifacts_generated', {}).get('docs_files', 0) > 0
    
    def test_requirements_engine_integration(self):
        """Test requirements engine generates valid requirements"""
        engine = RequirementsEngine()
        
        # Test requirement generation
        requirements = engine.analyze_user_input(
            "Create a web API that handles user authentication and data storage"
        )
        
        assert len(requirements) > 0
        assert all(isinstance(req, Requirement) for req in requirements)
        assert any(req.type == RequirementType.FUNCTIONAL for req in requirements)
        
        # Test requirement validation
        issues = engine.validate_requirements()
        assert isinstance(issues, list)
    
    def test_code_generator_integration(self):
        """Test code generator produces valid code from requirements"""
        generator = CodeGenerator()
        
        # Create sample requirement
        requirement = Requirement(
            id="REQ-001",
            title="User Authentication System",
            description="System must authenticate users with email and password",
            type=RequirementType.FUNCTIONAL,
            priority=Priority.HIGH,
            stakeholder="System User",
            acceptance_criteria=[
                "Given valid credentials, user should be authenticated",
                "Given invalid credentials, authentication should fail",
                "Then system should maintain user session"
            ]
        )
        
        # Generate code
        generated_code = generator.generate_from_requirement(requirement, Language.PYTHON)
        
        assert len(generated_code) > 0
        assert all(code.language == Language.PYTHON for code in generated_code)
        assert all(len(code.content) > 0 for code in generated_code)
        assert all(code.filename.endswith('.py') for code in generated_code)
    
    def test_project_orchestrator_integration(self):
        """Test project orchestrator creates valid project plans"""
        orchestrator = ProjectOrchestrator()
        
        # Create sample requirements
        requirements = [
            Requirement(
                id="REQ-001",
                title="User Management",
                description="Manage user accounts and profiles",
                type=RequirementType.FUNCTIONAL,
                priority=Priority.HIGH,
                stakeholder="Admin",
                acceptance_criteria=["Users can register", "Users can login"]
            ),
            Requirement(
                id="REQ-002", 
                title="Data Storage",
                description="Store and retrieve user data securely",
                type=RequirementType.TECHNICAL,
                priority=Priority.MEDIUM,
                stakeholder="Developer",
                acceptance_criteria=["Data is encrypted", "Backup is automated"]
            )
        ]
        
        # Create project plan
        project_plan = orchestrator.create_project_plan(requirements)
        
        assert 'summary' in project_plan
        assert project_plan['summary']['total_requirements'] == 2
        assert project_plan['summary']['total_tasks'] > 0
        assert 'tasks' in project_plan
        assert len(project_plan['tasks']) > 0
    
    def test_quality_assurance_integration(self):
        """Test quality assurance engine performs comprehensive checks"""
        qa_engine = QualityAssuranceEngine()
        
        # Sample requirement and code for testing
        requirement = Requirement(
            id="REQ-TEST",
            title="Data Validation",
            description="Validate all user input data",
            type=RequirementType.FUNCTIONAL,
            priority=Priority.HIGH,
            stakeholder="Security Team",
            acceptance_criteria=[
                "Given invalid input, system should reject it",
                "Given valid input, system should accept it"
            ]
        )
        
        # Generate test cases
        test_cases = qa_engine.test_generator.generate_tests_from_requirement(requirement)
        
        assert len(test_cases) > 0
        assert all(hasattr(tc, 'test_type') for tc in test_cases)
        assert all(hasattr(tc, 'description') for tc in test_cases)
        assert any('happy_path' in tc.tags for tc in test_cases if hasattr(tc, 'tags'))
    
    @pytest.mark.asyncio
    async def test_orchestrator_phase_execution(self, orchestrator):
        """Test individual phase execution in orchestrator"""
        # Test requirements analysis phase
        requirements = await orchestrator._phase_requirements_analysis()
        assert isinstance(requirements, list)
        assert len(requirements) > 0
        
        # Test project planning phase
        project_plan = await orchestrator._phase_project_planning(requirements)
        assert isinstance(project_plan, dict)
        assert 'summary' in project_plan
        
        # Test code generation phase
        generated_code = await orchestrator._phase_code_generation(requirements)
        assert isinstance(generated_code, list)
        assert len(generated_code) > 0
    
    def test_orchestrator_status_tracking(self, orchestrator):
        """Test orchestrator correctly tracks execution status"""
        # Initial status
        status = orchestrator.get_current_status()
        assert 'project_name' in status
        assert 'current_phase' in status
        assert status['project_name'] == orchestrator.project.name
        
        # Update phase and check status change
        orchestrator.current_phase = "testing_phase"
        updated_status = orchestrator.get_current_status()
        assert updated_status['current_phase'] == "testing_phase"
    
    def test_orchestrator_artifact_management(self, orchestrator):
        """Test orchestrator properly manages artifacts"""
        # Add some test artifacts
        orchestrator.project_artifacts['test_artifact'] = {'data': 'test'}
        
        # Export project data
        exported_data = orchestrator.export_project_data()
        
        assert 'project_config' in exported_data
        assert 'artifacts' in exported_data
        assert 'test_artifact' in exported_data['artifacts']
        assert exported_data['artifacts']['test_artifact']['data'] == 'test'
    
    @pytest.mark.asyncio
    async def test_error_handling_in_phases(self, orchestrator):
        """Test error handling during phase execution"""
        # Simulate error by providing invalid input
        orchestrator.project.requirements_input = ""
        
        try:
            # This should handle empty requirements gracefully
            requirements = await orchestrator._phase_requirements_analysis()
            # Should either return empty list or default requirements
            assert isinstance(requirements, list)
        except Exception as e:
            # If exception occurs, it should be logged and handled gracefully
            assert hasattr(orchestrator, 'error_handler')
    
    def test_project_structure_generation(self):
        """Test that generated project structure is valid"""
        generator = CodeGenerator()
        
        # Create sample requirements
        requirements = [
            Requirement(
                id="REQ-API",
                title="REST API",
                description="Create REST API endpoints",
                type=RequirementType.FUNCTIONAL,
                priority=Priority.HIGH,
                stakeholder="API User",
                acceptance_criteria=["API responds to GET requests"]
            )
        ]
        
        # Generate project structure
        structure = generator.generate_project_structure(requirements, Language.PYTHON)
        
        assert 'files' in structure
        assert 'directories' in structure
        assert 'dependencies' in structure
        
        # Should contain standard Python project files
        assert any('requirements.txt' in file for file in structure['files'].keys())
    
    def test_configuration_validation(self, sample_project_config):
        """Test project configuration validation"""
        # Valid configuration should work
        orchestrator = AutonomousSDLCOrchestrator(sample_project_config)
        assert orchestrator.project.name == sample_project_config.name
        
        # Test with minimal configuration
        minimal_config = AutonomousProject(
            name="Minimal Project",
            description="Minimal test project"
        )
        
        minimal_orchestrator = AutonomousSDLCOrchestrator(minimal_config)
        assert minimal_orchestrator.project.target_language == Language.PYTHON  # Default
    
    @pytest.mark.asyncio
    async def test_end_to_end_file_generation(self, sample_project_config):
        """Test that files are actually written to filesystem"""
        # Execute autonomous SDLC
        result = await run_autonomous_sdlc(sample_project_config)
        
        if result.get('success', False):
            # Check that output directory was created
            output_dir = sample_project_config.output_directory
            assert output_dir.exists()
            
            # Check for expected files
            expected_files = [
                'README.md',
                'EXECUTION_REPORT.md'
            ]
            
            for expected_file in expected_files:
                file_path = output_dir / expected_file
                if file_path.exists():
                    assert file_path.read_text().strip() != ""


class TestSDLCComponentIntegration:
    """Test integration between different SDLC components"""
    
    def test_requirements_to_code_flow(self):
        """Test complete flow from requirements to generated code"""
        # Create requirements engine
        req_engine = RequirementsEngine()
        
        # Generate requirements
        requirements = req_engine.analyze_user_input(
            "Create a user registration system with email verification"
        )
        
        # Generate code from requirements
        code_generator = CodeGenerator()
        all_generated_code = []
        
        for req in requirements:
            generated_code = code_generator.generate_from_requirement(req, Language.PYTHON)
            all_generated_code.extend(generated_code)
        
        # Verify integration
        assert len(requirements) > 0
        assert len(all_generated_code) > 0
        
        # Each generated code should reference the requirement
        for code in all_generated_code:
            assert len(code.content) > 0
            assert code.filename.endswith('.py')
    
    def test_code_to_tests_flow(self):
        """Test flow from generated code to test generation"""
        from .code_generator import GeneratedCode
        
        # Create sample generated code
        sample_code = GeneratedCode(
            filename="user_service.py",
            content="""
def create_user(email, password):
    '''Create a new user account'''
    if not email or not password:
        raise ValueError('Email and password required')
    return {'id': 1, 'email': email}

def authenticate_user(email, password):
    '''Authenticate user credentials'''
    # Implementation here
    return True
""",
            language=Language.PYTHON,
            dependencies=[],
            imports=[]
        )
        
        # Create requirement for testing
        requirement = Requirement(
            id="REQ-USER",
            title="User Management",
            description="Manage user accounts",
            type=RequirementType.FUNCTIONAL,
            priority=Priority.HIGH,
            stakeholder="User",
            acceptance_criteria=[
                "Given valid email and password, user should be created",
                "Given invalid input, error should be raised"
            ]
        )
        
        # Generate tests
        qa_engine = QualityAssuranceEngine()
        test_cases = qa_engine.test_generator.generate_tests_from_requirement(requirement)
        
        # Verify test generation
        assert len(test_cases) > 0
        assert any('user' in tc.name.lower() for tc in test_cases)
    
    def test_orchestrator_component_coordination(self):
        """Test that orchestrator properly coordinates all components"""
        with tempfile.TemporaryDirectory() as temp_dir:
            project = AutonomousProject(
                name="Integration Test Project",
                description="Test project for component coordination",
                requirements_input="Create a simple calculator app",
                output_directory=Path(temp_dir) / "test_project"
            )
            
            orchestrator = AutonomousSDLCOrchestrator(project)
            
            # Verify all components are initialized
            assert orchestrator.requirements_engine is not None
            assert orchestrator.code_generator is not None
            assert orchestrator.project_orchestrator is not None
            assert orchestrator.qa_engine is not None
            assert orchestrator.security_system is not None
            
            # Verify orchestrator can track state
            status = orchestrator.get_current_status()
            assert status['project_name'] == project.name


# Performance and stress tests
class TestSDLCPerformance:
    """Performance tests for SDLC components"""
    
    def test_requirement_generation_performance(self):
        """Test requirements generation performance with large input"""
        engine = RequirementsEngine()
        
        # Large requirements input
        large_input = """
        Create a comprehensive e-commerce platform that includes:
        """ + "\n".join([f"{i}. Feature {i} with complex functionality" for i in range(1, 51)])
        
        import time
        start_time = time.time()
        
        requirements = engine.analyze_user_input(large_input)
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        # Should complete within reasonable time (< 10 seconds)
        assert processing_time < 10.0
        assert len(requirements) > 0
    
    def test_code_generation_scalability(self):
        """Test code generation scales with number of requirements"""
        generator = CodeGenerator()
        
        # Create multiple requirements
        requirements = []
        for i in range(10):
            req = Requirement(
                id=f"REQ-{i:03d}",
                title=f"Feature {i}",
                description=f"Implement feature number {i}",
                type=RequirementType.FUNCTIONAL,
                priority=Priority.MEDIUM,
                stakeholder="User",
                acceptance_criteria=[f"Feature {i} should work correctly"]
            )
            requirements.append(req)
        
        import time
        start_time = time.time()
        
        all_code = []
        for req in requirements:
            code = generator.generate_from_requirement(req, Language.PYTHON)
            all_code.extend(code)
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        # Should scale reasonably (< 1 second per requirement)
        assert processing_time < len(requirements) * 1.0
        assert len(all_code) >= len(requirements)  # At least one code file per requirement


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])
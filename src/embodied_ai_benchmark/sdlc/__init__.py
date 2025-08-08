"""
Autonomous Software Development Life Cycle (SDLC) Module

Provides intelligent automation for the complete software development lifecycle:
- Requirements analysis and generation
- Automated code generation and scaffolding  
- Project orchestration and management
- Continuous integration and deployment
- Quality assurance and testing automation
- Documentation generation
"""

from .requirements_engine import RequirementsEngine, StakeholderAnalyzer, RequirementPrioritizer
from .code_generator import CodeGenerator, TemplateEngine, RefactoringEngine
from .project_orchestrator import ProjectOrchestrator, SprintPlanner, TaskDecomposer
from .cicd_automation import CICDPipeline, DeploymentManager, QualityGateManager
from .doc_generator import DocumentationGenerator, APIDocGenerator, ArchitectureDocGenerator
from .quality_assurance import QualityAssuranceEngine, TestGenerator, CodeReviewer

__version__ = "1.0.0"
__all__ = [
    # Requirements Management
    "RequirementsEngine",
    "StakeholderAnalyzer", 
    "RequirementPrioritizer",
    
    # Code Generation
    "CodeGenerator",
    "TemplateEngine",
    "RefactoringEngine",
    
    # Project Management
    "ProjectOrchestrator",
    "SprintPlanner",
    "TaskDecomposer",
    
    # CI/CD
    "CICDPipeline",
    "DeploymentManager", 
    "QualityGateManager",
    
    # Documentation
    "DocumentationGenerator",
    "APIDocGenerator",
    "ArchitectureDocGenerator",
    
    # Quality Assurance
    "QualityAssuranceEngine",
    "TestGenerator",
    "CodeReviewer",
]
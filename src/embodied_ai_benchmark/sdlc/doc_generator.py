"""
Documentation Generation Engine

Intelligent system for automatically generating comprehensive documentation
including API docs, architecture docs, user guides, and code documentation.
"""

import ast
import re
import json
from typing import Dict, List, Tuple, Optional, Any, Set
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
from pathlib import Path
import logging

from .requirements_engine import Requirement
from .code_generator import GeneratedCode, Language
from .project_orchestrator import DevelopmentTask


class DocumentationType(Enum):
    """Types of documentation"""
    API_REFERENCE = "api_reference"
    USER_GUIDE = "user_guide"
    ARCHITECTURE = "architecture"
    DEVELOPER_GUIDE = "developer_guide"
    DEPLOYMENT_GUIDE = "deployment_guide"
    CHANGELOG = "changelog"
    README = "readme"
    CONTRIBUTING = "contributing"


@dataclass
class DocumentationSection:
    """Individual documentation section"""
    title: str
    content: str
    level: int = 1  # Header level (1-6)
    subsections: List['DocumentationSection'] = field(default_factory=list)
    code_examples: List[str] = field(default_factory=list)
    images: List[str] = field(default_factory=list)
    
    def to_markdown(self, base_level: int = 0) -> str:
        """Convert section to Markdown"""
        lines = []
        header_level = min(base_level + self.level, 6)
        header_prefix = "#" * header_level
        
        lines.append(f"{header_prefix} {self.title}")
        lines.append("")
        
        if self.content:
            lines.append(self.content)
            lines.append("")
        
        # Add code examples
        for code in self.code_examples:
            lines.append("```python")
            lines.append(code)
            lines.append("```")
            lines.append("")
        
        # Add images
        for image in self.images:
            lines.append(f"![{self.title}]({image})")
            lines.append("")
        
        # Add subsections
        for subsection in self.subsections:
            lines.append(subsection.to_markdown(base_level + self.level))
            lines.append("")
        
        return "\n".join(lines)


@dataclass
class APIEndpoint:
    """API endpoint documentation"""
    method: str
    path: str
    summary: str
    description: str
    parameters: List[Dict[str, Any]] = field(default_factory=list)
    request_body: Optional[Dict[str, Any]] = None
    responses: Dict[int, Dict[str, Any]] = field(default_factory=dict)
    examples: List[Dict[str, Any]] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)


@dataclass
class ClassDocumentation:
    """Class documentation"""
    name: str
    description: str
    module_path: str
    methods: List[Dict[str, Any]] = field(default_factory=list)
    attributes: List[Dict[str, Any]] = field(default_factory=list)
    inheritance: List[str] = field(default_factory=list)
    examples: List[str] = field(default_factory=list)


class APIDocGenerator:
    """Generates API documentation"""
    
    def __init__(self):
        self.endpoints: List[APIEndpoint] = []
        self.logger = logging.getLogger(__name__)
    
    def analyze_code_for_apis(self, code_files: List[Path]) -> List[APIEndpoint]:
        """Analyze code files to extract API endpoints"""
        endpoints = []
        
        for file_path in code_files:
            if file_path.suffix == '.py':
                endpoints.extend(self._extract_flask_endpoints(file_path))
                endpoints.extend(self._extract_fastapi_endpoints(file_path))
        
        return endpoints
    
    def _extract_flask_endpoints(self, file_path: Path) -> List[APIEndpoint]:
        """Extract Flask API endpoints from Python file"""
        endpoints = []
        
        try:
            with open(file_path, 'r') as f:
                content = f.read()
            
            tree = ast.parse(content)
            
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    # Look for Flask route decorators
                    for decorator in node.decorator_list:
                        if self._is_flask_route_decorator(decorator):
                            endpoint = self._create_endpoint_from_flask_function(node, decorator)
                            if endpoint:
                                endpoints.append(endpoint)
        
        except Exception as e:
            self.logger.error(f"Error analyzing {file_path}: {e}")
        
        return endpoints
    
    def _is_flask_route_decorator(self, decorator: ast.AST) -> bool:
        """Check if decorator is a Flask route"""
        if isinstance(decorator, ast.Call):
            if isinstance(decorator.func, ast.Attribute):
                return decorator.func.attr == 'route'
        return False
    
    def _create_endpoint_from_flask_function(self, 
                                           func_node: ast.FunctionDef,
                                           route_decorator: ast.Call) -> Optional[APIEndpoint]:
        """Create API endpoint from Flask function"""
        try:
            # Extract route path
            if route_decorator.args:
                path = ast.literal_eval(route_decorator.args[0])
            else:
                return None
            
            # Extract HTTP methods
            methods = ['GET']  # default
            for keyword in route_decorator.keywords:
                if keyword.arg == 'methods':
                    methods = ast.literal_eval(keyword.value)
            
            # Extract docstring
            docstring = ast.get_docstring(func_node) or ""
            
            # Parse docstring for summary and description
            lines = docstring.split('\n')
            summary = lines[0] if lines else func_node.name
            description = '\n'.join(lines[1:]).strip() if len(lines) > 1 else summary
            
            # Create endpoint for each method
            primary_method = methods[0] if methods else 'GET'
            
            endpoint = APIEndpoint(
                method=primary_method,
                path=path,
                summary=summary,
                description=description,
                responses={200: {"description": "Success"}}
            )
            
            # Extract parameters from function signature
            endpoint.parameters = self._extract_function_parameters(func_node)
            
            return endpoint
        
        except Exception as e:
            self.logger.error(f"Error creating endpoint from function: {e}")
            return None
    
    def _extract_fastapi_endpoints(self, file_path: Path) -> List[APIEndpoint]:
        """Extract FastAPI endpoints from Python file"""
        # Similar to Flask but for FastAPI decorators
        # Implementation would be similar but looking for @app.get, @app.post, etc.
        return []
    
    def _extract_function_parameters(self, func_node: ast.FunctionDef) -> List[Dict[str, Any]]:
        """Extract function parameters for API documentation"""
        parameters = []
        
        for arg in func_node.args.args:
            if arg.arg not in ['self', 'cls']:  # Skip self/cls
                param = {
                    'name': arg.arg,
                    'in': 'query',  # default location
                    'type': 'string',  # default type
                    'required': True,
                    'description': f"Parameter {arg.arg}"
                }
                
                # Try to extract type from annotation
                if arg.annotation:
                    if isinstance(arg.annotation, ast.Name):
                        param['type'] = arg.annotation.id.lower()
                
                parameters.append(param)
        
        return parameters
    
    def generate_openapi_spec(self, 
                            title: str = "API Documentation",
                            version: str = "1.0.0") -> Dict[str, Any]:
        """Generate OpenAPI specification"""
        spec = {
            "openapi": "3.0.0",
            "info": {
                "title": title,
                "version": version,
                "description": "Auto-generated API documentation"
            },
            "paths": {}
        }
        
        # Group endpoints by path
        for endpoint in self.endpoints:
            path = endpoint.path
            if path not in spec["paths"]:
                spec["paths"][path] = {}
            
            spec["paths"][path][endpoint.method.lower()] = {
                "summary": endpoint.summary,
                "description": endpoint.description,
                "parameters": endpoint.parameters,
                "responses": endpoint.responses
            }
            
            if endpoint.request_body:
                spec["paths"][path][endpoint.method.lower()]["requestBody"] = endpoint.request_body
            
            if endpoint.tags:
                spec["paths"][path][endpoint.method.lower()]["tags"] = endpoint.tags
        
        return spec
    
    def generate_api_markdown(self) -> str:
        """Generate API documentation in Markdown format"""
        doc_lines = [
            "# API Documentation",
            "",
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "## Overview",
            "",
            "This document describes the available API endpoints.",
            ""
        ]
        
        # Group by tags
        endpoints_by_tag = {}
        for endpoint in self.endpoints:
            tags = endpoint.tags if endpoint.tags else ['Default']
            for tag in tags:
                if tag not in endpoints_by_tag:
                    endpoints_by_tag[tag] = []
                endpoints_by_tag[tag].append(endpoint)
        
        for tag, endpoints in endpoints_by_tag.items():
            doc_lines.append(f"## {tag}")
            doc_lines.append("")
            
            for endpoint in endpoints:
                doc_lines.append(f"### {endpoint.method} {endpoint.path}")
                doc_lines.append("")
                doc_lines.append(endpoint.description)
                doc_lines.append("")
                
                if endpoint.parameters:
                    doc_lines.append("**Parameters:**")
                    doc_lines.append("")
                    for param in endpoint.parameters:
                        required = " (required)" if param.get('required') else " (optional)"
                        doc_lines.append(f"- `{param['name']}` ({param['type']}){required}: {param['description']}")
                    doc_lines.append("")
                
                if endpoint.responses:
                    doc_lines.append("**Responses:**")
                    doc_lines.append("")
                    for code, response in endpoint.responses.items():
                        doc_lines.append(f"- `{code}`: {response.get('description', 'No description')}")
                    doc_lines.append("")
                
                if endpoint.examples:
                    doc_lines.append("**Examples:**")
                    doc_lines.append("")
                    for example in endpoint.examples:
                        doc_lines.append("```bash")
                        doc_lines.append(f"curl -X {endpoint.method} '{endpoint.path}'")
                        doc_lines.append("```")
                        doc_lines.append("")
        
        return "\n".join(doc_lines)


class ArchitectureDocGenerator:
    """Generates architecture documentation"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def generate_architecture_doc(self, 
                                requirements: List[Requirement],
                                code_structure: Dict[str, Any]) -> str:
        """Generate architecture documentation"""
        
        doc_lines = [
            "# Architecture Documentation",
            "",
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "## Overview",
            "",
            "This document describes the system architecture and design decisions.",
            "",
            "## System Components",
            ""
        ]
        
        # Analyze code structure
        components = self._analyze_components(code_structure)
        
        for component_name, component_info in components.items():
            doc_lines.append(f"### {component_name}")
            doc_lines.append("")
            doc_lines.append(component_info.get('description', f"Component: {component_name}"))
            doc_lines.append("")
            
            if component_info.get('dependencies'):
                doc_lines.append("**Dependencies:**")
                for dep in component_info['dependencies']:
                    doc_lines.append(f"- {dep}")
                doc_lines.append("")
        
        # Add architecture patterns
        doc_lines.extend([
            "## Architecture Patterns",
            "",
            "### Design Patterns Used",
            "",
            "- **Repository Pattern**: For data access abstraction",
            "- **Factory Pattern**: For object creation",
            "- **Observer Pattern**: For event handling",
            "- **Strategy Pattern**: For algorithm selection",
            "",
            "### Architectural Principles",
            "",
            "- **Separation of Concerns**: Each module has a single responsibility",
            "- **Dependency Injection**: Dependencies are injected rather than hardcoded",
            "- **Interface Segregation**: Interfaces are specific and focused",
            "- **Open/Closed Principle**: Open for extension, closed for modification",
            ""
        ])
        
        # Add quality attributes
        doc_lines.extend([
            "## Quality Attributes",
            "",
            "### Performance",
            "- Response time: < 200ms for API calls",
            "- Throughput: > 1000 requests per second",
            "- Memory usage: < 1GB for normal operations",
            "",
            "### Scalability",
            "- Horizontal scaling supported",
            "- Stateless design for easy clustering",
            "- Database sharding capability",
            "",
            "### Security",
            "- Authentication and authorization",
            "- Input validation and sanitization",
            "- Secure communication (HTTPS/TLS)",
            "",
            "### Reliability",
            "- Error handling and recovery",
            "- Circuit breaker pattern",
            "- Health checks and monitoring",
            ""
        ])
        
        return "\n".join(doc_lines)
    
    def _analyze_components(self, code_structure: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        """Analyze code structure to identify components"""
        components = {}
        
        # Extract components from directory structure
        if 'directories' in code_structure:
            for directory in code_structure['directories']:
                component_name = directory.split('/')[-1]
                components[component_name] = {
                    'description': f"Component handling {component_name} functionality",
                    'dependencies': []
                }
        
        # Add standard components
        standard_components = {
            'Core': {
                'description': 'Core business logic and domain models',
                'dependencies': []
            },
            'API': {
                'description': 'RESTful API endpoints and controllers',
                'dependencies': ['Core', 'Database']
            },
            'Database': {
                'description': 'Data persistence and repository layer',
                'dependencies': []
            },
            'Services': {
                'description': 'Business services and application logic',
                'dependencies': ['Core', 'Database']
            }
        }
        
        components.update(standard_components)
        return components
    
    def generate_deployment_diagram(self) -> str:
        """Generate deployment diagram in Mermaid format"""
        diagram = """
```mermaid
graph TB
    subgraph "Client Tier"
        WEB[Web Browser]
        MOBILE[Mobile App]
    end
    
    subgraph "Application Tier"
        LB[Load Balancer]
        API1[API Server 1]
        API2[API Server 2]
        CACHE[Redis Cache]
    end
    
    subgraph "Data Tier"
        DB[(Database)]
        BACKUP[(Backup)]
    end
    
    WEB --> LB
    MOBILE --> LB
    LB --> API1
    LB --> API2
    API1 --> CACHE
    API2 --> CACHE
    API1 --> DB
    API2 --> DB
    DB --> BACKUP
```
"""
        return diagram


class DocumentationGenerator:
    """Main documentation generator"""
    
    def __init__(self):
        self.api_doc_generator = APIDocGenerator()
        self.arch_doc_generator = ArchitectureDocGenerator()
        self.logger = logging.getLogger(__name__)
    
    def generate_comprehensive_docs(self, 
                                  project_path: Path,
                                  requirements: List[Requirement],
                                  generated_code: List[GeneratedCode]) -> Dict[str, str]:
        """Generate comprehensive documentation suite"""
        
        docs = {}
        
        # Generate README
        docs['README.md'] = self._generate_readme(project_path, requirements)
        
        # Generate API documentation
        code_files = list(project_path.rglob("*.py"))
        self.api_doc_generator.endpoints = self.api_doc_generator.analyze_code_for_apis(code_files)
        docs['docs/API.md'] = self.api_doc_generator.generate_api_markdown()
        
        # Generate architecture documentation
        code_structure = self._analyze_project_structure(project_path)
        docs['docs/ARCHITECTURE.md'] = self.arch_doc_generator.generate_architecture_doc(
            requirements, code_structure
        )
        
        # Generate user guide
        docs['docs/USER_GUIDE.md'] = self._generate_user_guide(requirements)
        
        # Generate developer guide
        docs['docs/DEVELOPER_GUIDE.md'] = self._generate_developer_guide(generated_code)
        
        # Generate deployment guide
        docs['docs/DEPLOYMENT.md'] = self._generate_deployment_guide()
        
        # Generate changelog
        docs['CHANGELOG.md'] = self._generate_changelog()
        
        # Generate contributing guide
        docs['CONTRIBUTING.md'] = self._generate_contributing_guide()
        
        self.logger.info(f"Generated {len(docs)} documentation files")
        return docs
    
    def _generate_readme(self, project_path: Path, requirements: List[Requirement]) -> str:
        """Generate README.md"""
        
        project_name = project_path.name.replace('_', ' ').title()
        
        readme_lines = [
            f"# {project_name}",
            "",
            "Auto-generated project with intelligent SDLC automation.",
            "",
            "## Overview",
            "",
            f"This project implements {len(requirements)} key requirements using modern development practices and automated workflows.",
            "",
            "## Features",
            ""
        ]
        
        # Add features from requirements
        for req in requirements[:10]:  # Limit to top 10
            readme_lines.append(f"- {req.title}")
        
        if len(requirements) > 10:
            readme_lines.append(f"- ... and {len(requirements) - 10} more features")
        
        readme_lines.extend([
            "",
            "## Quick Start",
            "",
            "```bash",
            "# Clone the repository",
            f"git clone <repository-url>",
            f"cd {project_path.name}",
            "",
            "# Install dependencies",
            "pip install -r requirements.txt",
            "",
            "# Run the application",
            "python -m src.main",
            "```",
            "",
            "## Project Structure",
            "",
            "```",
            f"{project_path.name}/",
            "├── src/                 # Source code",
            "├── tests/               # Test files",
            "├── docs/                # Documentation",
            "├── requirements.txt     # Dependencies",
            "└── README.md           # This file",
            "```",
            "",
            "## Documentation",
            "",
            "- [API Documentation](docs/API.md)",
            "- [Architecture Guide](docs/ARCHITECTURE.md)",
            "- [User Guide](docs/USER_GUIDE.md)",
            "- [Developer Guide](docs/DEVELOPER_GUIDE.md)",
            "- [Deployment Guide](docs/DEPLOYMENT.md)",
            "",
            "## Development",
            "",
            "### Prerequisites",
            "",
            "- Python 3.8+",
            "- pip or poetry",
            "",
            "### Setup Development Environment",
            "",
            "```bash",
            "# Create virtual environment",
            "python -m venv venv",
            "source venv/bin/activate  # On Windows: venv\\Scripts\\activate",
            "",
            "# Install development dependencies",
            "pip install -r requirements-dev.txt",
            "",
            "# Run tests",
            "pytest",
            "",
            "# Run linting",
            "flake8 src/",
            "```",
            "",
            "## Contributing",
            "",
            "Please read [CONTRIBUTING.md](CONTRIBUTING.md) for details on our code of conduct and the process for submitting pull requests.",
            "",
            "## License",
            "",
            "This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.",
            "",
            "## Acknowledgments",
            "",
            "- Built with autonomous SDLC automation",
            "- Generated documentation and code structure",
            "- Continuous integration and deployment ready",
        ])
        
        return "\n".join(readme_lines)
    
    def _generate_user_guide(self, requirements: List[Requirement]) -> str:
        """Generate user guide"""
        
        guide_lines = [
            "# User Guide",
            "",
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "## Getting Started",
            "",
            "This guide will help you get started with using the application.",
            "",
            "## Key Features",
            ""
        ]
        
        # Group requirements by type for better organization
        functional_reqs = [r for r in requirements if 'functional' in r.type.value]
        
        for req in functional_reqs[:5]:  # Top 5 functional requirements
            guide_lines.extend([
                f"### {req.title}",
                "",
                req.description,
                "",
                "**How to use:**",
                ""
            ])
            
            for criteria in req.acceptance_criteria[:3]:  # Top 3 criteria
                if 'when' in criteria.lower() or 'given' in criteria.lower():
                    guide_lines.append(f"1. {criteria}")
            
            guide_lines.append("")
        
        guide_lines.extend([
            "## Common Tasks",
            "",
            "### Task 1: Basic Setup",
            "",
            "1. Install the application",
            "2. Configure your settings",
            "3. Run initial setup",
            "",
            "### Task 2: Daily Operations",
            "",
            "1. Start the application",
            "2. Load your data",
            "3. Perform operations",
            "4. Review results",
            "",
            "## Troubleshooting",
            "",
            "### Common Issues",
            "",
            "**Issue**: Application won't start",
            "**Solution**: Check that all dependencies are installed",
            "",
            "**Issue**: Performance is slow",
            "**Solution**: Check system resources and configuration",
            "",
            "## Support",
            "",
            "For additional support, please:",
            "",
            "1. Check the [FAQ](FAQ.md)",
            "2. Search existing [issues](issues)",
            "3. Create a new issue if needed",
        ])
        
        return "\n".join(guide_lines)
    
    def _generate_developer_guide(self, generated_code: List[GeneratedCode]) -> str:
        """Generate developer guide"""
        
        guide_lines = [
            "# Developer Guide",
            "",
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "## Development Setup",
            "",
            "### Prerequisites",
            "",
            "- Python 3.8+",
            "- Git",
            "- Your favorite IDE",
            "",
            "### Local Development",
            "",
            "```bash",
            "# Clone repository",
            "git clone <repo-url>",
            "cd <project-name>",
            "",
            "# Setup virtual environment",
            "python -m venv venv",
            "source venv/bin/activate",
            "",
            "# Install dependencies",
            "pip install -r requirements-dev.txt",
            "",
            "# Run tests",
            "pytest",
            "```",
            "",
            "## Code Structure",
            "",
            f"The project contains {len(generated_code)} generated code files.",
            ""
        ]
        
        # Group by file type
        python_files = [gc for gc in generated_code if gc.language == Language.PYTHON]
        
        if python_files:
            guide_lines.extend([
                "### Python Modules",
                ""
            ])
            
            for code_file in python_files[:10]:  # Limit to 10 files
                guide_lines.append(f"- `{code_file.filename}`: Generated {code_file.language.value} code")
        
        guide_lines.extend([
            "",
            "## Testing",
            "",
            "### Running Tests",
            "",
            "```bash",
            "# Run all tests",
            "pytest",
            "",
            "# Run with coverage",
            "pytest --cov=src --cov-report=html",
            "",
            "# Run specific test file",
            "pytest tests/test_specific.py",
            "```",
            "",
            "### Writing Tests",
            "",
            "- Use pytest for testing framework",
            "- Follow AAA pattern (Arrange, Act, Assert)",
            "- Mock external dependencies",
            "- Aim for >80% code coverage",
            "",
            "## Code Style",
            "",
            "### Python Style Guide",
            "",
            "- Follow PEP 8",
            "- Use black for formatting",
            "- Use isort for import sorting",
            "- Use flake8 for linting",
            "",
            "```bash",
            "# Format code",
            "black src/",
            "",
            "# Sort imports",
            "isort src/",
            "",
            "# Lint code",
            "flake8 src/",
            "```",
            "",
            "## Adding Features",
            "",
            "### Development Workflow",
            "",
            "1. Create feature branch",
            "2. Write failing tests",
            "3. Implement feature",
            "4. Ensure tests pass",
            "5. Update documentation",
            "6. Submit pull request",
            "",
            "### Code Generation",
            "",
            "This project uses automated code generation. When adding new features:",
            "",
            "1. Update requirements specification",
            "2. Run code generator",
            "3. Review generated code",
            "4. Add custom logic as needed",
            "5. Generate tests",
            "",
            "## Debugging",
            "",
            "### Logging",
            "",
            "The application uses structured logging:",
            "",
            "```python",
            "import logging",
            "",
            "logger = logging.getLogger(__name__)",
            "logger.info('Information message')",
            "logger.error('Error message')",
            "```",
            "",
            "### IDE Setup",
            "",
            "Recommended VS Code extensions:",
            "",
            "- Python",
            "- Pylance", 
            "- Python Test Explorer",
            "- GitLens",
            "",
            "## Performance",
            "",
            "### Profiling",
            "",
            "```bash",
            "# Profile application",
            "python -m cProfile -o profile.stats main.py",
            "",
            "# Analyze profile",
            "python -c \"import pstats; pstats.Stats('profile.stats').sort_stats('cumulative').print_stats(10)\"",
            "```",
            "",
            "### Optimization Tips",
            "",
            "- Use appropriate data structures",
            "- Cache expensive operations",
            "- Profile before optimizing",
            "- Consider async/await for I/O operations",
        ])
        
        return "\n".join(guide_lines)
    
    def _generate_deployment_guide(self) -> str:
        """Generate deployment guide"""
        
        guide_lines = [
            "# Deployment Guide",
            "",
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "## Overview",
            "",
            "This guide covers deploying the application to various environments.",
            "",
            "## Prerequisites",
            "",
            "- Docker (optional)",
            "- Kubernetes (for production)",
            "- Cloud provider access (AWS/GCP/Azure)",
            "",
            "## Local Deployment",
            "",
            "### Using pip",
            "",
            "```bash",
            "# Install application",
            "pip install -e .",
            "",
            "# Run application",
            "python -m src.main",
            "```",
            "",
            "### Using Docker",
            "",
            "```bash",
            "# Build image",
            "docker build -t myapp:latest .",
            "",
            "# Run container",
            "docker run -p 8080:8080 myapp:latest",
            "```",
            "",
            "## Production Deployment",
            "",
            "### Environment Variables",
            "",
            "Required environment variables:",
            "",
            "```bash",
            "export DATABASE_URL='postgresql://...'",
            "export SECRET_KEY='your-secret-key'",
            "export DEBUG=False",
            "```",
            "",
            "### Using Docker Compose",
            "",
            "```yaml",
            "# docker-compose.yml",
            "version: '3.8'",
            "services:",
            "  app:",
            "    build: .",
            "    ports:",
            "      - '8080:8080'",
            "    environment:",
            "      - DATABASE_URL=postgresql://db:5432/myapp",
            "    depends_on:",
            "      - db",
            "",
            "  db:",
            "    image: postgres:13",
            "    environment:",
            "      - POSTGRES_DB=myapp",
            "      - POSTGRES_PASSWORD=password",
            "```",
            "",
            "### Kubernetes Deployment",
            "",
            "```bash",
            "# Apply configurations",
            "kubectl apply -f k8s/",
            "",
            "# Check deployment status",
            "kubectl get deployments",
            "",
            "# View logs",
            "kubectl logs -f deployment/myapp",
            "```",
            "",
            "## Monitoring",
            "",
            "### Health Checks",
            "",
            "The application provides health check endpoints:",
            "",
            "- `/health` - Basic health check",
            "- `/health/db` - Database connectivity check",
            "- `/metrics` - Prometheus metrics",
            "",
            "### Logging",
            "",
            "Logs are structured JSON format suitable for log aggregation systems:",
            "",
            "```json",
            "{",
            '  "timestamp": "2023-01-01T12:00:00Z",',
            '  "level": "INFO",',
            '  "message": "Application started",',
            '  "module": "main"',
            "}",
            "```",
            "",
            "## Security",
            "",
            "### HTTPS/TLS",
            "",
            "Always use HTTPS in production:",
            "",
            "```bash",
            "# Generate certificate (Let's Encrypt)",
            "certbot --nginx -d yourdomain.com",
            "```",
            "",
            "### Secrets Management",
            "",
            "Use environment variables or secret management systems:",
            "",
            "- Kubernetes Secrets",
            "- AWS Secrets Manager",
            "- Azure Key Vault",
            "- HashiCorp Vault",
            "",
            "## Backup and Recovery",
            "",
            "### Database Backup",
            "",
            "```bash",
            "# PostgreSQL backup",
            "pg_dump myapp > backup.sql",
            "",
            "# Restore backup",
            "psql myapp < backup.sql",
            "```",
            "",
            "### Application State",
            "",
            "- Store application state externally",
            "- Use database for persistent data",
            "- Cache can be rebuilt from database",
            "",
            "## Scaling",
            "",
            "### Horizontal Scaling",
            "",
            "```bash",
            "# Scale deployment",
            "kubectl scale deployment myapp --replicas=3",
            "```",
            "",
            "### Load Balancing",
            "",
            "- Use nginx or cloud load balancer",
            "- Configure health checks",
            "- Enable session stickiness if needed",
            "",
            "## Troubleshooting",
            "",
            "### Common Issues",
            "",
            "**Issue**: Application won't start",
            "**Solution**: Check environment variables and dependencies",
            "",
            "**Issue**: Database connection errors",
            "**Solution**: Verify database credentials and network connectivity",
            "",
            "**Issue**: High memory usage",
            "**Solution**: Check for memory leaks and optimize queries",
        ]
        
        return "\n".join(guide_lines)
    
    def _generate_changelog(self) -> str:
        """Generate changelog"""
        
        changelog_lines = [
            "# Changelog",
            "",
            "All notable changes to this project will be documented in this file.",
            "",
            "The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),",
            "and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).",
            "",
            f"## [1.0.0] - {datetime.now().strftime('%Y-%m-%d')}",
            "",
            "### Added",
            "- Initial release with autonomous SDLC features",
            "- Automated code generation from requirements",
            "- CI/CD pipeline automation",
            "- Comprehensive documentation generation",
            "- Quality gates and testing automation",
            "",
            "### Changed",
            "- N/A (initial release)",
            "",
            "### Deprecated",
            "- N/A (initial release)",
            "",
            "### Removed",
            "- N/A (initial release)",
            "",
            "### Fixed",
            "- N/A (initial release)",
            "",
            "### Security",
            "- Implemented secure coding practices",
            "- Added input validation and sanitization",
            "- Configured security scanning in CI/CD pipeline",
        ]
        
        return "\n".join(changelog_lines)
    
    def _generate_contributing_guide(self) -> str:
        """Generate contributing guide"""
        
        contributing_lines = [
            "# Contributing Guide",
            "",
            "Thank you for your interest in contributing to this project!",
            "",
            "## Code of Conduct",
            "",
            "This project follows the [Contributor Covenant](https://www.contributor-covenant.org/) code of conduct.",
            "",
            "## Getting Started",
            "",
            "### Prerequisites",
            "",
            "- Python 3.8+",
            "- Git",
            "- pytest for testing",
            "",
            "### Development Setup",
            "",
            "1. Fork the repository",
            "2. Clone your fork locally",
            "3. Create a virtual environment",
            "4. Install development dependencies",
            "",
            "```bash",
            "git clone https://github.com/yourusername/project-name.git",
            "cd project-name",
            "python -m venv venv",
            "source venv/bin/activate",
            "pip install -r requirements-dev.txt",
            "```",
            "",
            "## Development Workflow",
            "",
            "### 1. Create a Branch",
            "",
            "```bash",
            "git checkout -b feature/your-feature-name",
            "```",
            "",
            "### 2. Make Changes",
            "",
            "- Follow the coding standards",
            "- Write tests for new functionality",
            "- Update documentation as needed",
            "",
            "### 3. Test Your Changes",
            "",
            "```bash",
            "# Run tests",
            "pytest",
            "",
            "# Check code style",
            "flake8 src/",
            "black --check src/",
            "",
            "# Type checking",
            "mypy src/",
            "```",
            "",
            "### 4. Commit Changes",
            "",
            "Follow conventional commit format:",
            "",
            "```",
            "feat: add new feature",
            "fix: resolve bug issue",
            "docs: update documentation",
            "test: add missing tests",
            "refactor: improve code structure",
            "```",
            "",
            "### 5. Submit Pull Request",
            "",
            "1. Push your branch to your fork",
            "2. Create pull request to main branch",
            "3. Fill out pull request template",
            "4. Wait for review and address feedback",
            "",
            "## Coding Standards",
            "",
            "### Python Style",
            "",
            "- Follow PEP 8",
            "- Use type hints",
            "- Write docstrings for functions and classes",
            "- Maximum line length: 88 characters",
            "",
            "### Documentation",
            "",
            "- Update README.md for significant changes",
            "- Add docstrings to new functions/classes",
            "- Update API documentation if needed",
            "",
            "### Testing",
            "",
            "- Write unit tests for new functionality",
            "- Aim for >80% code coverage",
            "- Use meaningful test names",
            "- Mock external dependencies",
            "",
            "## Issue Guidelines",
            "",
            "### Bug Reports",
            "",
            "Include:",
            "- Clear description of the issue",
            "- Steps to reproduce",
            "- Expected vs actual behavior",
            "- Environment details",
            "",
            "### Feature Requests",
            "",
            "Include:",
            "- Clear description of the feature",
            "- Use case and motivation",
            "- Proposed implementation (optional)",
            "",
            "## Review Process",
            "",
            "1. All submissions require review",
            "2. Maintainers will provide feedback",
            "3. Address feedback and update PR",
            "4. Once approved, changes will be merged",
            "",
            "## Recognition",
            "",
            "Contributors will be recognized in:",
            "- CONTRIBUTORS.md file",
            "- Release notes",
            "- GitHub contributors list",
            "",
            "Thank you for contributing! 🎉",
        ]
        
        return "\n".join(contributing_lines)
    
    def _analyze_project_structure(self, project_path: Path) -> Dict[str, Any]:
        """Analyze project structure for documentation"""
        structure = {
            'directories': [],
            'files': [],
            'total_files': 0,
            'languages': set()
        }
        
        for item in project_path.rglob("*"):
            if item.is_dir():
                structure['directories'].append(str(item.relative_to(project_path)))
            elif item.is_file():
                structure['files'].append(str(item.relative_to(project_path)))
                structure['total_files'] += 1
                
                # Detect language from extension
                if item.suffix in ['.py']:
                    structure['languages'].add('Python')
                elif item.suffix in ['.js', '.ts']:
                    structure['languages'].add('JavaScript/TypeScript')
                elif item.suffix in ['.java']:
                    structure['languages'].add('Java')
        
        structure['languages'] = list(structure['languages'])
        return structure
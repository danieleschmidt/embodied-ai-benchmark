"""
Automated Code Generation Framework

Intelligent system for generating code from requirements, templates,
and architectural patterns. Supports multiple languages and frameworks.
"""

import re
import os
import ast
import json
from typing import Dict, List, Tuple, Optional, Any, Set
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
from pathlib import Path
import logging
from abc import ABC, abstractmethod

from .requirements_engine import Requirement, RequirementType


class Language(Enum):
    """Supported programming languages"""
    PYTHON = "python"
    JAVASCRIPT = "javascript"
    TYPESCRIPT = "typescript"
    JAVA = "java"
    GO = "go"
    RUST = "rust"
    CPP = "cpp"


class TemplateType(Enum):
    """Code template types"""
    CLASS = "class"
    FUNCTION = "function"
    MODULE = "module"
    API_ENDPOINT = "api_endpoint"
    DATABASE_MODEL = "database_model"
    TEST = "test"
    CONFIGURATION = "configuration"
    INTERFACE = "interface"


@dataclass
class CodeTemplate:
    """Code generation template"""
    name: str
    type: TemplateType
    language: Language
    template_string: str
    required_variables: Set[str]
    optional_variables: Dict[str, Any] = field(default_factory=dict)
    dependencies: List[str] = field(default_factory=list)
    imports: List[str] = field(default_factory=list)
    
    def render(self, variables: Dict[str, Any]) -> str:
        """Render template with provided variables"""
        # Check required variables
        missing_vars = self.required_variables - set(variables.keys())
        if missing_vars:
            raise ValueError(f"Missing required variables: {missing_vars}")
        
        # Merge with optional variables
        all_vars = {**self.optional_variables, **variables}
        
        # Render template
        try:
            return self.template_string.format(**all_vars)
        except KeyError as e:
            raise ValueError(f"Template variable not provided: {e}")


@dataclass
class GeneratedCode:
    """Generated code artifact"""
    filename: str
    content: str
    language: Language
    dependencies: List[str]
    imports: List[str]
    tests_needed: bool = True
    documentation_needed: bool = True
    created_at: datetime = field(default_factory=datetime.now)


class TemplateEngine:
    """Manages code templates and rendering"""
    
    def __init__(self):
        self.templates: Dict[str, CodeTemplate] = {}
        self.logger = logging.getLogger(__name__)
        self._initialize_builtin_templates()
    
    def _initialize_builtin_templates(self):
        """Initialize built-in code templates"""
        # Python class template
        python_class_template = CodeTemplate(
            name="python_class",
            type=TemplateType.CLASS,
            language=Language.PYTHON,
            template_string='''"""
{class_description}
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
import logging

{additional_imports}


@dataclass
class {class_name}:
    """{class_docstring}"""
    
{class_fields}
    
    def __post_init__(self):
        """Initialize instance after creation"""
        self.logger = logging.getLogger(self.__class__.__name__)
        self.created_at = datetime.now()
    
{class_methods}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert instance to dictionary"""
        return {{
{dict_fields}
        }}
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> '{class_name}':
        """Create instance from dictionary"""
        return cls(**data)
''',
            required_variables={'class_name', 'class_description', 'class_docstring'},
            optional_variables={
                'additional_imports': '',
                'class_fields': '    pass',
                'class_methods': '',
                'dict_fields': ''
            }
        )
        self.add_template(python_class_template)
        
        # Python function template
        python_function_template = CodeTemplate(
            name="python_function",
            type=TemplateType.FUNCTION,
            language=Language.PYTHON,
            template_string='''def {function_name}({function_args}) -> {return_type}:
    """
    {function_description}
    
    Args:
{arg_docs}
    
    Returns:
        {return_description}
    
    Raises:
{exception_docs}
    """
    logger = logging.getLogger(__name__)
    logger.debug(f"Executing {{function_name}} with args: {{locals()}}")
    
    try:
{function_body}
    except Exception as e:
        logger.error(f"Error in {{function_name}}: {{e}}")
        raise
''',
            required_variables={'function_name', 'function_description', 'return_type'},
            optional_variables={
                'function_args': '',
                'return_description': 'Function result',
                'arg_docs': '',
                'exception_docs': '        Exception: If operation fails',
                'function_body': '        pass'
            }
        )
        self.add_template(python_function_template)
        
        # API endpoint template
        api_endpoint_template = CodeTemplate(
            name="python_api_endpoint",
            type=TemplateType.API_ENDPOINT,
            language=Language.PYTHON,
            template_string='''from flask import Flask, request, jsonify
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)


@app.route('{endpoint_path}', methods=['{http_method}'])
def {endpoint_name}({endpoint_args}):
    """
    {endpoint_description}
    
    Returns:
        JSON response with {response_description}
    """
    try:
        # Validate input
{input_validation}
        
        # Process request
{request_processing}
        
        # Return response
        return jsonify({{
            'success': True,
            'data': result,
            'timestamp': datetime.now().isoformat()
        }}), 200
        
    except ValueError as e:
        logger.warning(f"Validation error in {{endpoint_name}}: {{e}}")
        return jsonify({{'error': str(e)}}), 400
        
    except Exception as e:
        logger.error(f"Error in {{endpoint_name}}: {{e}}")
        return jsonify({{'error': 'Internal server error'}}), 500
''',
            required_variables={'endpoint_path', 'endpoint_name', 'endpoint_description'},
            optional_variables={
                'http_method': 'GET',
                'endpoint_args': '',
                'response_description': 'operation result',
                'input_validation': '        pass',
                'request_processing': '        result = {}',
            }
        )
        self.add_template(api_endpoint_template)
        
        # Database model template
        db_model_template = CodeTemplate(
            name="python_db_model",
            type=TemplateType.DATABASE_MODEL,
            language=Language.PYTHON,
            template_string='''from sqlalchemy import Column, Integer, String, DateTime, Text, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from datetime import datetime
from typing import Dict, Any

Base = declarative_base()


class {model_name}(Base):
    """
    {model_description}
    
    Database table: {table_name}
    """
    __tablename__ = '{table_name}'
    
    # Primary key
    id = Column(Integer, primary_key=True, autoincrement=True)
    
    # Fields
{model_fields}
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
    
{relationships}
    
    def __repr__(self) -> str:
        return f"<{model_name}(id={{self.id}}{repr_fields})>"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert model to dictionary"""
        return {{
            'id': self.id,
{dict_fields}
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None
        }}
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> '{model_name}':
        """Create model from dictionary"""
        # Remove timestamps and id for creation
        clean_data = {{k: v for k, v in data.items() 
                     if k not in ['id', 'created_at', 'updated_at']}}
        return cls(**clean_data)
''',
            required_variables={'model_name', 'table_name', 'model_description'},
            optional_variables={
                'model_fields': '',
                'relationships': '',
                'repr_fields': '',
                'dict_fields': ''
            }
        )
        self.add_template(db_model_template)
    
    def add_template(self, template: CodeTemplate) -> None:
        """Add template to engine"""
        self.templates[template.name] = template
        self.logger.info(f"Added template: {template.name}")
    
    def get_template(self, name: str) -> Optional[CodeTemplate]:
        """Get template by name"""
        return self.templates.get(name)
    
    def list_templates(self, language: Optional[Language] = None, 
                      template_type: Optional[TemplateType] = None) -> List[CodeTemplate]:
        """List available templates with optional filters"""
        templates = list(self.templates.values())
        
        if language:
            templates = [t for t in templates if t.language == language]
        
        if template_type:
            templates = [t for t in templates if t.type == template_type]
        
        return templates


class CodeAnalyzer(ABC):
    """Base class for code analysis and extraction"""
    
    @abstractmethod
    def extract_components(self, code: str) -> Dict[str, Any]:
        """Extract code components for analysis"""
        pass
    
    @abstractmethod
    def analyze_complexity(self, code: str) -> int:
        """Analyze code complexity"""
        pass


class PythonCodeAnalyzer(CodeAnalyzer):
    """Python code analyzer"""
    
    def extract_components(self, code: str) -> Dict[str, Any]:
        """Extract Python code components"""
        try:
            tree = ast.parse(code)
            components = {
                'classes': [],
                'functions': [],
                'imports': [],
                'constants': []
            }
            
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    components['classes'].append({
                        'name': node.name,
                        'methods': [n.name for n in node.body if isinstance(n, ast.FunctionDef)],
                        'bases': [base.id if isinstance(base, ast.Name) else str(base) for base in node.bases]
                    })
                elif isinstance(node, ast.FunctionDef):
                    components['functions'].append({
                        'name': node.name,
                        'args': [arg.arg for arg in node.args.args],
                        'decorators': [d.id if isinstance(d, ast.Name) else str(d) for d in node.decorator_list]
                    })
                elif isinstance(node, ast.Import):
                    components['imports'].extend([alias.name for alias in node.names])
                elif isinstance(node, ast.ImportFrom):
                    module = node.module or ''
                    for alias in node.names:
                        components['imports'].append(f"{module}.{alias.name}")
            
            return components
            
        except SyntaxError as e:
            return {'error': f"Syntax error: {e}"}
    
    def analyze_complexity(self, code: str) -> int:
        """Calculate cyclomatic complexity"""
        try:
            tree = ast.parse(code)
            complexity = 1  # Base complexity
            
            for node in ast.walk(tree):
                if isinstance(node, (ast.If, ast.While, ast.For, ast.With)):
                    complexity += 1
                elif isinstance(node, ast.ExceptHandler):
                    complexity += 1
                elif isinstance(node, ast.BoolOp):
                    complexity += len(node.values) - 1
            
            return complexity
            
        except SyntaxError:
            return 0


class RefactoringEngine:
    """Handles code refactoring and optimization"""
    
    def __init__(self):
        self.analyzers = {
            Language.PYTHON: PythonCodeAnalyzer()
        }
        self.logger = logging.getLogger(__name__)
    
    def suggest_improvements(self, code: str, language: Language) -> List[Dict[str, str]]:
        """Suggest code improvements"""
        suggestions = []
        
        if language not in self.analyzers:
            return suggestions
        
        analyzer = self.analyzers[language]
        components = analyzer.extract_components(code)
        complexity = analyzer.analyze_complexity(code)
        
        # Complexity suggestions
        if complexity > 10:
            suggestions.append({
                'type': 'complexity',
                'message': f"Function has high complexity ({complexity}). Consider breaking into smaller functions.",
                'severity': 'warning'
            })
        
        # Length suggestions
        line_count = len(code.split('\n'))
        if line_count > 100:
            suggestions.append({
                'type': 'length',
                'message': f"Function is very long ({line_count} lines). Consider refactoring.",
                'severity': 'info'
            })
        
        # Python-specific suggestions
        if language == Language.PYTHON:
            self._add_python_suggestions(code, suggestions)
        
        return suggestions
    
    def _add_python_suggestions(self, code: str, suggestions: List[Dict[str, str]]):
        """Add Python-specific improvement suggestions"""
        # Check for common anti-patterns
        if 'except:' in code:
            suggestions.append({
                'type': 'exception_handling',
                'message': 'Avoid bare except clauses. Catch specific exceptions.',
                'severity': 'warning'
            })
        
        if re.search(r'print\(', code):
            suggestions.append({
                'type': 'logging',
                'message': 'Consider using logging instead of print statements.',
                'severity': 'info'
            })
        
        if re.search(r'[a-zA-Z_][a-zA-Z0-9_]*\s*=\s*\[\]', code):
            suggestions.append({
                'type': 'mutable_defaults',
                'message': 'Consider using dataclass or factory functions for mutable defaults.',
                'severity': 'info'
            })
    
    def apply_refactoring(self, code: str, refactoring_type: str) -> str:
        """Apply specific refactoring patterns"""
        if refactoring_type == 'extract_method':
            return self._extract_method(code)
        elif refactoring_type == 'add_logging':
            return self._add_logging(code)
        elif refactoring_type == 'add_error_handling':
            return self._add_error_handling(code)
        else:
            return code
    
    def _extract_method(self, code: str) -> str:
        """Extract method refactoring (simplified)"""
        # This is a simplified implementation
        # In practice, would use more sophisticated AST manipulation
        return code
    
    def _add_logging(self, code: str) -> str:
        """Add logging to code"""
        if 'import logging' not in code:
            code = 'import logging\n\n' + code
        
        # Add logger initialization
        if 'logger = logging.getLogger' not in code:
            code = code.replace(
                'def ', 
                '    logger = logging.getLogger(__name__)\n    def ', 1
            )
        
        return code
    
    def _add_error_handling(self, code: str) -> str:
        """Add basic error handling"""
        # Wrap function body in try-catch if not present
        if 'try:' not in code and 'def ' in code:
            lines = code.split('\n')
            for i, line in enumerate(lines):
                if line.strip().startswith('def ') and ':' in line:
                    # Insert try block after function definition
                    indent = len(line) - len(line.lstrip())
                    lines.insert(i + 1, ' ' * (indent + 4) + 'try:')
                    # Add except block at end (simplified)
                    lines.append(' ' * (indent + 4) + 'except Exception as e:')
                    lines.append(' ' * (indent + 8) + 'logger.error(f"Error: {e}")')
                    lines.append(' ' * (indent + 8) + 'raise')
                    break
            code = '\n'.join(lines)
        
        return code


class CodeGenerator:
    """Main code generation engine"""
    
    def __init__(self):
        self.template_engine = TemplateEngine()
        self.refactoring_engine = RefactoringEngine()
        self.logger = logging.getLogger(__name__)
    
    def generate_from_requirement(self, 
                                requirement: Requirement,
                                target_language: Language = Language.PYTHON) -> List[GeneratedCode]:
        """Generate code artifacts from requirement"""
        generated_code = []
        
        try:
            # Determine what to generate based on requirement type
            if requirement.type == RequirementType.FUNCTIONAL:
                generated_code.extend(self._generate_functional_code(requirement, target_language))
            elif requirement.type == RequirementType.TECHNICAL:
                generated_code.extend(self._generate_technical_code(requirement, target_language))
            elif requirement.type in [RequirementType.BUSINESS, RequirementType.NON_FUNCTIONAL]:
                generated_code.extend(self._generate_business_logic_code(requirement, target_language))
            
            self.logger.info(f"Generated {len(generated_code)} code artifacts for {requirement.id}")
            return generated_code
            
        except Exception as e:
            self.logger.error(f"Error generating code for {requirement.id}: {e}")
            return []
    
    def _generate_functional_code(self, requirement: Requirement, language: Language) -> List[GeneratedCode]:
        """Generate code for functional requirements"""
        code_artifacts = []
        
        # Extract key information from requirement
        class_name = self._extract_class_name(requirement.title)
        module_name = self._camel_to_snake(class_name)
        
        # Generate main class
        if language == Language.PYTHON:
            template = self.template_engine.get_template("python_class")
            if template:
                variables = {
                    'class_name': class_name,
                    'class_description': requirement.description,
                    'class_docstring': requirement.title,
                    'class_fields': self._generate_class_fields(requirement),
                    'class_methods': self._generate_class_methods(requirement),
                    'dict_fields': self._generate_dict_fields(requirement)
                }
                
                code = template.render(variables)
                code_artifacts.append(GeneratedCode(
                    filename=f"{module_name}.py",
                    content=code,
                    language=language,
                    dependencies=template.dependencies,
                    imports=template.imports
                ))
        
        # Generate API endpoint if needed
        if 'api' in requirement.description.lower() or 'endpoint' in requirement.description.lower():
            api_code = self._generate_api_endpoint(requirement, language)
            if api_code:
                code_artifacts.append(api_code)
        
        # Generate database model if needed
        if 'database' in requirement.description.lower() or 'model' in requirement.description.lower():
            db_code = self._generate_database_model(requirement, language)
            if db_code:
                code_artifacts.append(db_code)
        
        return code_artifacts
    
    def _generate_technical_code(self, requirement: Requirement, language: Language) -> List[GeneratedCode]:
        """Generate code for technical requirements"""
        code_artifacts = []
        
        # Generate utility functions
        function_name = self._extract_function_name(requirement.title)
        
        if language == Language.PYTHON:
            template = self.template_engine.get_template("python_function")
            if template:
                variables = {
                    'function_name': function_name,
                    'function_description': requirement.description,
                    'return_type': 'Any',
                    'function_body': self._generate_function_body(requirement)
                }
                
                code = template.render(variables)
                code_artifacts.append(GeneratedCode(
                    filename=f"{function_name}.py",
                    content=code,
                    language=language,
                    dependencies=[],
                    imports=['logging', 'typing']
                ))
        
        return code_artifacts
    
    def _generate_business_logic_code(self, requirement: Requirement, language: Language) -> List[GeneratedCode]:
        """Generate code for business logic requirements"""
        # Similar to functional but focused on business rules
        return self._generate_functional_code(requirement, language)
    
    def _extract_class_name(self, title: str) -> str:
        """Extract class name from requirement title"""
        # Remove common prefixes and clean up
        title = re.sub(r'^(system|user|application|feature)\s+', '', title.lower())
        title = re.sub(r'(must|should|shall|can|will)\s+', '', title)
        
        # Extract main noun/concept
        words = title.split()
        if words:
            # Take first meaningful word and capitalize
            return ''.join(word.capitalize() for word in words[:2] if word.isalpha())
        
        return "GeneratedClass"
    
    def _extract_function_name(self, title: str) -> str:
        """Extract function name from requirement title"""
        title = re.sub(r'^(system|user|application|feature)\s+', '', title.lower())
        title = re.sub(r'(must|should|shall|can|will)\s+', '', title)
        
        # Convert to snake_case
        words = [word for word in title.split() if word.isalpha()]
        return '_'.join(words[:3]) if words else "generated_function"
    
    def _camel_to_snake(self, camel_str: str) -> str:
        """Convert CamelCase to snake_case"""
        s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', camel_str)
        return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()
    
    def _generate_class_fields(self, requirement: Requirement) -> str:
        """Generate class fields from requirement"""
        # Extract field names from description and acceptance criteria
        fields = []
        
        # Look for field indicators in text
        text = f"{requirement.description} {' '.join(requirement.acceptance_criteria)}"
        field_patterns = [
            r'(\w+)\s+field',
            r'(\w+)\s+attribute',
            r'(\w+)\s+property',
            r'store\s+(\w+)',
            r'track\s+(\w+)',
            r'maintain\s+(\w+)'
        ]
        
        found_fields = set()
        for pattern in field_patterns:
            matches = re.findall(pattern, text.lower())
            found_fields.update(matches)
        
        # Generate field definitions
        for field in found_fields:
            if field.isalpha() and len(field) > 2:
                fields.append(f"    {field}: Optional[str] = None")
        
        if not fields:
            fields.append("    name: str")
            fields.append("    description: Optional[str] = None")
        
        return '\n'.join(fields)
    
    def _generate_class_methods(self, requirement: Requirement) -> str:
        """Generate class methods from requirement"""
        methods = []
        
        # Extract method names from acceptance criteria
        for criteria in requirement.acceptance_criteria:
            if 'when' in criteria.lower():
                # Extract action from "When they do X"
                match = re.search(r'when\s+.*?\s+(\w+)', criteria.lower())
                if match:
                    method_name = match.group(1)
                    if method_name not in ['they', 'user', 'system']:
                        methods.append(self._generate_method_template(method_name))
        
        if not methods:
            methods.append(self._generate_method_template('process'))
        
        return '\n'.join(methods)
    
    def _generate_method_template(self, method_name: str) -> str:
        """Generate method template"""
        return f'''
    def {method_name}(self, *args, **kwargs) -> Any:
        """Execute {method_name} operation"""
        self.logger.info(f"Executing {{method_name}}")
        # TODO: Implement {method_name} logic
        return True'''
    
    def _generate_dict_fields(self, requirement: Requirement) -> str:
        """Generate dictionary field mappings"""
        # This would match the class fields generated
        return '''            'name': self.name,
            'description': self.description'''
    
    def _generate_function_body(self, requirement: Requirement) -> str:
        """Generate function body from requirement"""
        body_lines = [
            '        # TODO: Implement requirement logic',
            f'        # {requirement.description}',
            '        result = None',
            '        ',
            '        return result'
        ]
        return '\n'.join(body_lines)
    
    def _generate_api_endpoint(self, requirement: Requirement, language: Language) -> Optional[GeneratedCode]:
        """Generate API endpoint code"""
        if language != Language.PYTHON:
            return None
        
        template = self.template_engine.get_template("python_api_endpoint")
        if not template:
            return None
        
        endpoint_name = self._extract_function_name(requirement.title)
        endpoint_path = f"/{endpoint_name.replace('_', '-')}"
        
        variables = {
            'endpoint_path': endpoint_path,
            'endpoint_name': endpoint_name,
            'endpoint_description': requirement.description,
            'http_method': 'POST' if 'create' in requirement.title.lower() or 'add' in requirement.title.lower() else 'GET'
        }
        
        code = template.render(variables)
        return GeneratedCode(
            filename=f"api_{endpoint_name}.py",
            content=code,
            language=language,
            dependencies=['flask'],
            imports=['flask', 'logging', 'datetime']
        )
    
    def _generate_database_model(self, requirement: Requirement, language: Language) -> Optional[GeneratedCode]:
        """Generate database model code"""
        if language != Language.PYTHON:
            return None
        
        template = self.template_engine.get_template("python_db_model")
        if not template:
            return None
        
        model_name = self._extract_class_name(requirement.title)
        table_name = self._camel_to_snake(model_name)
        
        variables = {
            'model_name': model_name,
            'table_name': table_name,
            'model_description': requirement.description,
            'model_fields': self._generate_db_fields(requirement),
            'dict_fields': self._generate_dict_fields(requirement)
        }
        
        code = template.render(variables)
        return GeneratedCode(
            filename=f"models/{self._camel_to_snake(model_name)}.py",
            content=code,
            language=language,
            dependencies=['sqlalchemy'],
            imports=['sqlalchemy', 'datetime', 'typing']
        )
    
    def _generate_db_fields(self, requirement: Requirement) -> str:
        """Generate database field definitions"""
        fields = [
            "    name = Column(String(255), nullable=False)",
            "    description = Column(Text, nullable=True)",
            "    is_active = Column(Boolean, default=True, nullable=False)"
        ]
        return '\n'.join(fields)
    
    def optimize_generated_code(self, code: GeneratedCode) -> GeneratedCode:
        """Optimize generated code using refactoring engine"""
        suggestions = self.refactoring_engine.suggest_improvements(
            code.content, code.language
        )
        
        optimized_content = code.content
        
        # Apply high-priority optimizations
        for suggestion in suggestions:
            if suggestion['severity'] == 'warning':
                if suggestion['type'] == 'logging':
                    optimized_content = self.refactoring_engine.apply_refactoring(
                        optimized_content, 'add_logging'
                    )
                elif suggestion['type'] == 'exception_handling':
                    optimized_content = self.refactoring_engine.apply_refactoring(
                        optimized_content, 'add_error_handling'
                    )
        
        return GeneratedCode(
            filename=code.filename,
            content=optimized_content,
            language=code.language,
            dependencies=code.dependencies,
            imports=code.imports,
            tests_needed=code.tests_needed,
            documentation_needed=code.documentation_needed,
            created_at=code.created_at
        )
    
    def generate_project_structure(self, requirements: List[Requirement], 
                                 language: Language = Language.PYTHON) -> Dict[str, Any]:
        """Generate complete project structure from requirements"""
        structure = {
            'files': {},
            'directories': set(),
            'dependencies': set(),
            'imports': set()
        }
        
        # Generate code for each requirement
        for requirement in requirements:
            code_artifacts = self.generate_from_requirement(requirement, language)
            
            for artifact in code_artifacts:
                # Add file to structure
                file_path = artifact.filename
                structure['files'][file_path] = artifact.content
                
                # Add directory
                directory = os.path.dirname(file_path)
                if directory:
                    structure['directories'].add(directory)
                
                # Collect dependencies and imports
                structure['dependencies'].update(artifact.dependencies)
                structure['imports'].update(artifact.imports)
        
        # Add standard project files
        if language == Language.PYTHON:
            structure['files']['__init__.py'] = ''
            structure['files']['requirements.txt'] = '\n'.join(sorted(structure['dependencies']))
            structure['files']['setup.py'] = self._generate_setup_py()
        
        return structure
    
    def _generate_setup_py(self) -> str:
        """Generate setup.py for Python project"""
        return '''from setuptools import setup, find_packages

setup(
    name="generated-project",
    version="1.0.0",
    description="Auto-generated project from requirements",
    packages=find_packages(),
    install_requires=[
        # Dependencies will be added here
    ],
    python_requires=">=3.8",
)'''
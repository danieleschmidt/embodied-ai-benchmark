# Contributing to Embodied-AI Benchmark++

We welcome contributions to make this benchmark even better! This document provides guidelines for contributing to the project.

## Development Setup

### Prerequisites
- Python 3.8+
- Git
- Virtual environment tool (venv, conda, etc.)

### Quick Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/embodied-ai-benchmark.git
cd embodied-ai-benchmark

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install
```

## Development Workflow

### 1. Create a Branch
```bash
git checkout -b feature/your-feature-name
# or
git checkout -b bugfix/issue-description
```

### 2. Make Changes
- Write clear, documented code
- Follow existing code style and patterns
- Add tests for new functionality
- Update documentation as needed

### 3. Test Your Changes
```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=embodied_ai_benchmark

# Run specific test categories
pytest -m unit              # Unit tests only
pytest -m integration       # Integration tests only
pytest -m "not slow"        # Skip slow tests
```

### 4. Code Quality Checks
```bash
# Format code
black src/ tests/
isort src/ tests/

# Lint code
flake8 src/ tests/

# Type check
mypy src/
```

### 5. Commit and Push
```bash
git add .
git commit -m "feat: add multi-agent coordination protocol"
git push origin feature/your-feature-name
```

### 6. Create Pull Request
- Use descriptive title and detailed description
- Reference related issues with "Fixes #123"
- Ensure all CI checks pass
- Request review from maintainers

## Contribution Types

### New Tasks
- Add tasks to `src/embodied_ai_benchmark/tasks/`
- Follow existing task interface patterns
- Include comprehensive documentation
- Add evaluation metrics
- Provide example usage

### New Metrics
- Implement metric classes in `src/embodied_ai_benchmark/metrics/`
- Include mathematical description
- Add visualization components
- Provide baseline comparisons

### Bug Fixes
- Reference issue number in commit message
- Add regression tests when possible
- Update documentation if behavior changes

### Documentation
- Update docstrings for new/changed APIs
- Add examples to README when relevant
- Update configuration documentation
- Contribute to user guides

## Code Style

### Python Guidelines
- Follow PEP 8 style guide
- Use type hints for all public APIs
- Write comprehensive docstrings (Google style)
- Maximum line length: 88 characters
- Use meaningful variable and function names

### Example Code Structure
```python
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from embodied_ai_benchmark.core import BaseTask


class YourTask(BaseTask):
    """Brief description of the task.
    
    This task implements [specific functionality]. It extends BaseTask
    to provide [key features].
    
    Args:
        param1: Description of parameter 1.
        param2: Description of parameter 2.
        
    Example:
        >>> task = YourTask(param1="value")
        >>> observation = task.reset()
        >>> action = your_agent.act(observation)
        >>> obs, reward, done, info = task.step(action)
    """
    
    def __init__(
        self, 
        param1: str,
        param2: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.param1 = param1
        self.param2 = param2 or 42
        
    def reset(self) -> np.ndarray:
        """Reset task to initial state.
        
        Returns:
            Initial observation as numpy array.
        """
        # Implementation here
        return self._get_observation()
        
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        """Execute action and return result.
        
        Args:
            action: Action to execute.
            
        Returns:
            Tuple of (observation, reward, done, info).
        """
        # Implementation here
        pass
```

### Commit Message Format
Use conventional commits format:

```
type(scope): description

[optional body]

[optional footer]
```

Types:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes
- `refactor`: Code refactoring
- `test`: Test changes
- `chore`: Build/tooling changes

Examples:
- `feat(tasks): add cooperative furniture assembly task`
- `fix(metrics): correct efficiency calculation for multi-agent scenarios`
- `docs(readme): update installation instructions`

## Testing Guidelines

### Test Structure
```
tests/
├── unit/                 # Fast, isolated tests
│   ├── test_tasks.py
│   ├── test_metrics.py
│   └── test_utils.py
├── integration/          # Tests with external dependencies
│   ├── test_simulator_integration.py
│   └── test_benchmark_suite.py
└── fixtures/             # Test data and utilities
    ├── sample_configs/
    └── mock_environments/
```

### Test Requirements
- All new code must have tests
- Aim for >90% code coverage
- Use meaningful test names: `test_furniture_assembly_success_rate_calculation`
- Mock external dependencies in unit tests
- Use fixtures for reusable test data

### Example Test
```python
import pytest
import numpy as np

from embodied_ai_benchmark.tasks import FurnitureAssemblyTask


class TestFurnitureAssemblyTask:
    """Test suite for FurnitureAssemblyTask."""
    
    @pytest.fixture
    def task(self):
        """Create a standard task instance for testing."""
        return FurnitureAssemblyTask(
            furniture="simple_table",
            difficulty="easy"
        )
    
    def test_reset_returns_valid_observation(self, task):
        """Test that reset returns properly shaped observation."""
        obs = task.reset()
        
        assert isinstance(obs, np.ndarray)
        assert obs.shape == task.observation_space.shape
        assert obs.dtype == np.float32
    
    @pytest.mark.parametrize("action_type", ["pick", "place", "rotate"])
    def test_valid_actions_accepted(self, task, action_type):
        """Test that all valid action types are accepted."""
        task.reset()
        action = task.action_space.sample()  # Generate valid action
        
        obs, reward, done, info = task.step(action)
        
        assert isinstance(reward, float)
        assert isinstance(done, bool)
        assert isinstance(info, dict)
```

## Documentation

### API Documentation
- Use Google-style docstrings
- Include type information
- Provide usage examples
- Document exceptions and edge cases

### User Documentation
- Update README.md for major features
- Add examples to demonstrate usage
- Include performance benchmarks when relevant
- Update troubleshooting guides

## Performance Considerations

### Benchmarking
- Profile new features for performance impact
- Include timing tests for critical paths
- Document computational complexity
- Consider memory usage implications

### Optimization Guidelines
- Vectorize operations when possible
- Use appropriate data structures
- Cache expensive computations
- Profile before optimizing

## Release Process

### Version Numbering
We follow Semantic Versioning (SemVer):
- Major: Breaking changes
- Minor: New features, backward compatible
- Patch: Bug fixes, backward compatible

### Changelog
Update CHANGELOG.md with:
- New features
- Bug fixes
- Breaking changes
- Deprecations

## Community Guidelines

### Code of Conduct
Be respectful, inclusive, and constructive in all interactions.

### Getting Help
- Check existing issues and documentation first
- Use GitHub Discussions for questions
- Tag maintainers only for urgent issues
- Provide minimal reproducible examples

### Review Process
- All changes require review from maintainers
- Reviews focus on correctness, style, and design
- Be responsive to feedback
- Reviews should be constructive and educational

## Advanced Contributions

### New Simulators
- Implement simulator interface in `src/embodied_ai_benchmark/simulators/`
- Follow existing patterns (Habitat, ManiSkill examples)
- Include comprehensive tests
- Document setup and requirements

### Evaluation Protocols
- Define standardized evaluation procedures
- Include statistical significance testing
- Provide baseline implementations
- Document methodology thoroughly

### Multi-Agent Extensions
- Follow communication protocol standards
- Include coordination primitives
- Test scalability thoroughly
- Document agent interaction patterns

## Questions?

If you have questions about contributing:
1. Check this document and existing issues
2. Start a GitHub Discussion
3. Contact maintainers directly for urgent matters

Thank you for contributing to Embodied-AI Benchmark++!
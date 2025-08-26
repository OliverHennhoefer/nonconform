# Contributing to nonconform

We welcome contributions to nonconform! This guide will help you get started.

## Types of Contributions

### Bug Reports
- Use the GitHub issue tracker
- Include minimal reproducible examples
- Specify your environment (Python version, OS, etc.)

### Feature Requests
- Describe the use case clearly
- Explain how it fits with the project's goals
- Consider proposing an implementation approach

### Code Contributions
- Bug fixes
- New conformalization strategies
- Performance improvements
- Documentation improvements

### Documentation
- Fix typos or unclear explanations
- Add examples or tutorials
- Improve API documentation

## Development Setup

### Prerequisites
- Python 3.12 or higher
- Git

### Setup Instructions

1. **Fork and clone the repository**
   ```bash
   git clone https://github.com/yourusername/nonconform.git
   cd nonconform
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install in development mode**
   ```bash
   pip install -e ".[dev,all]"
   ```

4. **Run tests to verify setup**
   ```bash
   pytest tests/
   ```

## Development Workflow

### Before Making Changes

1. **Create a new branch**
   ```bash
   git checkout -b feature/your-feature-name
   # or
   git checkout -b fix/issue-description
   ```

2. **Sync with upstream**
   ```bash
   git remote add upstream https://github.com/original/nonconform.git
   git fetch upstream
   git rebase upstream/main
   ```

### Making Changes

1. **Write tests first** (TDD approach recommended)
   ```bash
   # Add tests in tests/
   pytest tests/test_your_feature.py -v
   ```

2. **Implement your changes**
   - Follow the existing code style
   - Add docstrings to new functions/classes
   - Keep commits atomic and well-described

3. **Run the full test suite**
   ```bash
   pytest tests/
   ```

4. **Check code quality**
   ```bash
   # Format code
   black nonconform/ tests/

   # Check imports
   isort nonconform/ tests/

   # Lint code
   flake8 nonconform/ tests/

   # Type checking
   mypy nonconform/
   ```

### Documentation

1. **Update docstrings**
   - Use NumPy style docstrings
   - Include examples where helpful
   - Document all parameters and return values

2. **Update user documentation**
   - Add new features to appropriate guides
   - Update examples if needed
   - Test documentation builds locally

3. **Build documentation locally**
   ```bash
   cd docs/
   make html
   # Open docs/build/html/index.html in browser
   ```

### Submitting Changes

1. **Commit your changes**
   ```bash
   git add .
   git commit -m "feat: add weighted conformal p-values for covariate shift"
   ```

2. **Push to your fork**
   ```bash
   git push origin feature/your-feature-name
   ```

3. **Create a pull request**
   - Use a clear, descriptive title
   - Explain what changes you made and why
   - Reference any related issues
   - Include tests and documentation updates

## Code Style Guidelines

### Python Code Style
- Follow PEP 8
- Use Black for formatting
- Use isort for import sorting

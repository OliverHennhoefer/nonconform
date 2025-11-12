# Contributing

Thank you for your interest in contributing to this project! We welcome contributions from the community.

This project uses [uv](https://docs.astral.sh/uv/) for dependency management and [ruff](https://docs.astral.sh/ruff/) for formatting and linting.

## Getting Started

1. Fork the repository on GitHub
2. Clone your fork locally:
   ```bash
   git clone https://github.com/OliverHennhoefer/nonconform.git
   cd nonconform
   ```
3. Create a new branch for your changes:
   ```bash
   git checkout -b feature/your-feature-name
   ```

## Development Setup

1. Install [uv](https://docs.astral.sh/uv/) if you haven't already:
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh  # On Windows: irm https://astral.sh/uv/install.ps1 | iex
   ```

2. Sync the project and install dependencies:
   ```bash
   uv sync
   ```

3. The virtual environment will be automatically created and managed by uv

4. Install ``nonconform`` from the local repository.
   ```bash
    uv pip install -e .[all] --upgrade
   ```

## Making Changes

1. Make your changes in your feature branch
2. Add tests for any new functionality
3. Ensure all tests pass:
   ```bash
   uv run pytest
   ```
4. Format and check code style:
   ```bash
   uv run ruff format .
   uv run ruff check .
   ```

## Code Style

- Follow [PEP 8](https://pep8.org/) style guidelines (enforced by ruff)
- Use meaningful variable and function names
- Add docstrings to functions and classes
- Keep functions focused and concise
- Write clear commit messages
- Run `uv run ruff format .` before committing to ensure consistent formatting

## Testing

- Write tests for new features and bug fixes
- Ensure all existing tests pass before submitting with `uv run pytest`
- Aim for good test coverage of your changes
- Run `uv run pytest --cov` to check coverage (if configured)

## Submitting a Pull Request

1. Push your changes to your fork:
   ```bash
   git push origin feature/your-feature-name
   ```

2. Open a pull request on GitHub with:
   - A clear title and description
   - Reference to any related issues
   - Summary of changes made
   - Any breaking changes highlighted

3. Wait for review and address any feedback

## Reporting Issues

When reporting issues, please include:
- A clear description of the problem
- Steps to reproduce the issue
- Expected vs actual behavior
- Your environment (Python version, OS, etc.)
- Any relevant error messages or logs

## Questions?

Feel free to open an issue for questions or discussions about contributions.

# Contributing to svd-imputer

First off, thank you for considering contributing to `svd-imputer`! Contributions from the scientific community help make this tool more robust for everyone.

## Code of Conduct

By participating in this project, you agree to abide by the terms of the [MIT License](LICENSE) and maintain a respectful, collaborative environment.

## How to Contribute

### Reporting Bugs
- Check the [GitHub Issue Tracker](https://github.com/rhugman/svd_imputer/issues) to see if the bug has already been reported.
- If not, open a new issue. Include a **minimal reproducible example**, your operating system, and Python version.

### Suggesting Enhancements
- Open an issue to discuss your idea before implementing it. 
- We are particularly interested in features that improve the handling of environmental time series, such as new augmentation strategies or improved uncertainty metrics.

### Pull Request Process



1. **Fork** the repository and create your branch from `main`.
2. **Install** the development environment (see below).
3. **Write tests** for any new features or bug fixes.
4. **Submit** the PR with a clear description of what you’ve changed and why.

---

## Development Setup

`svd-imputer` requires Python 3.8 or higher. We use `setuptools` with a `pyproject.toml` configuration.

### 1. Environment Setup
Clone the repository and install the package in editable mode with development dependencies:

```bash
git clone [https://github.com/rhugman/svd_imputer.git](https://github.com/rhugman/svd_imputer.git)
cd svd_imputer
pip install -e ".[dev]"
```
# 2. Testing
We use pytest for our suite. Please ensure all tests pass before submitting a PR.

```
# Run all tests
pytest

# Run tests with coverage report
pytest --cov=svd_imputer```
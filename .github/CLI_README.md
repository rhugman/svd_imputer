# CI/CD Pipeline Documentation

This document describes the GitHub Actions CI/CD pipeline implemented for the SVD Imputer project.

## Overview

The CI/CD pipeline provides:
- ✅ **Automated Testing**: Fast feedback on every push/PR (Ubuntu × Python 3.9, 3.11, 3.12).
- 📦 **Release Management**: Comprehensive multi-platform testing and automated PyPI publishing on release tags.
- 📄 **Paper Generation**: Automated PDF generation for the project paper.
- 🛠 **Local Development Tools**: `Makefile` integration for code quality, security scanning, and documentation.

## Workflows

### 1. Main CI Workflow (`.github/workflows/ci.yml`)

**Triggers:** Push/PR to main/develop branches, workflow calls.

**Jobs:**
- **Test**: Runs `pytest` with coverage on Ubuntu-latest across Python 3.9, 3.11, and 3.12.

**Key Features:**
- Fast execution for rapid feedback.
- Coverage reporting with `pytest-cov`.
- Verifies package installation.

### 2. Release Workflow (`.github/workflows/release.yml`)

**Triggers:** Version tags (v*), manual dispatch.

**Jobs:**
- **Validation**: Checks version format and changelog presence.
- **Full Testing**: Runs the Main CI suite.
- **Package Building**: Builds source (sdist) and wheel distributions.
- **Installation Testing**: Validates installation on **Ubuntu, Windows, and macOS** across Python 3.8, 3.11, and 3.12.
- **GitHub Release**: Creates a GitHub release with auto-generated notes.
- **PyPI Publishing**: Publishes to PyPI (if not a dry run).

**Features:**
- Multi-OS, Multi-Python verification before publishing.
- Automated release notes generation.
- Artifact verification.

### 3. Draft PDF Workflow (`.github/workflows/draft-pdf.yml`)

**Triggers:** Push to any branch.

**Jobs:**
- **Paper Draft**: Builds a PDF version of the paper (`paper/paper.md`) using `openjournals/openjournals-draft-action`.
- **Artifacts**: Uploads the generated PDF as a workflow artifact.

## Configuration Files

### Code Quality & Project Configuration

**`pyproject.toml`** - Main configuration file:
- Project metadata and dependencies.
- Black formatting settings.
- Tool configurations (setuptools_scm, etc.).

**`setup.cfg`** - Additional tool configuration:
- Flake8 linting rules.

**`requirements-dev.txt`** - Development dependencies:
- Testing frameworks (pytest, coverage).
- Code quality tools (black, isort, flake8, mypy).
- Security tools (bandit, safety, pip-audit).
- Documentation tools (sphinx).

### Development Tools

**`Makefile`** - Common development tasks:
- **Testing**: `make test`, `make test-all`, `make test-fast`.
- **Quality**: `make lint` (runs black, isort, flake8, mypy, pydocstyle).
- **Formatting**: `make format` (runs black, isort).
- **Security**: `make security` (runs bandit, safety, pip-audit).
- **Docs**: `make docs` (builds Sphinx documentation).

## Usage Guide

### For Developers

1. **Initial Setup:**
   ```bash
   make setup-dev          # Complete development setup
   make pre-commit         # Run all pre-commit checks
   ```

2. **Daily Development:**
   ```bash
   make test-fast          # Quick test run
   make lint              # Code quality checks
   make format            # Auto-format code
   ```

3. **Before Committing:**
   ```bash
   make ci-local          # Run full CI suite locally
   ```

4. **Documentation:**
   ```bash
   make docs              # Build documentation
   make docs-serve        # Serve docs locally
   ```

### For CI/CD

**Automatic Triggers:**
- **Push to main/develop**: Runs Main CI (tests).
- **Version tags (v*)**: Triggers the full Release pipeline.

**Manual Triggers:**
- **Release Workflow**: Can be manually triggered to test the release process or publish with specific options.

### Release Process

1. **Automated (Recommended):**
   ```bash
   git tag v1.2.3
   git push origin v1.2.3  # Triggers full release pipeline
   ```

2. **Manual with Workflow Dispatch:**
   - Go to Actions → Release & Publish
   - Click "Run workflow"
   - Select version bump type and dry-run options.

## Security Features

### Secrets Management
Required secrets for release functionality:
- `PYPI_API_TOKEN`: PyPI publishing.

### Security Scanning
Security checks are configured for local execution via `make security`. This runs:
- **Bandit**: Common security issues in Python code.
- **Safety**: Checks installed dependencies for known vulnerabilities.
- **Pip-audit**: Audits Python environments for packages with known vulnerabilities.

## Contributing

When contributing to the CI/CD pipeline:

1. Test changes locally with `make ci-local`.
2. Update documentation if workflows change.
3. Ensure `Makefile` targets remain functional as they drive local development.

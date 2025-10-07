# CI/CD Pipeline Documentation

This document describes the comprehensive GitHub Actions CI/CD pipeline implemented for the SVD Imputer project.

## Overview

The CI/CD pipeline provides:
- ✅ **Multi-OS, Multi-Python Testing**: Ubuntu, Windows, macOS × Python 3.8-3.12
- 🔍 **Code Quality**: Black, isort, flake8, mypy, pydocstyle
- 🔒 **Security Scanning**: Bandit, Safety, pip-audit, Semgrep, secret scanning
- 📚 **Documentation**: Automated building and deployment to GitHub Pages
- 📦 **Release Management**: Automated PyPI publishing with testing
- ⚡ **Performance Monitoring**: Benchmark tracking and memory profiling
- 🧪 **Comprehensive Testing**: Unit, integration, and performance tests

## Workflows

### 1. Main CI Workflow (`.github/workflows/ci.yml`)

**Triggers:** Push/PR to main/develop branches, scheduled weekly runs, manual dispatch

**Jobs:**
- **Code Quality**: Black formatting, isort, flake8 linting, mypy type checking, bandit security
- **Test Matrix**: Tests across 3 OS × 5 Python versions (with smart exclusions)
- **Performance**: Benchmark tests with historical tracking
- **Documentation**: Build validation and artifact generation
- **Package Build**: Test package building and validation
- **Security**: Dependency vulnerability scanning

**Key Features:**
- Fail-fast disabled for comprehensive testing
- Coverage reporting with Codecov integration
- Parallel execution for efficiency
- Artifact preservation for debugging

### 2. Documentation Workflow (`.github/workflows/docs.yml`)

**Triggers:** Push/PR affecting docs, manual dispatch

**Jobs:**
- **Docs Build**: Sphinx documentation generation with multiple themes
- **GitHub Pages Deploy**: Automatic deployment to GitHub Pages (main branch only)
- **Quality Checks**: RST syntax, doc8 style checking, docstring coverage
- **Documentation Coverage**: API coverage analysis with badge generation

**Features:**
- Automatic API documentation generation with `sphinx-apidoc`
- Link checking for external references
- Multiple output formats (HTML, PDF ready)
- Documentation coverage metrics

### 3. Security Workflow (`.github/workflows/security.yml`)

**Triggers:** Push/PR to main/develop, weekly scheduled scans, manual dispatch

**Jobs:**
- **Static Analysis**: Bandit, Semgrep, code quality security checks
- **Dependency Scanning**: Safety, pip-audit, OWASP Dependency Check
- **Secret Scanning**: TruffleHog for exposed secrets
- **License Compliance**: License compatibility checking
- **Container Security**: Trivy scanning (when Docker support added)

**Features:**
- SARIF report generation for GitHub Security tab
- Automated security issue creation for scheduled scans
- Comprehensive security reporting with artifacts
- Integration with GitHub Advanced Security features

### 4. Release Workflow (`.github/workflows/release.yml`)

**Triggers:** Version tags (v*), manual dispatch with version bump options

**Jobs:**
- **Validation**: Prerequisites checking, version validation
- **Full Testing**: Complete CI suite execution before release
- **Package Building**: Source and wheel distribution creation
- **Installation Testing**: Multi-platform installation validation
- **GitHub Release**: Automated release creation with changelog
- **Test PyPI**: Publishing to Test PyPI for validation
- **PyPI Publishing**: Production release to PyPI
- **Post-Release**: Cleanup and notification tasks

**Features:**
- Semantic version validation
- Automated changelog generation
- Multi-stage release process with validation
- Rollback capabilities
- Post-release task tracking

## Configuration Files

### Code Quality Configuration

**`pyproject.toml`** - Main configuration file:
- Black formatting settings (127 char line length)
- isort import sorting configuration
- Coverage reporting settings
- Bandit security configuration
- MyPy type checking settings

**`setup.cfg`** - Additional tool configuration:
- Pytest test discovery and execution settings
- Flake8 linting rules and exclusions
- Coverage reporting options
- MyPy type checking overrides

**`.pre-commit-config.yaml`** - Pre-commit hooks:
- Automated code formatting on commit
- Linting checks before commits
- Security scanning integration
- Documentation validation

### Testing Configuration

**`tox.ini`** - Multi-environment testing:
- Python version matrix testing
- Isolated environment testing
- Specialized test environments (lint, docs, security)
- GitHub Actions integration mapping

**`requirements-dev.txt`** - Development dependencies:
- Testing frameworks (pytest, coverage)
- Code quality tools (black, isort, flake8, mypy)
- Security tools (bandit, safety, pip-audit)
- Documentation tools (sphinx, themes)
- Build and release tools

### Development Tools

**`Makefile`** - Common development tasks:
- Quick setup and installation commands
- Testing shortcuts (fast, slow, integration)
- Code quality checks and formatting
- Documentation building and serving
- Release preparation and publishing
- CI simulation for local testing

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
- **Push to main/develop**: Full CI suite + documentation deployment
- **Pull Requests**: Full testing and validation
- **Version tags**: Complete release process
- **Weekly**: Security scans and dependency updates

**Manual Triggers:**
- Workflow dispatch for any workflow
- Release workflow with version bump options
- Security scans on-demand

### Release Process

1. **Automated (Recommended):**
   ```bash
   git tag v1.2.3
   git push origin v1.2.3  # Triggers full release pipeline
   ```

2. **Manual with Workflow Dispatch:**
   - Go to Actions → Release & Publish
   - Click "Run workflow"
   - Select version bump type (patch/minor/major)
   - Optionally enable dry-run mode

## Security Features

### Secrets Management
Required secrets for full functionality:
- `PYPI_API_TOKEN`: PyPI publishing (production)
- `TEST_PYPI_API_TOKEN`: Test PyPI publishing
- `CODECOV_TOKEN`: Coverage reporting (optional but recommended)
- `NVD_API_KEY`: Enhanced dependency vulnerability scanning

### Security Scanning
- **Daily**: Dependency vulnerability scanning
- **Weekly**: Comprehensive security suite
- **PR/Push**: Basic security checks
- **Continuous**: Secret scanning on all commits

### Compliance
- License compatibility checking
- SBOM (Software Bill of Materials) generation
- SARIF security report integration
- Automated vulnerability reporting

## Performance Monitoring

### Benchmarks
- **Execution Time**: Performance regression detection
- **Memory Usage**: Memory leak and efficiency monitoring  
- **Scaling**: Performance across different data sizes
- **Comparison**: Method performance comparisons

### Monitoring
- Historical benchmark tracking
- Performance regression alerts
- Memory profiling reports
- Scalability analysis

## Troubleshooting

### Common Issues

1. **Test Failures:**
   - Check specific OS/Python version combinations
   - Review test artifacts for detailed logs
   - Run `make test-all` locally to reproduce

2. **Security Alerts:**
   - Review security scan artifacts
   - Update dependencies: `pip install -U -r requirements-dev.txt`
   - Check bandit configuration for false positives

3. **Documentation Failures:**
   - Validate RST syntax: `make docs`
   - Check for broken links in artifacts
   - Ensure all docstrings follow NumPy convention

4. **Release Issues:**
   - Verify version tag format (v1.2.3)
   - Check that all CI tests pass
   - Ensure secrets are properly configured

### Performance Optimization

The pipeline is optimized for:
- **Parallel execution** where possible
- **Smart caching** of dependencies
- **Selective testing** with exclusion matrices
- **Artifact reuse** between jobs
- **Fail-fast disabled** for comprehensive feedback

### Monitoring and Alerts

- GitHub Actions status badges in README
- Codecov integration for coverage tracking
- Security alerts through GitHub Advanced Security
- Performance regression detection
- Automated issue creation for failures

## Future Enhancements

Planned improvements:
- Container-based testing with Docker
- Integration with external code quality services
- Advanced performance profiling
- Multi-language support preparation
- Enhanced release automation
- Dependency update automation (Dependabot integration)

## Contributing

When contributing to the CI/CD pipeline:

1. Test changes locally with `make ci-local`
2. Update documentation for new features
3. Follow the existing workflow patterns
4. Add appropriate error handling
5. Include monitoring and alerting where appropriate

For questions or issues with the CI/CD pipeline, please create an issue with the `ci/cd` label.
# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Comprehensive GitHub Actions CI/CD pipeline
- Multi-OS and multi-Python version testing
- Code quality tools (Black, isort, flake8, mypy)
- Security scanning with Bandit, Safety, and pip-audit
- Performance benchmarking infrastructure
- Documentation building and deployment
- Automated release workflow
- Pre-commit hooks configuration
- Tox configuration for local testing
- Development Makefile for common tasks

### Changed
- Updated test suite to use pytest framework
- Improved code organization and structure
- Enhanced error handling and logging

### Fixed
- Various bug fixes and improvements

## [1.0.0] - TBD

### Added
- Initial release of SVD Imputer package
- SVD-based time series imputation algorithm
- Monte Carlo uncertainty estimation
- Automatic rank optimization with rank='auto'
- Data-centric API design: Imputer(data=df)
- Comprehensive logging system
- Support for Python 3.8-3.12
- Cross-platform compatibility (Windows, macOS, Linux)

### Features
- **Core Imputation**: Efficient SVD-based missing value imputation for time series
- **Uncertainty Quantification**: Monte Carlo method for uncertainty estimation
- **Rank Optimization**: Automatic rank selection based on variance threshold
- **Robust Validation**: Input data validation and error handling
- **Flexible API**: Easy-to-use interface with sensible defaults
- **Performance**: Optimized for both small and large datasets

### Dependencies
- numpy >= 1.20.0
- pandas >= 1.3.0
- scikit-learn >= 1.0.0

### Documentation
- Complete API documentation
- Usage examples and tutorials
- Installation guide
- Contributing guidelines

---

## Release Process

### Version Numbering
This project follows [Semantic Versioning](https://semver.org/):
- **MAJOR** version for incompatible API changes
- **MINOR** version for backwards-compatible functionality additions  
- **PATCH** version for backwards-compatible bug fixes

### Release Types
- **Alpha** releases (X.Y.ZaN): Early testing versions
- **Beta** releases (X.Y.ZbN): Feature-complete pre-releases
- **Release Candidate** (X.Y.ZrcN): Final testing before stable release
- **Stable** releases (X.Y.Z): Production-ready versions

### Automated Releases
Releases are automatically created when:
1. A version tag (e.g., `v1.0.0`) is pushed to the main branch
2. All CI tests pass
3. Security scans complete successfully
4. Documentation builds without errors

The release process includes:
- Building and testing distribution packages
- Publishing to Test PyPI for validation
- Creating GitHub release with changelog
- Publishing to PyPI for stable releases
- Updating documentation

### Manual Release Process
For manual releases:
1. Update version in `setup.py` or use `setuptools_scm`
2. Update this CHANGELOG.md with new version details
3. Commit changes: `git commit -m "Release vX.Y.Z"`
4. Create and push tag: `git tag vX.Y.Z && git push origin vX.Y.Z`
5. GitHub Actions will handle the rest automatically
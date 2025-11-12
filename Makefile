# Makefile for SVD Imputer development

.PHONY: help install dev-install test test-all lint format security docs clean build publish-test publish

# Variables
PYTHON ?= python
PIP ?= pip
PYTEST ?= pytest

help: ## Show this help message
	@echo "Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install: ## Install package in development mode
	$(PIP) install -e .

dev-install: ## Install package with development dependencies
	$(PIP) install -e .
	$(PIP) install -r requirements-dev.txt
	pre-commit install

test: ## Run basic tests
	$(PYTEST) tests/ -v

test-all: ## Run all tests including slow ones
	$(PYTEST) tests/ -v --cov=svd_imputer --cov-report=html --cov-report=term-missing

test-fast: ## Run fast tests only (exclude slow and integration tests)
	$(PYTEST) tests/ -v -m "not slow and not integration"

test-integration: ## Run integration tests only
	$(PYTEST) tests/ -v -m integration

benchmark: ## Run performance benchmarks
	$(PYTEST) benchmarks/ --benchmark-only

lint: ## Run linting checks
	black --check svd_imputer/ tests/
	isort --check-only svd_imputer/ tests/
	flake8 svd_imputer/ tests/
	mypy svd_imputer/ --ignore-missing-imports
	pydocstyle svd_imputer/ --convention=numpy

format: ## Format code with black and isort
	black svd_imputer/ tests/ benchmarks/
	isort svd_imputer/ tests/ benchmarks/

security: ## Run security checks
	bandit -r svd_imputer/
	safety check
	pip-audit

docs: ## Build documentation
	sphinx-build -b html docs docs/_build/html

docs-serve: ## Serve documentation locally
	cd docs/_build/html && $(PYTHON) -m http.server 8000

docs-clean: ## Clean documentation build
	rm -rf docs/_build/

clean: ## Clean build artifacts
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .coverage
	rm -rf htmlcov/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	find . -name "*.pyc" -delete
	find . -name "__pycache__" -delete

build: ## Build distribution packages
	$(PYTHON) -m build

check-build: ## Check built packages
	twine check dist/*

publish-test: build ## Publish to Test PyPI
	twine upload --repository testpypi dist/*

publish: build ## Publish to PyPI (use with caution!)
	twine upload dist/*

version: ## Show current version
	$(PYTHON) -c "import svd_imputer; print(svd_imputer.__version__)"

pre-commit: ## Run pre-commit hooks on all files
	pre-commit run --all-files

setup-dev: dev-install ## Complete development setup
	@echo "Development setup complete!"
	@echo "Next steps:"
	@echo "  - Run 'make test' to run tests"
	@echo "  - Run 'make lint' to check code quality"
	@echo "  - Run 'make docs' to build documentation"

ci-local: ## Run CI checks locally
	make lint
	make test-all
	make security
	make docs
	@echo "✅ All CI checks passed locally!"

# Docker targets (if you add Docker support)
docker-build: ## Build Docker image
	docker build -t svd-imputer:latest .

docker-test: ## Test in Docker container
	docker run --rm svd-imputer:latest make test
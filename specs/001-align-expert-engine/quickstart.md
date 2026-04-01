# Quickstart: Align Guidance Engine with Expert Engine Architecture

**Date**: 2026-03-27

## Prerequisites

- Python 3.12+
- Poetry installed
- Docker (for build verification)

## Setup

```bash
# Install all dependencies (including dev)
poetry install

# Verify linting passes
poetry run flake8

# Run tests with coverage
poetry run pytest --cov --cov-report=term-missing --cov-fail-under=90
```

## Verify Changes

### 1. Stream-based execution

After modifying `ai_adapter.py`, verify the graph uses stream mode:

```bash
# Run tests that validate stream execution
poetry run pytest tests/test_ai_adapter.py -k "stream" -v
```

### 2. Rich source metadata

Verify source entries contain uri, title, type, and score:

```bash
poetry run pytest tests/test_ai_adapter.py -k "source" -v
```

### 3. Legacy cleanup

Verify removed dependencies are not importable:

```bash
# These should fail (not installed)
poetry run python -c "import faiss" 2>&1 | grep -q "ModuleNotFoundError" && echo "PASS: faiss removed"
poetry run python -c "import git" 2>&1 | grep -q "ModuleNotFoundError" && echo "PASS: gitpython removed"
```

### 4. Docker build

```bash
# Build should succeed without Hugo stage
docker build -f Dockerfile --target runtime -t guidance-engine:test .
```

### 5. Full test suite

```bash
# Must pass with 90%+ coverage
poetry run pytest --cov --cov-report=term-missing --cov-fail-under=90
```

## Validation Checklist

- [ ] `poetry run flake8` passes with no errors
- [ ] `poetry run pytest --cov --cov-fail-under=90` passes
- [ ] `docker build` succeeds (runtime stage)
- [ ] CLAUDE.md exists and covers: overview, technologies, patterns, style
- [ ] `.github/workflows/ci.yml` exists
- [ ] No faiss-cpu, gitpython, beautifulsoup4, tiktoken in pyproject.toml
- [ ] No Hugo/Stage 3 in Dockerfile

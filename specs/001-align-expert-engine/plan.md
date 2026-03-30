# Implementation Plan: Align Guidance Engine with Expert Engine Architecture

**Branch**: `001-align-expert-engine` | **Date**: 2026-03-27 | **Spec**: [spec.md](./spec.md)
**Input**: Feature specification from `/specs/001-align-expert-engine/spec.md`

## Summary

Upgrade the guidance engine to match the expert engine architecture: switch from `graph.invoke()` to stream-based execution, enrich source metadata (uri, title, type, score), add pytest suite with 90%+ coverage, add GitHub Actions CI pipeline, create CLAUDE.md, and remove legacy dependencies (faiss-cpu, gitpython, beautifulsoup4, tiktoken) and the Hugo Docker stage.

## Technical Context

**Language/Version**: Python 3.12+
**Primary Dependencies**: alkemio-virtual-contributor-engine v0.8.0, aio-pika 9.5.7, LangChain 1.1.0+, Pydantic
**Storage**: ChromaDB (via base library)
**Testing**: pytest with pytest-cov, pytest-asyncio
**Target Platform**: Linux containers (Docker multi-arch: x86_64, arm64), Kubernetes
**Project Type**: Async microservice (RabbitMQ message consumer)
**Performance Goals**: Match current response times; stream-based execution adds per-step logging overhead but no user-facing latency change
**Constraints**: Must maintain backward-compatible response format for consuming services; base library v0.8.0 API surface
**Scale/Scope**: 2 source files (ai_adapter.py, main.py), ~130 lines of application code

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

| # | Principle | Status | Notes |
|---|-----------|--------|-------|
| I | Knowledge-Grounded Responses | PASS | No changes to knowledge grounding; retrieve function still queries only ChromaDB collections |
| II | Async Message-Driven Architecture | PASS | No HTTP endpoints introduced; all changes stay within async RabbitMQ handler |
| III | Source Attribution & Scoring | PASS | This feature adds rich metadata (uri, title, type, score) — directly fulfills this principle |
| IV | Observability | PASS | Stream-based execution adds per-step logging and duration tracking |
| V | Security & Prompt Integrity | PASS | No changes to prompt handling; no hardcoded secrets introduced |
| VI | Test Coverage | PASS | Feature adds pytest suite targeting 90%+ coverage |

No violations. All changes align with or directly advance constitutional principles.

## Project Structure

### Documentation (this feature)

```text
specs/001-align-expert-engine/
├── plan.md              # This file
├── research.md          # Phase 0 output
├── data-model.md        # Phase 1 output
├── quickstart.md        # Phase 1 output
└── tasks.md             # Phase 2 output (/speckit.tasks command)
```

### Source Code (repository root)

```text
.
├── main.py              # Entry point — register handler, start engine
├── ai_adapter.py        # Core logic — retrieve, invoke (stream-based), source building
├── pyproject.toml       # Poetry dependencies
├── poetry.lock          # Locked dependencies
├── Dockerfile           # Production container (2 stages: builder + runtime)
├── Dockerfile.local     # Local dev container
├── .flake8              # Linting config
├── .env.default         # Environment variable documentation
├── CLAUDE.md            # Developer guide (NEW)
├── tests/
│   ├── conftest.py      # Shared fixtures (NEW)
│   ├── test_ai_adapter.py  # Tests for retrieve + invoke (NEW)
│   └── test_main.py     # Tests for main entry point (NEW)
└── .github/
    └── workflows/
        ├── ci.yml       # Lint + test + Docker build (NEW)
        └── build-deploy-k8s-*.yml  # Existing deploy workflows
```

**Structure Decision**: Flat project structure (no src/ directory) — consistent with the existing codebase and the expert engine. Tests in a top-level `tests/` directory.

## Complexity Tracking

No constitution violations to justify.

## Post-Design Constitution Re-Check

| # | Principle | Status | Notes |
|---|-----------|--------|-------|
| I | Knowledge-Grounded Responses | PASS | retrieve function unchanged in knowledge scope |
| II | Async Message-Driven Architecture | PASS | invoke remains async, no sync endpoints |
| III | Source Attribution & Scoring | PASS | Rich metadata added per design |
| IV | Observability | PASS | Per-step logging + duration tracking added |
| V | Security & Prompt Integrity | PASS | No prompt changes; env vars for config |
| VI | Test Coverage | PASS | 90%+ target with pytest-cov |

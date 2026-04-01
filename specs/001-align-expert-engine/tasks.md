# Tasks: Align Guidance Engine with Expert Engine Architecture

**Input**: Design documents from `/specs/001-align-expert-engine/`
**Prerequisites**: plan.md (required), spec.md (required for user stories), research.md, data-model.md

**Tests**: Explicitly requested as User Story 3 (P1) — test tasks are included.

**Organization**: Tasks are grouped by user story to enable independent implementation and testing of each story.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (e.g., US1, US2, US3)
- Include exact file paths in descriptions

## Phase 1: Setup

**Purpose**: Add test infrastructure and dev dependencies

- [x] T001 Add pytest, pytest-asyncio, pytest-cov to dev dependencies in pyproject.toml
- [x] T002 Create tests/ directory with empty tests/__init__.py
- [x] T003 [P] Create shared test fixtures in tests/conftest.py (mock query_documents, mock PromptGraph, mock mistral_small, sample Input/Response factories)

**Checkpoint**: Test infrastructure ready — user story implementation can begin

---

## Phase 2: User Story 1 — Stream-Based Graph Execution (Priority: P1)

**Goal**: Replace `graph.invoke()` with `graph.stream(input_state, stream_mode="updates")` and add per-step logging with duration tracking.

**Independent Test**: Run a query and verify per-step log entries and total duration in logs.

### Implementation for User Story 1

- [x] T004 [US1] Refactor invoke() in ai_adapter.py to use graph.stream() with stream_mode="updates", accumulating results per step (see research.md R1 for pattern)
- [x] T005 [US1] Add per-step logging (node name on each step completion) and total execution duration logging in ai_adapter.py
- [x] T006 [US1] Add try/except per collection in retrieve() in ai_adapter.py — log warning and skip on failure per FR-006a
- [x] T007 [US1] Update input_state construction in invoke() in ai_adapter.py to include current_question, bok_id fields matching expert engine pattern

**Checkpoint**: Stream-based execution working with per-step logging. Existing response format preserved.

---

## Phase 3: User Story 2 — Rich Source Metadata (Priority: P2)

**Goal**: Enrich source entries with uri, title, type, and score fields. Apply metadata defaults and deduplication.

**Independent Test**: Send a query and verify each source in the response contains all 4 fields.

**Note**: Depends on US1 (stream-based result accumulation provides `source_scores` and `knowledge_docs` in `result` dict).

### Implementation for User Story 2

- [x] T008 [US2] Update source construction in invoke() in ai_adapter.py to build rich metadata entries using dict(doc) | computed fields pattern (see research.md R2)
- [x] T009 [US2] Add title formatting with regex humanization of type field in ai_adapter.py (format: "[Type] Title")
- [x] T010 [US2] Add metadata defaults in source construction in ai_adapter.py — empty string for missing title, "unknown" for missing type per FR-006b
- [x] T011 [US2] Update source deduplication in ai_adapter.py to deduplicate by source field (raw URI), last occurrence wins
- [x] T012 [US2] Update Response construction in invoke() in ai_adapter.py to use Response(**json_result) pattern matching expert engine

**Checkpoint**: Responses include rich source metadata. Format matches expert engine.

---

## Phase 4: User Story 3 — Test Suite with 90% Coverage (Priority: P1)

**Goal**: Create pytest test suite covering ai_adapter.py and main.py with at least 90% code coverage.

**Independent Test**: Run `pytest --cov --cov-fail-under=90` and verify all tests pass with coverage met.

**Note**: Depends on US1 and US2 (tests validate the final code state).

### Tests for User Story 3

- [x] T013 [P] [US3] Write tests for retrieve() in tests/test_ai_adapter.py — test successful aggregation across 3 collections, test partial collection failure (one collection raises exception), test all collections empty, test missing metadata fields in results
- [x] T014 [P] [US3] Write tests for invoke() happy path in tests/test_ai_adapter.py — mock PromptGraph and graph.stream, verify Response fields (result, sources with rich metadata, language metadata)
- [x] T015 [P] [US3] Write tests for invoke() error handling in tests/test_ai_adapter.py — test missing prompt_graph raises exception and returns fallback, test graph.stream exception returns fallback response
- [x] T016 [P] [US3] Write tests for invoke() source edge cases in tests/test_ai_adapter.py — test all scores zero yields empty sources, test duplicate source URIs are deduplicated, test missing title/type uses defaults
- [x] T017 [US3] Write tests for main.py in tests/test_main.py — test engine creation, handler registration, and engine.start() is called
- [x] T018 [US3] Run full test suite with coverage and verify 90%+ threshold: `pytest --cov=ai_adapter --cov=main --cov-report=term-missing --cov-fail-under=90`

**Checkpoint**: All tests pass with 90%+ coverage. Ready for CI integration.

---

## Phase 5: User Story 5 — Legacy Cleanup (Priority: P3)

**Goal**: Remove unused dependencies and Docker stages.

**Independent Test**: Docker build succeeds; no import errors at runtime.

**Note**: Independent of US1-US4. Can be done in parallel with any phase after Setup.

### Implementation for User Story 5

- [x] T019 [P] [US5] Remove faiss-cpu, gitpython, beautifulsoup4, tiktoken from pyproject.toml
- [x] T020 [P] [US5] Remove Stage 3 (runtime-full with Hugo) from Dockerfile — keep only builder and runtime stages
- [x] T021 [US5] Run `poetry lock` to regenerate poetry.lock without removed dependencies
- [x] T022 [US5] Verify Docker build succeeds: `docker build -f Dockerfile --target runtime .`

**Checkpoint**: Clean dependency manifest and Dockerfile. No legacy bloat.

---

## Phase 6: User Story 6 — CLAUDE.md Development Guide (Priority: P3)

**Goal**: Create CLAUDE.md with project overview, technologies, patterns, and style guide.

**Independent Test**: File exists and covers all required sections.

**Note**: Independent of all other stories. Can be done in parallel.

### Implementation for User Story 6

- [x] T023 [US6] Create CLAUDE.md at repository root with sections: Project Overview, Active Technologies (with versions), Project Structure, Commands, Code Style, Key Patterns (matching expert engine CLAUDE.md format)

**Checkpoint**: CLAUDE.md provides complete developer onboarding guide.

---

## Phase 7: User Story 4 — CI Build Pipeline (Priority: P2)

**Goal**: GitHub Actions workflow that runs linting, tests, and Docker build on PRs to develop.

**Independent Test**: Open a PR to develop and verify all 3 CI jobs run.

**Note**: Depends on US3 (test suite must exist for the test job to pass).

### Implementation for User Story 4

- [x] T024 [US4] Create .github/workflows/ci.yml with 3 parallel jobs: lint (flake8), test (pytest --cov --cov-fail-under=90), docker-build (docker build --target runtime)
- [x] T025 [US4] Configure CI trigger on pull_request to develop branch in .github/workflows/ci.yml
- [x] T026 [US4] Verify CI workflow syntax: `act -l` or push branch and open draft PR

**Checkpoint**: CI pipeline gates PRs with linting, test coverage, and Docker build checks.

---

## Phase 8: Polish & Cross-Cutting Concerns

**Purpose**: Final validation across all stories

- [x] T027 Run flake8 linting across all modified files and fix any violations
- [x] T028 Run full test suite with coverage to confirm 90%+ maintained after all changes
- [x] T029 Run quickstart.md validation checklist in specs/001-align-expert-engine/quickstart.md
- [x] T030 Verify Docker build succeeds end-to-end: `docker build -f Dockerfile --target runtime .`

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies — can start immediately
- **US1 (Phase 2)**: Depends on Setup — core code change
- **US2 (Phase 3)**: Depends on US1 — extends stream-based result handling
- **US3 (Phase 4)**: Depends on US1 + US2 — tests validate final code state
- **US5 (Phase 5)**: Independent — can run in parallel with any phase after Setup
- **US6 (Phase 6)**: Independent — can run in parallel with any phase
- **US4 (Phase 7)**: Depends on US3 — CI needs test suite to exist
- **Polish (Phase 8)**: Depends on all stories being complete

### User Story Dependencies

- **US1 (P1)**: Can start after Setup — no dependencies on other stories
- **US2 (P2)**: Depends on US1 — builds on stream-based result dict
- **US3 (P1)**: Depends on US1 + US2 — tests must cover final implementation
- **US4 (P2)**: Depends on US3 — CI test job requires test suite
- **US5 (P3)**: Independent — different files (pyproject.toml, Dockerfile)
- **US6 (P3)**: Independent — new file (CLAUDE.md)

### Parallel Opportunities

- T002 and T003 can run in parallel (different files)
- T013, T014, T015, T016 can all run in parallel (different test functions in same file, but no dependencies)
- T019 and T020 can run in parallel (different files)
- US5 and US6 can run in parallel with US1/US2 (no file conflicts)

---

## Parallel Example: User Story 3 (Tests)

```bash
# Launch all test-writing tasks together:
Task: "Write tests for retrieve() in tests/test_ai_adapter.py"
Task: "Write tests for invoke() happy path in tests/test_ai_adapter.py"
Task: "Write tests for invoke() error handling in tests/test_ai_adapter.py"
Task: "Write tests for invoke() source edge cases in tests/test_ai_adapter.py"
```

---

## Implementation Strategy

### MVP First (US1 + US2)

1. Complete Phase 1: Setup
2. Complete Phase 2: US1 — Stream-based execution
3. Complete Phase 3: US2 — Rich source metadata
4. **STOP and VALIDATE**: Engine produces correct responses with rich sources and per-step logging

### Full Delivery

1. Setup → US1 → US2 → US3 (tests) → US4 (CI) → Polish
2. In parallel: US5 (cleanup) + US6 (CLAUDE.md) anytime after Setup
3. Each story adds value without breaking previous stories

---

## Notes

- [P] tasks = different files, no dependencies
- [Story] label maps task to specific user story for traceability
- US1 and US2 both modify ai_adapter.py — must be sequential
- US3 tests the combined result of US1 + US2
- US5 and US6 are fully independent and can be interleaved anywhere
- Commit after each task or logical group
- Stop at any checkpoint to validate story independently

# Feature Specification: Align Guidance Engine with Expert Engine Architecture

**Feature Branch**: `001-align-expert-engine`
**Created**: 2026-03-27
**Status**: Draft
**Input**: User description: "Upgrade the virtual-contributor-guidance-engine to align with the expert engine architecture"

## Clarifications

### Session 2026-03-27

- Q: When a knowledge base collection is unavailable during retrieval, how should the system behave? → A: Skip the failed collection silently, return results from remaining collections, log a warning.
- Q: When source document metadata is missing fields (e.g., no title or type), how should the source entry be constructed? → A: Use sensible defaults (empty string for title, "unknown" for type).

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Stream-Based Graph Execution (Priority: P1)

As the Alkemio platform, when a user sends a query to the guidance engine, the system processes it through a stream-based prompt graph execution pipeline (matching the expert engine pattern) so that each graph step is logged individually and results are accumulated progressively, improving observability and debuggability.

**Why this priority**: Stream-based execution is the core architectural alignment. It changes how every query is processed and is a prerequisite for proper step-level logging and observability. All other stories build on this foundation.

**Independent Test**: Can be fully tested by sending a query message via RabbitMQ and verifying that the response is generated through stream-based graph execution with per-step logging.

**Acceptance Scenarios**:

1. **Given** a valid query message with a prompt graph, **When** the engine processes it, **Then** the graph executes using stream mode with step-by-step updates and each step is logged with its node name.
2. **Given** a valid query message, **When** the graph completes, **Then** the total execution duration is logged.
3. **Given** a query that triggers an error during graph execution, **When** the error occurs, **Then** the system returns a graceful fallback response and logs the exception.

---

### User Story 2 - Rich Source Metadata (Priority: P2)

As a consuming service receiving guidance engine responses, the source attribution for each answer includes rich metadata (URI, title, document type, and relevance score) so that downstream systems can filter, rank, and present sources meaningfully to users.

**Why this priority**: Source metadata enrichment directly impacts answer quality presentation for end users. It aligns the response format with the expert engine, enabling consistent handling by consuming services.

**Independent Test**: Can be tested by sending a query and verifying that each source in the response contains uri, title, type, and score fields.

**Acceptance Scenarios**:

1. **Given** a query that matches documents in the knowledge base, **When** a response is generated, **Then** each source includes uri, title, type, and a relevance score (0-10).
2. **Given** a query that matches multiple documents from the same source URI, **When** sources are returned, **Then** duplicates are removed by source URI, keeping the entry with the richest metadata.
3. **Given** a query with no matching documents, **When** a response is generated, **Then** the sources list is empty.

---

### User Story 3 - Test Suite with 90% Coverage (Priority: P1)

As the development team, a pytest test suite covers the guidance engine's core logic (retrieve function, invoke function, error handling, main entry point) with at least 90% code coverage so that regressions are caught automatically.

**Why this priority**: Tests are a prerequisite for the CI pipeline and for confident refactoring. They validate that the stream-based execution and rich source metadata work correctly.

**Independent Test**: Can be tested by running the test suite and verifying that coverage is at or above 90% and all tests pass.

**Acceptance Scenarios**:

1. **Given** the test suite, **When** tests run, **Then** all tests pass.
2. **Given** the test suite, **When** coverage is measured, **Then** overall code coverage is at least 90%.
3. **Given** the retrieve function, **When** tested with mocked knowledge base responses, **Then** it correctly aggregates documents from all 3 collections and handles empty or missing results.
4. **Given** the invoke function, **When** tested with a valid prompt graph, **Then** it produces a response with knowledge answer, rich sources, and language metadata.
5. **Given** the invoke function, **When** an exception occurs during graph execution, **Then** it returns the fallback error response.

---

### User Story 4 - CI Build Pipeline (Priority: P2)

As the development team, a CI pipeline runs automatically on pull requests to the develop branch, executing linting, tests, and Docker build verification so that broken code is caught before merge.

**Why this priority**: CI is essential infrastructure that gates code quality. It enforces the test coverage and linting standards defined in the constitution.

**Independent Test**: Can be tested by opening a PR to develop and verifying that the pipeline runs linting, tests with coverage, and Docker build steps.

**Acceptance Scenarios**:

1. **Given** a pull request targeting develop, **When** CI is triggered, **Then** linting runs and the pipeline fails if linting errors are found.
2. **Given** a pull request targeting develop, **When** CI is triggered, **Then** tests run with coverage reporting and the pipeline fails if coverage drops below 90%.
3. **Given** a pull request targeting develop, **When** CI is triggered, **Then** the Docker image builds successfully.

---

### User Story 5 - Legacy Cleanup (Priority: P3)

As a maintainer of the guidance engine, unused legacy dependencies and Docker build stages are removed so that the project has a clean, minimal footprint matching its actual runtime requirements.

**Why this priority**: Cleanup reduces build times, image size, and confusion for new contributors. It does not affect runtime behavior but is important for long-term maintainability.

**Independent Test**: Can be tested by verifying the Docker image builds successfully without legacy dependencies and that no import or runtime errors occur.

**Acceptance Scenarios**:

1. **Given** the updated dependency manifest, **When** dependencies are installed, **Then** faiss-cpu, gitpython, beautifulsoup4, and tiktoken are no longer present.
2. **Given** the updated Dockerfile, **When** the image is built, **Then** there is no Hugo installation stage and no Git-dependent ingest stage.
3. **Given** the cleaned-up project, **When** the engine starts and processes a query, **Then** it operates correctly without the removed dependencies.

---

### User Story 6 - CLAUDE.md Development Guide (Priority: P3)

As a developer (human or AI) working on the guidance engine, a CLAUDE.md file provides a concise project overview, active technologies, key code patterns, and style guidelines so that contributors can orient themselves quickly.

**Why this priority**: Documentation supports all future development but does not block any runtime functionality.

**Independent Test**: Can be tested by verifying the file exists, covers the required sections (overview, technologies, patterns, style), and accurately reflects the current project state.

**Acceptance Scenarios**:

1. **Given** a new contributor opens the project, **When** they read CLAUDE.md, **Then** they find a project overview, list of active technologies with versions, key code patterns (PromptGraph flow, retrieve node, RabbitMQ handling), and code style rules (linting conventions, logging patterns).

---

### Edge Cases

- When one of the 3 knowledge base collections is unavailable or returns an error, the system skips the failed collection, logs a warning, and returns results from the remaining collections.
- What happens when the prompt graph definition in the input payload is malformed? The existing exception handler catches it and returns the fallback error response.
- When source metadata is missing fields (e.g., no title or type), the system uses sensible defaults: empty string for title, "unknown" for type.
- When all source relevance scores are 0 (no useful sources), the sources list is empty.
- When the graph stream produces no steps (empty graph), the system returns the fallback error response.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: System MUST execute the prompt graph using stream mode, accumulating results from each step incrementally.
- **FR-002**: System MUST log each graph step completion with the node name and log total execution duration.
- **FR-003**: System MUST include rich metadata for each source in responses: uri, title, type, and relevance score (0-10 scale).
- **FR-004**: System MUST deduplicate sources by URI, retaining the entry with the richest metadata.
- **FR-005**: System MUST include language metadata (ISO-639-1) for human language, result language, and knowledge language in every response.
- **FR-006**: System MUST return a graceful fallback response when graph execution fails, with the exception logged.
- **FR-006a**: When a knowledge base collection is unavailable during retrieval, the system MUST skip the failed collection, log a warning, and continue with results from the remaining collections.
- **FR-006b**: When source document metadata is missing title or type fields, the system MUST use sensible defaults (empty string for title, "unknown" for type) rather than omitting the source.
- **FR-007**: Project MUST NOT include faiss-cpu, gitpython, beautifulsoup4, or tiktoken as dependencies.
- **FR-008**: Dockerfile MUST NOT include a Hugo installation stage or a Git-dependent full runtime stage.
- **FR-009**: Project MUST include a CLAUDE.md file with: project overview, active technologies and versions, key code patterns, and code style guidelines.
- **FR-010**: Project MUST include a CI workflow that runs on pull requests to develop.
- **FR-011**: CI workflow MUST execute linting, tests with coverage reporting, and Docker build verification.
- **FR-012**: CI workflow MUST fail if test coverage drops below 90%.
- **FR-013**: Project MUST include a test suite achieving at least 90% code coverage across the core source files.

### Key Entities

- **Query Response**: The structured response returned for each query, containing: result text, original result, sources (with uri, title, type, score), human language, result language, knowledge language.
- **Source**: A knowledge base document reference with uri, title, document type, and relevance score.
- **Graph Step**: An individual node execution within the prompt graph stream, identified by node name and producing partial output.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: Every query response includes source metadata with uri, title, type, and score fields when relevant sources exist.
- **SC-002**: Graph execution logs show per-step completion and total duration for every query processed.
- **SC-003**: Test suite achieves at least 90% code coverage.
- **SC-004**: CI pipeline runs linting, tests, and Docker build on every pull request to develop, blocking merge on failure.
- **SC-005**: Container image builds successfully without Hugo or unused legacy dependencies.
- **SC-006**: New contributors can orient themselves using CLAUDE.md without needing to read the full codebase.

## Assumptions

- The base library v0.8.0 supports stream-based graph execution (graph.stream with "updates" mode) — confirmed by the expert engine's usage.
- Knowledge base document metadata already contains source, title, and type fields — consistent with the expert engine's usage of these fields.
- The existing main.py entry point and RabbitMQ integration remain unchanged; only the ai_adapter query handling logic changes.
- The CI platform is GitHub Actions, consistent with the existing .github/ directory in the project.
- Tests will mock external dependencies (knowledge base, message queue, LLM) rather than requiring live services.
- The ingest operation is no longer supported by this engine — only query and reset operations remain.

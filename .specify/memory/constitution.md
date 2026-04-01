<!--
Sync Impact Report
- Version change: N/A → 1.0.0 (initial creation)
- Principles added:
  1. Knowledge-Grounded Responses
  2. Async Message-Driven Architecture
  3. Source Attribution & Scoring
  4. Observability
  5. Security & Prompt Integrity
  6. Test Coverage
- Sections added:
  - Technology Stack Constraints
  - Development Workflow
  - Governance
- Templates requiring updates:
  - .specify/templates/plan-template.md ✅ no changes needed (generic)
  - .specify/templates/spec-template.md ✅ no changes needed (generic)
  - .specify/templates/tasks-template.md ✅ no changes needed (generic)
- Follow-up TODOs: RATIFICATION_DATE set to today (first adoption)
-->

# Virtual Contributor Guidance Engine Constitution

## Core Principles

### I. Knowledge-Grounded Responses

All generated answers MUST be derived exclusively from the Alkemio website
knowledge collections stored in ChromaDB. The system MUST NOT hallucinate,
speculate, or supplement answers with information outside the retrieved
documents. When the knowledge base contains insufficient information to
answer a query, the system MUST explicitly state that it cannot answer
rather than fabricate a response.

**Rationale**: The engine serves as the general-purpose guidance persona
for the Alkemio platform. Ungrounded answers erode trust and may propagate
misinformation to users seeking platform information.

### II. Async Message-Driven Architecture

All request handling MUST be fully asynchronous, using RabbitMQ as the
message broker. The engine MUST NOT expose synchronous HTTP endpoints or
block the event loop. New features MUST integrate with the existing
`aio-pika` message consumer pattern and the
`alkemio-virtual-contributor-engine` base library.

**Rationale**: The engine runs as one of potentially many virtual
contributors within the Alkemio platform. Async message-driven design
ensures the system scales horizontally and integrates cleanly with the
platform's event bus.

### III. Source Attribution & Scoring

Every response MUST include the source documents used to generate the
answer, each with rich metadata: source URI, title, document type, and
a relevance score (0–10 scale). Responses MUST also include language
detection metadata (ISO-639-1 format) for the question, answer, and
knowledge base content.

**Rationale**: Users and downstream systems rely on source attribution
to evaluate answer quality and trace information provenance. Rich
metadata enables filtering, ranking, and presentation by consuming
services.

### IV. Observability

All LLM interactions MUST be traceable via LangSmith (or equivalent
tracing backend). The system MUST use structured logging at appropriate
levels. New features MUST NOT degrade existing tracing or logging
coverage. Error conditions MUST produce actionable log entries with
sufficient context for debugging.

**Rationale**: LLM-based systems are inherently non-deterministic.
Without end-to-end observability, diagnosing quality regressions,
latency issues, or incorrect answers becomes impractical in production.

### V. Security & Prompt Integrity

The system MUST enforce prompt boundaries that prevent user input from
overriding system instructions or persona definitions. The prompt graph
MUST constrain the LLM to respond only within the defined persona and
knowledge scope. New prompt modifications MUST be reviewed for injection
vulnerabilities. Sensitive configuration (API keys, credentials) MUST
be loaded from environment variables or secrets — never hardcoded.

**Rationale**: The engine processes untrusted user input and passes it
to an LLM. Without prompt integrity enforcement, adversarial inputs
could bypass knowledge grounding, leak system prompts, or cause the
persona to behave outside its intended role.

### VI. Test Coverage

All new features and bug fixes MUST include corresponding tests using
pytest. Test coverage MUST NOT decrease with new changes. Tests MUST
cover both success paths and meaningful error scenarios. Integration
tests MUST validate the full message-handling pipeline where applicable.

**Rationale**: The engine's correctness directly impacts user trust in
the Alkemio platform. Automated test coverage prevents regressions and
enables confident refactoring as the codebase evolves.

## Technology Stack Constraints

- **Language**: Python 3.12+
- **LLM Provider**: Mistral AI (mistral-small-latest or as configured)
- **Embeddings**: Scaleway-hosted model (Qwen3-Embedding-8B or as
  configured)
- **Vector Database**: ChromaDB for document storage and semantic
  retrieval
- **Message Queue**: RabbitMQ via aio-pika (async)
- **Orchestration**: LangChain for prompt graph compilation and
  stream-based execution
- **Base Library**: `alkemio-virtual-contributor-engine` — engine
  lifecycle, message handling, prompt graph (from_dict → compile →
  stream), and shared types
- **Validation**: Pydantic for all data models
- **Testing**: pytest with coverage tracking
- **Containerization**: Docker (multi-arch: x86_64, arm64), deployed
  on Kubernetes via Scaleway container registry
- **License**: EUPL-1.2

Changes to the core technology stack (LLM provider, vector DB, message
broker, or base library) MUST be treated as a major architectural
decision requiring explicit justification and a migration plan.

## Development Workflow

- All changes MUST be developed on feature branches and merged via pull
  request into `develop`.
- Version bumps follow semantic versioning (MAJOR.MINOR.PATCH).
- The `Dockerfile` MUST remain buildable and produce a working container
  after every merge to `develop`.
- Environment configuration MUST be documented in `.env.default` with
  sensible placeholder values for all required variables.
- Dependencies are managed via Poetry (`pyproject.toml` / `poetry.lock`).
  Dependency additions or upgrades MUST not break the existing lock file
  without explicit intent.

## Governance

This constitution defines the non-negotiable principles for the
virtual-contributor-guidance-engine project. All feature specifications,
implementation plans, and code changes MUST be evaluated against these
principles.

**Amendment procedure**:
1. Propose the change with rationale in a pull request modifying this
   file.
2. Document the version bump (MAJOR for principle removal/redefinition,
   MINOR for new principles or material expansion, PATCH for
   clarifications).
3. Update the Sync Impact Report at the top of this file.
4. Verify dependent templates still align with updated principles.

**Compliance**: All PRs and reviews SHOULD verify that changes do not
violate the core principles. The Constitution Check section in
implementation plans MUST reference these principles by number.

**Version**: 1.0.0 | **Ratified**: 2026-03-27 | **Last Amended**: 2026-03-27

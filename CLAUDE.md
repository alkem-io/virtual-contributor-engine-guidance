# virtual-contributor-engine-guidance Development Guidelines

## Project Overview

AI-powered guidance engine for the Alkemio platform. Receives questions via RabbitMQ, retrieves relevant documents from 3 hardcoded Alkemio website knowledge collections in ChromaDB, generates answers using Mistral AI with stream-based prompt graph execution, and returns responses with rich source citations (uri, title, type, score) and language metadata.

## Active Technologies

- Python 3.12+
- alkemio-virtual-contributor-engine v0.8.0 (base library)
- aio-pika 9.5.7 (RabbitMQ async client)
- ChromaDB (vector database, via base library)
- Mistral AI (LLM provider)
- LangChain / LangSmith (orchestration + tracing)

## Project Structure

```text
.
├── ai_adapter.py        # Core invocation logic (retrieve, stream-based graph execution, response building)
├── main.py              # Entry point, request handler, engine bootstrap
├── pyproject.toml       # Dependencies (Poetry)
├── Dockerfile           # Multi-stage container build (builder + runtime)
├── Dockerfile.local     # Local dev build (parent directory context)
├── .env.default         # Environment variable documentation
├── .flake8              # Linting configuration
├── tests/               # pytest test suite
│   ├── conftest.py      # Shared fixtures
│   ├── test_ai_adapter.py  # Tests for retrieve + invoke
│   └── test_main.py     # Tests for main entry point
└── .github/workflows/   # CI/CD pipelines
```

## Commands

```bash
# Install dependencies
poetry install

# Run the engine
poetry run python main.py

# Run tests
poetry run pytest

# Run tests with coverage
poetry run pytest --cov=ai_adapter --cov=main --cov-report=term-missing --cov-fail-under=90

# Run linting
poetry run flake8
```

## Code Style

- Follow flake8 rules (max-line-length=100, see `.flake8`)
- Use `setup_logger(__name__)` for all logging — never `print()`/`pprint()`
- All async request handling via aio-pika — no sync HTTP endpoints
- External dependencies (LLM, embeddings, vector DB) are accessed via the base library

## Key Patterns

- `PromptGraph.from_dict()` → `compile(llm=, special_nodes=)` → `graph.stream(input_state, stream_mode="updates")`
- Special nodes (e.g., `retrieve`) are injected at compile time
- Stream execution: iterate steps, log per node, accumulate result dict
- All provider config via environment variables — never hardcoded
- Response always includes: answer, sources with rich metadata (uri, title, type, score), language metadata
- 3 hardcoded collections: `alkem.io-knowledge`, `welcome.alkem.io-knowledge`, `www.alkemio.org-knowledge`
- Collection query failures are logged and skipped (partial results preferred over failure)

<!-- MANUAL ADDITIONS START -->
<!-- MANUAL ADDITIONS END -->

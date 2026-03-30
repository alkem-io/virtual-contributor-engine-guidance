# Research: Align Guidance Engine with Expert Engine Architecture

**Date**: 2026-03-27

## R1: Stream-Based Graph Execution Pattern

**Decision**: Use `graph.stream(input_state, stream_mode="updates")` with per-step result accumulation, matching the expert engine pattern.

**Rationale**: The expert engine (ai_adapter.py) demonstrates the proven pattern:
```python
result = {}
for step in graph.stream(input_state, stream_mode="updates"):
    for node_name, node_output in step.items():
        logger.info(f"Step '{node_name}' completed")
        result.update(node_output)
```
This provides per-step observability and progressive result building. The base library's `PromptGraph.compile()` returns a LangGraph `CompiledGraph` which supports both `.invoke()` and `.stream()`.

**Alternatives considered**:
- `graph.invoke()` (current): Simpler but no per-step logging. Rejected because it doesn't meet observability requirements.
- `stream_mode="values"`: Returns full state after each step. Rejected because "updates" mode returns only the delta, which is more efficient for logging.

## R2: Rich Source Metadata Construction

**Decision**: Build source entries using full ChromaDB metadata dict merged with computed fields (score, formatted title, uri alias), deduplicate by `source` key.

**Rationale**: The expert engine pattern:
```python
sources.append(
    dict(doc) | {
        "score": source_scores[str_index],
        "uri": doc["source"],
        "title": "[{}] {}".format(
            re.sub(r'(?<=[a-z])(?=[A-Z])|_', ' ', str(doc["type"])).capitalize(),
            doc["title"],
        ),
    }
)
```
This preserves all original metadata while adding computed fields. The title format is `[Type] Title` with camelCase/snake_case type names humanized.

**Alternatives considered**:
- Constructing sources with only the 4 required fields: Rejected because preserving full metadata allows downstream flexibility.
- Separate title formatting logic: Rejected — inline regex is simple enough and matches the expert engine.

## R3: Collection Failure Handling in Retrieve

**Decision**: Wrap each collection query in try/except, log warning on failure, continue with remaining collections.

**Rationale**: Per clarification decision (spec Session 2026-03-27). The current guidance engine has no error handling per-collection. The expert engine queries a single collection so this pattern is unique to guidance. Defensive per-collection handling ensures partial results are better than no results.

**Alternatives considered**:
- Fail entire query on any collection error: Rejected per clarification — too aggressive for a guidance persona.

## R4: Missing Metadata Defaults

**Decision**: When source metadata lacks `title` or `type`, use `""` for title and `"unknown"` for type.

**Rationale**: Per clarification decision. Using `dict.get("title", "")` and `dict.get("type", "unknown")` ensures the source entry always has consistent shape for consuming services.

## R5: Test Strategy

**Decision**: Use pytest with pytest-asyncio for async tests, pytest-cov for coverage. Mock external dependencies using unittest.mock.

**Rationale**: The project uses async handlers, so pytest-asyncio is needed for testing `invoke()`. External dependencies (ChromaDB via `query_documents`, LLM via `mistral_small`, `PromptGraph`) should be mocked since they require live services. Target files: ai_adapter.py (~90 lines) and main.py (~25 lines).

**Key test scenarios**:
- `retrieve()`: mock `query_documents` for each collection, test aggregation, test partial failure, test empty results, test missing metadata
- `invoke()`: mock `PromptGraph` and graph stream, test happy path with rich sources, test error fallback, test missing prompt_graph
- `main.py`: test handler registration and engine start

**Alternatives considered**:
- Integration tests with live ChromaDB: Rejected — requires infrastructure setup, not suitable for CI.

## R6: CI Pipeline Design

**Decision**: Single GitHub Actions workflow (`ci.yml`) triggered on PRs to `develop`, with 3 jobs: lint, test, docker-build.

**Rationale**: Existing workflows are deploy-only (triggered on push to develop). A separate CI workflow for PRs provides pre-merge quality gates. Three parallel jobs minimize feedback time.

**Workflow structure**:
1. **lint**: Install Poetry, install dev deps, run `flake8`
2. **test**: Install Poetry, install all deps, run `pytest --cov --cov-fail-under=90`
3. **docker-build**: Run `docker build` to verify Dockerfile validity (no push)

**Alternatives considered**:
- Adding lint/test steps to existing deploy workflows: Rejected — deploy workflows run on push to develop (post-merge), not on PRs.
- Single sequential job: Rejected — parallel jobs are faster.

## R7: Legacy Dependency Cleanup

**Decision**: Remove faiss-cpu, gitpython, beautifulsoup4, tiktoken from pyproject.toml. Remove Dockerfile Stage 3 (runtime-full with Hugo).

**Rationale**: These dependencies were used for the legacy ingest pipeline (cloning Hugo websites, parsing HTML, building FAISS indexes). The engine now uses ChromaDB via the base library and no longer supports `ingest`. Confirmed unused — no imports in current ai_adapter.py or main.py.

**Impact**: poetry.lock will be regenerated (significant diff). Docker image will be simpler (2 stages instead of 3).

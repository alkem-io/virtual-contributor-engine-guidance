# Data Model: Align Guidance Engine with Expert Engine Architecture

**Date**: 2026-03-27

## Entities

### QueryResponse

The structured response returned by the engine for each query.

| Field | Type | Description |
|-------|------|-------------|
| result | string | Final answer text (potentially translated/refined) |
| original_result | string | Raw knowledge answer before any post-processing |
| human_language | string | ISO-639-1 code of the user's question language |
| result_language | string | ISO-639-1 code of the answer language |
| knowledge_language | string | ISO-639-1 code of the knowledge base content language |
| sources | list[Source] | Deduplicated list of source documents used (may be empty) |
| source_scores | dict | Raw score mapping (source index → relevance score) |

**Notes**: The `Response` type is defined in the base library (`alkemio-virtual-contributor-engine`). The engine constructs a dict matching this shape and unpacks it into `Response(**json_result)`.

### Source

A knowledge base document reference included in the response.

| Field | Type | Description |
|-------|------|-------------|
| uri | string | Document URI (aliased from metadata `source` field) |
| title | string | Formatted as `[Type] Title` where Type is humanized from metadata |
| type | string | Document type from ChromaDB metadata |
| score | number | Relevance score (0-10 scale) from LLM source scoring |
| source | string | Raw source URI from ChromaDB metadata |
| *(other)* | any | Additional ChromaDB metadata fields preserved as-is |

**Defaults**: If `title` is missing → `""`. If `type` is missing → `"unknown"`.
**Deduplication**: By `source` field (raw URI). Last occurrence wins.

### GraphStep

An individual node execution within the prompt graph stream.

| Field | Type | Description |
|-------|------|-------------|
| node_name | string | Name of the graph node that completed |
| node_output | dict | Partial state produced by this step |

**Notes**: Not a persisted entity — exists only during stream iteration for logging purposes.

### KnowledgeDocs

Aggregated ChromaDB query results across all 3 collections.

| Field | Type | Description |
|-------|------|-------------|
| documents | list[list[string]] | Retrieved document texts (nested: outer=batch, inner=results) |
| metadatas | list[list[dict]] | Metadata for each document (source, title, type, etc.) |
| distances | list[list[float]] | Embedding distances for each document |

**Notes**: Aggregated by concatenating results from `alkem.io-knowledge`, `welcome.alkem.io-knowledge`, and `www.alkemio.org-knowledge` collections.

## Relationships

```text
QueryResponse 1──* Source       (response contains 0..N deduplicated sources)
QueryResponse 1──1 KnowledgeDocs (raw retrieval results used to build sources)
GraphStep *──1 QueryResponse    (stream produces steps that accumulate into response)
```

## State Transitions

### Graph Execution Flow

```text
[Input received] → [Compile prompt graph] → [Stream execution starts]
    → [Step: retrieve] → [Step: ...LLM nodes...] → [Stream complete]
    → [Build response with sources] → [Return Response]

Error at any step → [Log exception] → [Return fallback Response]
```

### Collection Query (within retrieve)

```text
For each of 3 collections:
    [Query collection] → Success → [Append to aggregated results]
                       → Failure → [Log warning, skip, continue]
```

from unittest.mock import MagicMock
import pytest


@pytest.fixture
def mock_query_documents():
    """Mock query_documents to return ChromaDB-style results."""
    def _make_result(docs=None, metadatas=None, distances=None):
        return {
            "documents": [docs or []],
            "metadatas": [metadatas or []],
            "distances": [distances or []],
        }
    return _make_result


@pytest.fixture
def sample_chromadb_result():
    """A realistic ChromaDB query result for one collection."""
    return {
        "documents": [["doc1 content", "doc2 content"]],
        "metadatas": [[
            {"source": "https://alkem.io/page1", "title": "Page One", "type": "webPage"},
            {"source": "https://alkem.io/page2", "title": "Page Two", "type": "blogPost"},
        ]],
        "distances": [[0.1, 0.3]],
    }


@pytest.fixture
def sample_graph_stream_result():
    """Simulated graph.stream() output (list of step dicts)."""
    return [
        {"retrieve": {
            "knowledge_docs": {
                "documents": [["doc1", "doc2"]],
                "metadatas": [[
                    {"source": "https://alkem.io/p1", "title": "P1", "type": "webPage"},
                    {"source": "https://alkem.io/p2", "title": "P2", "type": "blogPost"},
                ]],
                "distances": [[0.1, 0.2]],
            },
            "combined_knowledge_docs": "doc1\n\ndoc2",
        }},
        {"generate": {
            "knowledge_answer": "Test answer",
            "final_answer": "Test answer",
            "source_scores": {"0": 8, "1": 5},
            "human_language": "en",
            "answer_language": "en",
            "knowledge_language": "en",
        }},
    ]


@pytest.fixture
def mock_input():
    """Create a mock Input object."""
    inp = MagicMock()
    inp.prompt_graph = {"nodes": [], "edges": []}
    inp.message = "What is Alkemio?"
    inp.history = []
    inp.body_of_knowledge_id = "test-bok"
    inp.description = "Test persona"
    inp.display_name = "Test VC"
    return inp


@pytest.fixture
def mock_prompt_graph():
    """Mock PromptGraph that returns a mock compiled graph."""
    mock_pg = MagicMock()
    mock_compiled = MagicMock()
    mock_pg.compile.return_value = mock_compiled
    return mock_pg, mock_compiled

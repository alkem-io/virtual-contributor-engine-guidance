from unittest.mock import MagicMock, patch
import pytest


# --- retrieve() tests ---

class TestRetrieve:
    """Tests for the retrieve() function."""

    @patch("ai_adapter.combine_query_results")
    @patch("ai_adapter.query_documents")
    def test_aggregates_across_all_collections(
        self, mock_qd, mock_combine, sample_chromadb_result
    ):
        mock_qd.return_value = sample_chromadb_result
        mock_combine.return_value = "combined"

        state = MagicMock()
        state.rephrased_question = "test query"
        state.messages = [{"content": "test"}]

        from ai_adapter import retrieve
        result = retrieve(state)

        assert mock_qd.call_count == 3
        # 2 docs per collection * 3 collections = 6
        assert len(result["knowledge_docs"]["documents"][0]) == 6
        assert len(result["knowledge_docs"]["metadatas"][0]) == 6
        assert len(result["knowledge_docs"]["distances"][0]) == 6
        mock_combine.assert_called_once()

    @patch("ai_adapter.combine_query_results")
    @patch("ai_adapter.query_documents")
    def test_partial_collection_failure(
        self, mock_qd, mock_combine, sample_chromadb_result
    ):
        """When one collection fails, others still contribute."""
        mock_qd.side_effect = [
            sample_chromadb_result,
            Exception("Collection unavailable"),
            sample_chromadb_result,
        ]
        mock_combine.return_value = "combined"

        state = MagicMock()
        state.rephrased_question = "test query"
        state.messages = [{"content": "test"}]

        from ai_adapter import retrieve
        result = retrieve(state)

        # 2 successful collections * 2 docs each = 4
        assert len(result["knowledge_docs"]["documents"][0]) == 4

    @patch("ai_adapter.combine_query_results")
    @patch("ai_adapter.query_documents")
    def test_all_collections_empty(self, mock_qd, mock_combine):
        """When all collections return empty results."""
        mock_qd.return_value = {
            "documents": [[]],
            "metadatas": [[]],
            "distances": [[]],
        }
        mock_combine.return_value = ""

        state = MagicMock()
        state.rephrased_question = "test query"
        state.messages = [{"content": "test"}]

        from ai_adapter import retrieve
        result = retrieve(state)

        assert result["knowledge_docs"]["documents"][0] == []
        assert result["knowledge_docs"]["metadatas"][0] == []

    @patch("ai_adapter.combine_query_results")
    @patch("ai_adapter.query_documents")
    def test_all_collections_fail(self, mock_qd, mock_combine):
        """When all collections raise exceptions."""
        mock_qd.side_effect = Exception("DB down")
        mock_combine.return_value = ""

        state = MagicMock()
        state.rephrased_question = "test query"
        state.messages = [{"content": "test"}]

        from ai_adapter import retrieve
        result = retrieve(state)

        assert result["knowledge_docs"]["documents"][0] == []

    @patch("ai_adapter.combine_query_results")
    @patch("ai_adapter.query_documents")
    def test_uses_rephrased_question_when_available(
        self, mock_qd, mock_combine
    ):
        mock_qd.return_value = {
            "documents": [[]],
            "metadatas": [[]],
            "distances": [[]],
        }
        mock_combine.return_value = ""

        state = MagicMock()
        state.rephrased_question = "rephrased query"
        state.messages = [{"content": "original"}]

        from ai_adapter import retrieve
        retrieve(state)

        first_call_query = mock_qd.call_args_list[0][0][0]
        assert first_call_query == "rephrased query"

    @patch("ai_adapter.combine_query_results")
    @patch("ai_adapter.query_documents")
    def test_falls_back_to_message_content(
        self, mock_qd, mock_combine
    ):
        mock_qd.return_value = {
            "documents": [[]],
            "metadatas": [[]],
            "distances": [[]],
        }
        mock_combine.return_value = ""

        state = MagicMock()
        state.rephrased_question = None
        state.messages = [{"content": "fallback message"}]

        from ai_adapter import retrieve
        retrieve(state)

        first_call_query = mock_qd.call_args_list[0][0][0]
        assert first_call_query == "fallback message"

    @patch("ai_adapter.combine_query_results")
    @patch("ai_adapter.query_documents")
    def test_query_returns_none(self, mock_qd, mock_combine):
        """When query_documents returns None."""
        mock_qd.return_value = None
        mock_combine.return_value = ""

        state = MagicMock()
        state.rephrased_question = "test"
        state.messages = [{"content": "test"}]

        from ai_adapter import retrieve
        result = retrieve(state)

        assert result["knowledge_docs"]["documents"][0] == []

    @patch("ai_adapter.combine_query_results")
    @patch("ai_adapter.query_documents")
    def test_empty_messages_no_rephrased(
        self, mock_qd, mock_combine
    ):
        """When both rephrased_question and messages are empty."""
        mock_qd.return_value = {
            "documents": [[]],
            "metadatas": [[]],
            "distances": [[]],
        }
        mock_combine.return_value = ""

        state = MagicMock()
        state.rephrased_question = None
        state.messages = []

        from ai_adapter import retrieve
        retrieve(state)

        first_call_query = mock_qd.call_args_list[0][0][0]
        assert first_call_query == ""


# --- invoke() tests ---

class TestInvoke:
    """Tests for the invoke() function."""

    @pytest.mark.asyncio
    @patch("ai_adapter.history_as_conversation", return_value="conv")
    @patch("ai_adapter.history_as_dict", return_value=[
        {"content": "What is Alkemio?", "role": "human"}
    ])
    @patch("ai_adapter.PromptGraph")
    async def test_happy_path_with_rich_sources(
        self, MockPG, mock_hd, mock_hc,
        mock_input, sample_graph_stream_result
    ):
        mock_compiled = MagicMock()
        mock_compiled.stream.return_value = iter(
            sample_graph_stream_result
        )
        mock_pg_instance = MagicMock()
        mock_pg_instance.compile.return_value = mock_compiled
        MockPG.from_dict.return_value = mock_pg_instance

        from ai_adapter import invoke
        response = await invoke(mock_input)

        assert response.result == "Test answer"
        assert response.original_result == "Test answer"
        assert response.human_language == "en"
        assert len(response.sources) > 0
        for src in response.sources:
            assert hasattr(src, "uri") or "uri" in src
            assert hasattr(src, "title") or "title" in src
            assert hasattr(src, "score") or "score" in src

    @pytest.mark.asyncio
    async def test_missing_prompt_graph_returns_fallback(
        self, mock_input
    ):
        mock_input.prompt_graph = None

        from ai_adapter import invoke
        response = await invoke(mock_input)

        assert "unavailable" in response.result.lower()
        assert response.sources == []

    @pytest.mark.asyncio
    @patch("ai_adapter.history_as_conversation", return_value="conv")
    @patch("ai_adapter.history_as_dict", return_value=[])
    @patch("ai_adapter.PromptGraph")
    async def test_graph_stream_exception_returns_fallback(
        self, MockPG, mock_hd, mock_hc, mock_input
    ):
        mock_compiled = MagicMock()
        mock_compiled.stream.side_effect = RuntimeError("LLM error")
        mock_pg_instance = MagicMock()
        mock_pg_instance.compile.return_value = mock_compiled
        MockPG.from_dict.return_value = mock_pg_instance

        from ai_adapter import invoke
        response = await invoke(mock_input)

        assert "unavailable" in response.result.lower()
        assert response.sources == []

    @pytest.mark.asyncio
    @patch("ai_adapter.history_as_conversation", return_value="conv")
    @patch("ai_adapter.history_as_dict", return_value=[
        {"content": "test", "role": "human"}
    ])
    @patch("ai_adapter.PromptGraph")
    async def test_all_scores_zero_yields_empty_sources(
        self, MockPG, mock_hd, mock_hc, mock_input
    ):
        stream_result = [
            {"retrieve": {
                "knowledge_docs": {
                    "documents": [["doc1"]],
                    "metadatas": [[
                        {"source": "https://a.io/1",
                         "title": "T", "type": "web"},
                    ]],
                    "distances": [[0.1]],
                },
                "combined_knowledge_docs": "doc1",
            }},
            {"generate": {
                "knowledge_answer": "Answer",
                "final_answer": "Answer",
                "source_scores": {"0": 0},
                "human_language": "en",
                "answer_language": "en",
                "knowledge_language": "en",
            }},
        ]
        mock_compiled = MagicMock()
        mock_compiled.stream.return_value = iter(stream_result)
        mock_pg_instance = MagicMock()
        mock_pg_instance.compile.return_value = mock_compiled
        MockPG.from_dict.return_value = mock_pg_instance

        from ai_adapter import invoke
        response = await invoke(mock_input)

        assert response.result == "Answer"
        assert response.sources == []

    @pytest.mark.asyncio
    @patch("ai_adapter.history_as_conversation", return_value="conv")
    @patch("ai_adapter.history_as_dict", return_value=[
        {"content": "test", "role": "human"}
    ])
    @patch("ai_adapter.PromptGraph")
    async def test_duplicate_source_uris_deduplicated(
        self, MockPG, mock_hd, mock_hc, mock_input
    ):
        stream_result = [
            {"retrieve": {
                "knowledge_docs": {
                    "documents": [["d1", "d2"]],
                    "metadatas": [[
                        {"source": "https://a.io/same",
                         "title": "First", "type": "web"},
                        {"source": "https://a.io/same",
                         "title": "Second", "type": "web"},
                    ]],
                    "distances": [[0.1, 0.2]],
                },
                "combined_knowledge_docs": "d1\nd2",
            }},
            {"generate": {
                "knowledge_answer": "Answer",
                "final_answer": "Answer",
                "source_scores": {"0": 5, "1": 8},
                "human_language": "en",
                "answer_language": "en",
                "knowledge_language": "en",
            }},
        ]
        mock_compiled = MagicMock()
        mock_compiled.stream.return_value = iter(stream_result)
        mock_pg_instance = MagicMock()
        mock_pg_instance.compile.return_value = mock_compiled
        MockPG.from_dict.return_value = mock_pg_instance

        from ai_adapter import invoke
        response = await invoke(mock_input)

        sources = response.sources
        uris = [
            getattr(s, "source", None) or getattr(s, "uri", "")
            for s in sources
        ]
        # Deduplicated: only one entry for same URI
        assert len(sources) == 1
        assert len(set(uris)) == len(uris)

    @pytest.mark.asyncio
    @patch("ai_adapter.history_as_conversation", return_value="conv")
    @patch("ai_adapter.history_as_dict", return_value=[
        {"content": "test", "role": "human"}
    ])
    @patch("ai_adapter.PromptGraph")
    async def test_missing_title_type_uses_defaults(
        self, MockPG, mock_hd, mock_hc, mock_input
    ):
        stream_result = [
            {"retrieve": {
                "knowledge_docs": {
                    "documents": [["doc1"]],
                    "metadatas": [[
                        {"source": "https://a.io/1"},
                    ]],
                    "distances": [[0.1]],
                },
                "combined_knowledge_docs": "doc1",
            }},
            {"generate": {
                "knowledge_answer": "Answer",
                "final_answer": "Answer",
                "source_scores": {"0": 7},
                "human_language": "en",
                "answer_language": "en",
                "knowledge_language": "en",
            }},
        ]
        mock_compiled = MagicMock()
        mock_compiled.stream.return_value = iter(stream_result)
        mock_pg_instance = MagicMock()
        mock_pg_instance.compile.return_value = mock_compiled
        MockPG.from_dict.return_value = mock_pg_instance

        from ai_adapter import invoke
        response = await invoke(mock_input)

        assert len(response.sources) == 1
        src = response.sources[0]
        title = getattr(src, "title", "") or ""
        score = getattr(src, "score", None)
        assert "[Unknown]" in title
        assert score == 7

    @pytest.mark.asyncio
    @patch("ai_adapter.history_as_conversation", return_value="conv")
    @patch("ai_adapter.history_as_dict", return_value=[])
    @patch("ai_adapter.PromptGraph")
    async def test_empty_history(
        self, MockPG, mock_hd, mock_hc, mock_input
    ):
        stream_result = [
            {"generate": {
                "knowledge_answer": "Answer",
                "final_answer": "Answer",
                "source_scores": {},
                "human_language": "en",
                "answer_language": "en",
                "knowledge_language": "en",
            }},
        ]
        mock_compiled = MagicMock()
        mock_compiled.stream.return_value = iter(stream_result)
        mock_pg_instance = MagicMock()
        mock_pg_instance.compile.return_value = mock_compiled
        MockPG.from_dict.return_value = mock_pg_instance

        from ai_adapter import invoke
        response = await invoke(mock_input)

        assert response.result == "Answer"
        assert response.sources == []

    @pytest.mark.asyncio
    @patch("ai_adapter.history_as_conversation", return_value="conv")
    @patch("ai_adapter.history_as_dict", return_value=[
        {"content": "test", "role": "human"}
    ])
    @patch("ai_adapter.PromptGraph")
    async def test_result_language_uses_answer_language(
        self, MockPG, mock_hd, mock_hc, mock_input
    ):
        """Verify result_language maps to answer_language, not knowledge_language."""
        stream_result = [
            {"generate": {
                "knowledge_answer": "Respuesta",
                "final_answer": "Respuesta",
                "source_scores": {},
                "human_language": "es",
                "answer_language": "es",
                "knowledge_language": "en",
            }},
        ]
        mock_compiled = MagicMock()
        mock_compiled.stream.return_value = iter(stream_result)
        mock_pg_instance = MagicMock()
        mock_pg_instance.compile.return_value = mock_compiled
        MockPG.from_dict.return_value = mock_pg_instance

        from ai_adapter import invoke
        response = await invoke(mock_input)

        assert response.result_language == "es"
        assert response.knowledge_language == "en"

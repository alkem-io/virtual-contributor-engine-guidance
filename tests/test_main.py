from unittest.mock import patch, MagicMock

# Import main at module level so coverage can track it.
# The patches must be active before import to prevent
# actual engine startup.
with patch("asyncio.run"), \
     patch(
         "alkemio_virtual_contributor_engine"
         ".alkemio_vc_engine.AlkemioVirtualContributorEngine"
     ):
    import main  # noqa: F401


class TestMain:
    """Tests for main.py entry point."""

    @patch("asyncio.run")
    @patch(
        "alkemio_virtual_contributor_engine"
        ".alkemio_vc_engine.AlkemioVirtualContributorEngine"
    )
    def test_engine_creation_and_start(
        self, MockEngine, mock_run
    ):
        mock_instance = MagicMock()
        mock_instance.start = MagicMock(
            return_value="started"
        )
        MockEngine.return_value = mock_instance

        import importlib
        importlib.reload(main)

        assert MockEngine.call_count >= 1
        assert mock_instance.register_handler.call_count >= 1
        assert mock_run.call_count >= 1

    @patch("asyncio.run")
    @patch(
        "alkemio_virtual_contributor_engine"
        ".alkemio_vc_engine.AlkemioVirtualContributorEngine"
    )
    def test_registered_handler_is_callable(
        self, MockEngine, mock_run
    ):
        mock_instance = MagicMock()
        mock_instance.start = MagicMock(
            return_value="started"
        )
        MockEngine.return_value = mock_instance

        import importlib
        importlib.reload(main)

        handler = mock_instance.register_handler.call_args[0][0]
        assert callable(handler)

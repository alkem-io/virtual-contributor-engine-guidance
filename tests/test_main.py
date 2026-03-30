from unittest.mock import patch, MagicMock


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
        import main
        importlib.reload(main)

        # reload may trigger multiple calls if already imported
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
        import main
        importlib.reload(main)

        handler = mock_instance.register_handler.call_args[0][0]
        assert callable(handler)

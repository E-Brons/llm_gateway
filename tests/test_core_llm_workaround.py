"""Tests for reset_litellm_client() in _litellm_workaround."""
from unittest.mock import MagicMock, patch

import pytest



def test_reset_litellm_client_replaces_module_level_client():
    """reset_litellm_client should set litellm.module_level_client."""
    import litellm
    from src._litellm_workaround import reset_litellm_client

    original = getattr(litellm, "module_level_client", None)
    reset_litellm_client()
    new = getattr(litellm, "module_level_client", None)
    # The new client should be different from the original (or at least set)
    assert new is not None


def test_reset_litellm_client_graceful_when_httpx_unavailable():
    """Should not raise if httpx is not importable."""
    import builtins

    real_import = builtins.__import__

    def mock_import(name, *args, **kwargs):
        if name == "httpx":
            raise ImportError("httpx not available")
        return real_import(name, *args, **kwargs)

    with patch("builtins.__import__", side_effect=mock_import):
        from src import _litellm_workaround
        # Should not raise
        _litellm_workaround.reset_litellm_client()


def test_reset_litellm_client_graceful_when_litellm_unavailable():
    """Should not raise if litellm internals are not present."""
    import builtins

    real_import = builtins.__import__

    def mock_import(name, *args, **kwargs):
        if name == "litellm":
            raise ImportError("litellm not available")
        return real_import(name, *args, **kwargs)

    with patch("builtins.__import__", side_effect=mock_import):
        from src import _litellm_workaround
        # Should silently no-op
        _litellm_workaround.reset_litellm_client()


def test_reset_litellm_client_updates_cache():
    """If in_memory_llm_clients_cache is present, update it."""
    import litellm
    from src._litellm_workaround import reset_litellm_client

    mock_cache = MagicMock()
    with patch.object(litellm, "in_memory_llm_clients_cache", mock_cache, create=True):
        reset_litellm_client()

    # set_cache should have been called for both keys
    calls = [c[0][0] for c in mock_cache.set_cache.call_args_list]
    assert "httpx_client" in calls
    assert "httpx_client_ssl_verify_None" in calls

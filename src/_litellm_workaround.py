"""Centralized httpx / litellm keep-alive workaround.

Ollama's /api/show endpoint returns ``Transfer-Encoding: chunked, chunked``
which httpx raises as a RemoteProtocolError.  Under sustained load this
corrupts the keep-alive TCP connection reused for subsequent completion calls.
Replacing litellm's HTTP clients with no-keepalive instances ensures every
request opens a fresh connection.
"""
from __future__ import annotations


def reset_litellm_client() -> None:
    """Replace litellm's HTTP clients with no-keepalive instances.

    Safe to call at any time; silently no-ops if httpx or litellm internals
    are unavailable (e.g. in environments without httpx installed).
    """
    try:
        import httpx
        import litellm
        from litellm.llms.custom_httpx.http_handler import HTTPHandler

        no_keepalive = httpx.Limits(max_connections=10, max_keepalive_connections=0)

        # module_level_client — used by Ollama's get_model_info (/api/show)
        litellm.module_level_client = HTTPHandler(
            client=httpx.Client(limits=no_keepalive, follow_redirects=True)
        )

        # in_memory_llm_clients_cache — used by actual completion calls
        cache = getattr(litellm, "in_memory_llm_clients_cache", None)
        if cache is not None:
            fresh = HTTPHandler(
                client=httpx.Client(limits=no_keepalive, follow_redirects=True)
            )
            try:
                cache.set_cache("httpx_client", fresh)
                cache.set_cache("httpx_client_ssl_verify_None", fresh)
            except Exception:
                pass
    except Exception:
        pass

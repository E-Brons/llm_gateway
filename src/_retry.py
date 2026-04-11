"""Shared retry helpers for text and image LLM calls."""

from __future__ import annotations

import time
from typing import Callable

from .responses import ImageResponse, TextResponse

# Exception types that indicate a timeout — retrying is pointless.
# We import lazily to avoid hard dependencies on requests/httpx/litellm.


def _is_client_error(exc: BaseException) -> bool:
    """Return True for 4xx HTTP errors — bad input, retrying the same payload won't help."""
    try:
        import requests.exceptions

        if isinstance(exc, requests.exceptions.HTTPError):
            resp = getattr(exc, "response", None)
            if resp is not None and 400 <= resp.status_code < 500:
                return True
    except ImportError:
        pass
    try:
        import httpx

        if isinstance(exc, httpx.HTTPStatusError) and 400 <= exc.response.status_code < 500:
            return True
    except ImportError:
        pass
    return False


def _is_timeout(exc: BaseException) -> bool:
    try:
        import requests.exceptions

        if isinstance(exc, (requests.exceptions.Timeout, requests.exceptions.ReadTimeout)):
            return True
    except ImportError:
        pass
    try:
        import httpx

        if isinstance(exc, (httpx.TimeoutException, httpx.ReadTimeout)):
            return True
    except ImportError:
        pass
    # litellm wraps timeouts — check the message as a fallback
    msg = str(exc).lower()
    return "timed out" in msg or "timeout" in msg or "read timeout" in msg


def retry_text_completion(
    call_fn: Callable[[list[dict]], tuple[str, str]],
    messages: list[dict],
    max_retries: int,
    model: str,
    *,
    on_transfer_error: Callable[[], None] | None = None,
) -> TextResponse:
    """Retry a text completion call, appending a correction hint on failure.

    Parameters
    ----------
    call_fn:
        Callable that receives the current messages list and returns
        ``(content, model_used)``.  Raise on hard errors; return empty string
        for a soft "empty response" failure.
    messages:
        Initial messages list.  A correction hint is appended in-place on
        each failed attempt.
    max_retries:
        Maximum number of attempts (including the first).
    model:
        Model string used in the returned TextResponse.
    on_transfer_error:
        Optional callback invoked when a ``Transfer-Encoding`` error is
        detected (e.g. to reset litellm's HTTP client).

    Returns
    -------
    TextResponse
        On success.

    Raises
    ------
    ValueError
        After *max_retries* failed attempts.
    """
    msgs = list(messages)
    last_error: str | None = None
    t0 = time.monotonic()

    for attempt in range(1, max_retries + 1):
        try:
            content, used_model = call_fn(msgs)
            if not content or not content.strip():
                err = f"attempt {attempt}: empty response"
                last_error = err
                msgs.append(
                    {
                        "role": "user",
                        "content": "Your last response was empty. Please try again.",
                    }
                )
                continue

            duration_ms = (time.monotonic() - t0) * 1000
            return TextResponse(
                content=content,
                model=used_model or model,
                duration_ms=duration_ms,
                attempts=attempt,
                last_error=last_error,
            )

        except Exception as exc:
            if _is_timeout(exc):
                raise
            if _is_client_error(exc):
                raise
            err = str(exc)
            last_error = err
            if "Transfer-Encoding" in err and on_transfer_error is not None:
                on_transfer_error()
            if attempt == max_retries:
                break
            msgs.append(
                {
                    "role": "user",
                    "content": f"An error occurred: {err}. Please try again.",
                }
            )

    duration_ms = (time.monotonic() - t0) * 1000
    raise ValueError(
        f"Text completion failed after {max_retries} attempts. Last error: {last_error}"
    )


def retry_image_generation(
    call_fn: Callable[[], tuple[bytes, str]],
    max_retries: int,
    model: str,
    *,
    validator: Callable[[bytes], bool] | None = None,
) -> ImageResponse:
    """Retry an image-generation call, validating output on each attempt.

    Parameters
    ----------
    call_fn:
        Callable that returns ``(image_bytes, model_used)``.  Return
        empty bytes to signal a soft failure.
    max_retries:
        Maximum number of attempts.
    model:
        Model string used in the returned ImageResponse.
    validator:
        Optional callable: receives the raw image bytes and returns True if
        the image is acceptable.

    Returns
    -------
    ImageResponse
        On success.

    Raises
    ------
    ValueError
        After *max_retries* failed attempts.
    """
    last_error: str | None = None
    t0 = time.monotonic()

    for attempt in range(1, max_retries + 1):
        try:
            image_bytes, used_model = call_fn()

            if not image_bytes:
                err = f"attempt {attempt}: no image data returned"
                last_error = err
                continue

            if validator is not None and not validator(image_bytes):
                err = f"attempt {attempt}: image failed validator"
                last_error = err
                continue

            duration_ms = (time.monotonic() - t0) * 1000
            return ImageResponse(
                image=image_bytes,
                model=used_model or model,
                duration_ms=duration_ms,
                attempts=attempt,
                last_error=last_error,
            )

        except Exception as exc:
            if _is_timeout(exc):
                raise
            if _is_client_error(exc):
                raise
            err = str(exc)
            last_error = err
            if attempt == max_retries:
                break

    duration_ms = (time.monotonic() - t0) * 1000
    raise ValueError(
        f"Image generation failed after {max_retries} attempts. Last error: {last_error}"
    )

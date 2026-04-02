"""Tests for retry_text_completion and retry_image_generation helpers."""
import pytest



# ---------------------------------------------------------------------------
# retry_text_completion
# ---------------------------------------------------------------------------


def test_retry_text_happy_path():
    from src._retry import retry_text_completion

    def call_fn(msgs):
        return "hello world", "test-model"

    result = retry_text_completion(call_fn, [{"role": "user", "content": "hi"}], 3, "test-model")
    assert result.content == "hello world"
    assert result.model == "test-model"
    assert result.attempts == 1
    assert result.last_error is None


def test_retry_text_empty_then_success():
    from src._retry import retry_text_completion

    calls = []

    def call_fn(msgs):
        calls.append(len(msgs))
        if len(calls) < 2:
            return "", "m"
        return "good", "m"

    result = retry_text_completion(call_fn, [{"role": "user", "content": "x"}], 3, "m")
    assert result.content == "good"
    assert result.attempts == 2
    assert result.last_error is not None  # empty response error from attempt 1


def test_retry_text_correction_hint_appended():
    """Verify a correction message is appended to msgs on empty response."""
    from src._retry import retry_text_completion

    seen_msg_counts = []

    def call_fn(msgs):
        seen_msg_counts.append(len(msgs))
        if len(seen_msg_counts) == 1:
            return "", "m"
        return "ok", "m"

    retry_text_completion(call_fn, [{"role": "user", "content": "q"}], 3, "m")
    # First call: 1 message; second call: 2 messages (original + correction)
    assert seen_msg_counts[0] == 1
    assert seen_msg_counts[1] == 2


def test_retry_text_exhaustion_raises():
    from src._retry import retry_text_completion

    def call_fn(msgs):
        return "", "m"

    with pytest.raises(ValueError, match="failed after 3 attempts"):
        retry_text_completion(call_fn, [{"role": "user", "content": "x"}], 3, "m")


def test_retry_text_exception_then_success():
    from src._retry import retry_text_completion

    calls = [0]

    def call_fn(msgs):
        calls[0] += 1
        if calls[0] == 1:
            raise RuntimeError("network error")
        return "recovered", "m"

    result = retry_text_completion(call_fn, [{"role": "user", "content": "x"}], 3, "m")
    assert result.content == "recovered"
    assert result.attempts == 2


def test_retry_text_exception_exhaustion_raises():
    from src._retry import retry_text_completion

    def call_fn(msgs):
        raise RuntimeError("always fails")

    with pytest.raises(ValueError, match="failed after 2 attempts"):
        retry_text_completion(call_fn, [{"role": "user", "content": "x"}], 2, "m")


def test_retry_text_transfer_encoding_callback():
    from src._retry import retry_text_completion

    callback_calls = [0]

    def on_transfer_error():
        callback_calls[0] += 1

    calls = [0]

    def call_fn(msgs):
        calls[0] += 1
        if calls[0] == 1:
            raise RuntimeError("Transfer-Encoding error")
        return "fixed", "m"

    result = retry_text_completion(
        call_fn,
        [{"role": "user", "content": "x"}],
        3,
        "m",
        on_transfer_error=on_transfer_error,
    )
    assert result.content == "fixed"
    assert callback_calls[0] == 1


def test_retry_text_attempt_count():
    from src._retry import retry_text_completion

    calls = [0]

    def call_fn(msgs):
        calls[0] += 1
        if calls[0] < 3:
            return "", "m"
        return "done", "m"

    result = retry_text_completion(call_fn, [{"role": "user", "content": "x"}], 5, "m")
    assert result.attempts == 3


# ---------------------------------------------------------------------------
# retry_image_generation
# ---------------------------------------------------------------------------


def test_retry_image_happy_path():
    from src._retry import retry_image_generation

    def call_fn():
        return b"\x89PNG", "img-model"

    result = retry_image_generation(call_fn, 3, "img-model")
    assert result.image == b"\x89PNG"
    assert result.attempts == 1
    assert result.last_error is None


def test_retry_image_empty_then_success():
    from src._retry import retry_image_generation

    calls = [0]

    def call_fn():
        calls[0] += 1
        if calls[0] < 2:
            return b"", "m"
        return b"data", "m"

    result = retry_image_generation(call_fn, 3, "m")
    assert result.image == b"data"
    assert result.attempts == 2


def test_retry_image_validator_fail_then_success():
    from src._retry import retry_image_generation

    calls = [0]

    def call_fn():
        calls[0] += 1
        return b"img", "m"

    def validator(img: bytes) -> bool:
        return calls[0] >= 2  # fail first time, pass second

    result = retry_image_generation(call_fn, 3, "m", validator=validator)
    assert result.image == b"img"
    assert result.attempts == 2


def test_retry_image_exhaustion_raises():
    from src._retry import retry_image_generation

    def call_fn():
        return b"", "m"

    with pytest.raises(ValueError, match="failed after 2 attempts"):
        retry_image_generation(call_fn, 2, "m")


def test_retry_image_exception_exhaustion_raises():
    from src._retry import retry_image_generation

    def call_fn():
        raise RuntimeError("always fails")

    with pytest.raises(ValueError, match="failed after 2 attempts"):
        retry_image_generation(call_fn, 2, "m")

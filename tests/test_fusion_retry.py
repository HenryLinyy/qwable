"""G13-3: tests for chat_with_retry helper."""

from unittest.mock import MagicMock

import pytest

from qwable.fusion_retry import _is_retryable, chat_with_retry


def test_retryable_recognizes_known_transients():
    assert _is_retryable(TimeoutError("x")) is True
    assert _is_retryable(ConnectionError("x")) is True


def test_retryable_rejects_programming_bugs():
    assert _is_retryable(ValueError("x")) is False
    assert _is_retryable(KeyError("x")) is False
    assert _is_retryable(TypeError("x")) is False


def test_retry_succeeds_on_first_try():
    """No retry if first attempt succeeds."""
    fn = MagicMock(return_value="ok")
    result = chat_with_retry(fn, max_retries=2, base_delay=0.01, model="m1")
    assert result == "ok"
    assert fn.call_count == 1


def test_retry_succeeds_on_second_try():
    """One failure → retry → success."""
    fn = MagicMock(side_effect=[TimeoutError("transient"), "ok"])
    result = chat_with_retry(fn, max_retries=2, base_delay=0.01, model="m1")
    assert result == "ok"
    assert fn.call_count == 2


def test_retry_succeeds_on_third_try():
    """Two failures → 2 retries → success."""
    fn = MagicMock(side_effect=[TimeoutError("e1"), ConnectionError("e2"), "ok"])
    result = chat_with_retry(fn, max_retries=2, base_delay=0.01, model="m1")
    assert result == "ok"
    assert fn.call_count == 3


def test_retry_exhausts_after_max_retries():
    """All retries fail → last exception raised."""
    fn = MagicMock(side_effect=TimeoutError("always"))
    with pytest.raises(TimeoutError):
        chat_with_retry(fn, max_retries=2, base_delay=0.01, model="m1")
    assert fn.call_count == 3  # 1 initial + 2 retries


def test_retry_disabled_when_max_retries_zero():
    """max_retries=0 → 1 attempt only, no retry."""
    fn = MagicMock(side_effect=TimeoutError("e"))
    with pytest.raises(TimeoutError):
        chat_with_retry(fn, max_retries=0, base_delay=0.01, model="m1")
    assert fn.call_count == 1


def test_retry_does_not_retry_value_error():
    """ValueError is NOT retryable (programming bug)."""
    fn = MagicMock(side_effect=ValueError("bad input"))
    with pytest.raises(ValueError):
        chat_with_retry(fn, max_retries=2, base_delay=0.01, model="m1")
    assert fn.call_count == 1  # no retry


def test_retry_passes_kwargs_to_fn():
    """All kwargs forwarded to fn on each attempt."""
    fn = MagicMock(side_effect=[TimeoutError("e"), "ok"])
    result = chat_with_retry(
        fn,
        max_retries=2,
        base_delay=0.01,
        model="m1",
        messages=[{"role": "user", "content": "hi"}],
        temperature=0.3,
    )
    assert result == "ok"
    # Both calls should have all kwargs
    for call in fn.call_args_list:
        assert call.kwargs["model"] == "m1"
        assert call.kwargs["messages"] == [{"role": "user", "content": "hi"}]
        assert call.kwargs["temperature"] == 0.3

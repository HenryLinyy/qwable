"""G11: Tests for OllamaClient.chat_completion_stream() sync iterator.

Verifies the streaming method yields (delta_text, finish_reason) tuples
from LM Studio /v1/chat/completions SSE responses.
"""

from unittest.mock import MagicMock


from qwable.models import OllamaClient


def _sse_line(data_obj):
    """Build a single SSE line: 'data: {json}\\n'."""
    import json

    return f"data: {json.dumps(data_obj)}\n"


def _make_client_with_stream_response(stream_lines):
    """Build a mock httpx-style stream context from a list of byte lines."""
    client = OllamaClient("http://127.0.0.1:1234/v1", backend="lmstudio")

    response = MagicMock()

    def iter_lines():
        for ln in stream_lines:
            yield ln.encode("utf-8") if isinstance(ln, str) else ln

    response.iter_lines = iter_lines
    response.__enter__ = lambda self: self
    response.__exit__ = lambda self, *args: None

    stream_ctx = MagicMock()
    stream_ctx.__enter__ = lambda self: response
    stream_ctx.__exit__ = lambda self, *args: None

    def fake_stream(method, url, json=None, **kwargs):
        return stream_ctx

    client.client = MagicMock()
    client.client.stream = fake_stream

    return client, stream_ctx


def test_chat_completion_stream_yields_token_deltas():
    """Each 'data:' line yields (delta, finish_reason)."""
    client, _ = _make_client_with_stream_response(
        [
            _sse_line(
                {"choices": [{"delta": {"content": "Hello"}, "finish_reason": None}]}
            ),
            _sse_line(
                {"choices": [{"delta": {"content": " world"}, "finish_reason": None}]}
            ),
            _sse_line({"choices": [{"delta": {}, "finish_reason": "stop"}]}),
        ]
    )
    chunks = list(
        client.chat_completion_stream(
            model="m1",
            messages=[{"role": "user", "content": "hi"}],
        )
    )
    assert chunks == [("Hello", None), (" world", None), ("", "stop")]


def test_chat_completion_stream_stops_at_done_marker():
    """A 'data: [DONE]' line ends the stream."""
    client, _ = _make_client_with_stream_response(
        [
            _sse_line(
                {"choices": [{"delta": {"content": "x"}, "finish_reason": None}]}
            ),
            "data: [DONE]\n",
            # Anything after [DONE] should not appear
            _sse_line(
                {
                    "choices": [
                        {
                            "delta": {"content": "should not appear"},
                            "finish_reason": None,
                        }
                    ]
                }
            ),
        ]
    )
    chunks = list(
        client.chat_completion_stream(
            model="m1",
            messages=[{"role": "user", "content": "hi"}],
        )
    )
    # Only one chunk before [DONE]
    assert len(chunks) == 1
    assert chunks[0] == ("x", None)


def test_chat_completion_stream_skips_blank_lines():
    """Blank / comment lines (start with ':') are ignored."""
    client, _ = _make_client_with_stream_response(
        [
            "\n",  # blank
            ": keep-alive\n",  # SSE comment
            _sse_line(
                {"choices": [{"delta": {"content": "x"}, "finish_reason": None}]}
            ),
        ]
    )
    chunks = list(
        client.chat_completion_stream(
            model="m1",
            messages=[{"role": "user", "content": "hi"}],
        )
    )
    assert len(chunks) == 1
    assert chunks[0][0] == "x"


def test_chat_completion_stream_unicode_safe():
    """Traditional Chinese in deltas must round-trip."""
    client, _ = _make_client_with_stream_response(
        [
            _sse_line(
                {"choices": [{"delta": {"content": "繁體中文"}, "finish_reason": None}]}
            ),
            _sse_line(
                {"choices": [{"delta": {"content": "測試"}, "finish_reason": "stop"}]}
            ),
        ]
    )
    chunks = list(
        client.chat_completion_stream(
            model="m1",
            messages=[{"role": "user", "content": "hi"}],
        )
    )
    assert chunks[0][0] == "繁體中文"
    assert chunks[1][0] == "測試"
    assert chunks[1][1] == "stop"


def test_chat_completion_stream_handles_empty_delta():
    """Lines with empty delta are still yielded (finish_reason may be set)."""
    client, _ = _make_client_with_stream_response(
        [
            _sse_line({"choices": [{"delta": {}, "finish_reason": None}]}),
            _sse_line({"choices": [{"delta": {}, "finish_reason": "length"}]}),
        ]
    )
    chunks = list(
        client.chat_completion_stream(
            model="m1",
            messages=[],
        )
    )
    assert chunks == [("", None), ("", "length")]

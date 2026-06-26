"""G13-1: tests for QwableSDK."""

from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from qwable_sdk import (
    FusionPreset,
    QwableClient,
)


def _mock_response(json_data: dict, status_code: int = 200) -> MagicMock:
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = json_data
    resp.raise_for_status = MagicMock()
    if status_code >= 400:
        resp.raise_for_status.side_effect = httpx.HTTPStatusError(
            f"{status_code}",
            request=MagicMock(),
            response=resp,
        )
    return resp


def test_client_initialization():
    """Default base_url, timeout, max_tokens."""
    client = QwableClient()
    assert client.base_url == "http://127.0.0.1:8088"
    assert client.timeout == 300.0
    assert client.max_tokens == 4000


def test_client_custom_init():
    """Custom base_url strips trailing slash."""
    client = QwableClient(base_url="http://example.com:9000/", timeout=60.0)
    assert client.base_url == "http://example.com:9000"


def test_list_presets_calls_correct_endpoint():
    """list_presets() hits GET /v1/fusion/presets."""
    client = QwableClient()
    mock_resp = _mock_response(
        {
            "presets": {"quality": {"panel": ["a", "b"]}},
            "loaded_now": [],
            "ds4_reachable": True,
        }
    )
    with patch("httpx.Client") as MockClient:
        MockClient.return_value.__enter__.return_value.get.return_value = mock_resp
        result = client.list_presets()
    assert "presets" in result
    assert result["presets"]["quality"]["panel"] == ["a", "b"]


def test_fusion_chat_non_streaming_builds_correct_body():
    """fusion_chat() sends correct payload shape."""
    client = QwableClient()
    mock_resp = _mock_response(
        {
            "choices": [
                {"message": {"content": "use mergesort"}, "finish_reason": "stop"}
            ]
        }
    )
    with patch("httpx.Client") as MockClient:
        ctx = MockClient.return_value.__enter__.return_value
        ctx.post.return_value = mock_resp
        result = client.fusion_chat(
            messages=[{"role": "user", "content": "hi"}],
            preset=FusionPreset.BUDGET,
        )
    # Verify request body
    call_args = ctx.post.call_args
    sent_body = call_args.kwargs["json"]
    assert sent_body["model"] == "qwable-fusion"
    assert sent_body["fusion"]["preset"] == "budget"
    assert sent_body["stream"] is False
    assert sent_body["max_tokens"] == 4000
    # Verify response parsed
    assert result.text == "use mergesort"


def test_fusion_chat_with_custom_panel():
    """Custom preset requires analysis_models + judge_model."""
    client = QwableClient()
    mock_resp = _mock_response(
        {"choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}]}
    )
    with patch("httpx.Client") as MockClient:
        ctx = MockClient.return_value.__enter__.return_value
        ctx.post.return_value = mock_resp
        client.fusion_chat(
            messages=[{"role": "user", "content": "x"}],
            preset=FusionPreset.CUSTOM,
            analysis_models=["m1", "m2"],
            judge_model="m-judge",
        )
    sent_body = ctx.post.call_args.kwargs["json"]
    assert sent_body["fusion"]["preset"] == "custom"
    assert sent_body["fusion"]["analysis_models"] == ["m1", "m2"]
    assert sent_body["fusion"]["judge_model"] == "m-judge"


def test_fusion_chat_string_preset_accepted():
    """preset can be a string in addition to FusionPreset enum."""
    client = QwableClient()
    mock_resp = _mock_response({"choices": [{"message": {"content": "ok"}}]})
    with patch("httpx.Client") as MockClient:
        ctx = MockClient.return_value.__enter__.return_value
        ctx.post.return_value = mock_resp
        client.fusion_chat(
            messages=[{"role": "user", "content": "x"}],
            preset="coding",  # string instead of enum
        )
    sent_body = ctx.post.call_args.kwargs["json"]
    assert sent_body["fusion"]["preset"] == "coding"


def test_fusion_chat_stream_yields_judge_tokens():
    """Stream parses SSE data lines into FusionEvent with judge.delta."""
    client = QwableClient()
    sse_lines = [
        'data: {"choices":[{"delta":{"content":"Hello "},"finish_reason":null}]}',
        'data: {"choices":[{"delta":{"content":"world"},"finish_reason":null}]}',
        'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}',
        "data: [DONE]",
    ]
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.raise_for_status = MagicMock()
    mock_resp.iter_lines.return_value = sse_lines
    mock_resp.__enter__ = lambda self: mock_resp
    mock_resp.__exit__ = lambda self, *args: None

    with patch("httpx.Client") as MockClient:
        ctx = MockClient.return_value.__enter__.return_value
        ctx.stream.return_value = mock_resp
        events = list(
            client.fusion_chat_stream(
                messages=[{"role": "user", "content": "x"}],
                preset=FusionPreset.BUDGET,
            )
        )
    # 2 judge_token + 1 judge_done (DONE doesn't yield an event)
    assert len(events) == 3
    assert events[0].event == "judge_token"
    assert events[0].judge.delta == "Hello "
    assert events[1].event == "judge_token"
    assert events[1].judge.delta == "world"
    assert events[2].event == "judge_done"


def test_fusion_chat_stream_skips_sse_comments():
    """Lines starting with ':' (fusion panel events) are ignored by SDK."""
    client = QwableClient()
    sse_lines = [
        'data: {"choices":[{"delta":{"content":"real token"}}]}',
        'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}',
        "data: [DONE]",
    ]
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.raise_for_status = MagicMock()
    mock_resp.iter_lines.return_value = sse_lines
    mock_resp.__enter__ = lambda self: mock_resp
    mock_resp.__exit__ = lambda self, *args: None

    with patch("httpx.Client") as MockClient:
        ctx = MockClient.return_value.__enter__.return_value
        ctx.stream.return_value = mock_resp
        events = list(
            client.fusion_chat_stream(
                messages=[{"role": "user", "content": "x"}],
            )
        )
    # Should yield 2 (judge_token + judge_done), no SSE comments
    assert len(events) == 2
    assert events[0].event == "judge_token"
    assert events[1].event == "judge_done"


def test_fusion_chat_raises_on_http_error():
    """4xx/5xx responses raise HTTPStatusError."""
    client = QwableClient()
    mock_resp = _mock_response({"error": "bad preset"}, status_code=400)
    with patch("httpx.Client") as MockClient:
        ctx = MockClient.return_value.__enter__.return_value
        ctx.post.return_value = mock_resp
        with pytest.raises(httpx.HTTPStatusError):
            client.fusion_chat(
                messages=[{"role": "user", "content": "x"}],
                preset="bogus",
            )


@pytest.mark.asyncio
async def test_afusion_chat_non_streaming():
    """Async non-streaming variant."""
    client = QwableClient()

    # Build an async context manager that yields our mock response
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "choices": [{"message": {"content": "async result"}}]
    }
    mock_resp.raise_for_status = MagicMock()

    # AsyncClient.post returns a coroutine directly (no context manager for post)
    with patch("httpx.AsyncClient.post", new=AsyncMock(return_value=mock_resp)):
        result = await client.afusion_chat(
            messages=[{"role": "user", "content": "x"}],
            preset=FusionPreset.QUALITY,
        )
    assert result.text == "async result"

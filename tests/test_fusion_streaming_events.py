"""G11: Tests for FusionStreamEvent dataclass + event ordering."""

import pytest

from qwable.streaming_events import (
    FUSION_STREAM_EVENT_PANEL_START,
    FUSION_STREAM_EVENT_PANEL_DONE,
    FUSION_STREAM_EVENT_JUDGE_START,
    FUSION_STREAM_EVENT_JUDGE_TOKEN,
    FUSION_STREAM_EVENT_JUDGE_DONE,
    FUSION_STREAM_EVENT_FINAL,
    FUSION_STREAM_EVENT_ERROR,
    FUSION_STREAM_EVENT_TYPES,
    FusionStreamEvent,
    format_fusion_sse,
)


def test_event_type_constants_present():
    """Fusion baseline and agent stream event types should be defined."""
    from qwable.streaming_events import AGENT_STREAM_EVENT_TYPES

    expected = {
        "panel_start", "panel_done",
        "judge_start", "judge_token", "judge_done",
        "final", "error",
    }
    assert expected.issubset(set(FUSION_STREAM_EVENT_TYPES))
    assert "agent_run_created" in AGENT_STREAM_EVENT_TYPES
    assert "agent_completed" in AGENT_STREAM_EVENT_TYPES
    assert "agent_run_created" in FUSION_STREAM_EVENT_TYPES


def test_fusion_stream_event_construct():
    """FusionStreamEvent should hold event name + data dict."""
    ev = FusionStreamEvent(event="panel_start", data={"model_id": "m1", "index": 0})
    assert ev.event == "panel_start"
    assert ev.data == {"model_id": "m1", "index": 0}


def test_format_fusion_sse_basic():
    """format_fusion_sse should produce valid SSE format: 'event: ...\\ndata: ...\\n\\n'."""
    ev = FusionStreamEvent(event="panel_start", data={"model_id": "m1"})
    sse = format_fusion_sse(ev)
    assert sse.startswith("event: panel_start\n")
    assert "data: " in sse
    assert sse.endswith("\n\n")


def test_format_fusion_sse_json_payload():
    """data payload must be valid JSON."""
    import json
    ev = FusionStreamEvent(event="judge_token", data={"delta": "hello"})
    sse = format_fusion_sse(ev)
    # Extract the data: line
    data_line = [ln for ln in sse.split("\n") if ln.startswith("data: ")][0]
    payload = json.loads(data_line[len("data: "):])
    assert payload == {"delta": "hello"}


def test_format_fusion_sse_unicode_safe():
    """Unicode (Traditional Chinese) content must round-trip."""
    ev = FusionStreamEvent(event="judge_token", data={"delta": "繁體中文測試"})
    sse = format_fusion_sse(ev)
    assert "繁體中文測試" in sse


def test_format_fusion_sse_final_event_shape():
    """final event should carry the full structured synthesis."""
    ev = FusionStreamEvent(
        event="final",
        data={
            "text": "## Final Answer\n\nUse mergesort.\n",
            "structured": {
                "consensus": ["point 1"],
                "contradictions": ["none"],
                "blind_spots": [],
                "per_model_notes": {"m1": "note"},
            },
            "trace": {"preset": "quality"},
        },
    )
    sse = format_fusion_sse(ev)
    assert "event: final" in sse
    assert "Use mergesort" in sse
    assert "preset" in sse


def test_format_fusion_sse_error_event():
    """error event should carry error detail."""
    ev = FusionStreamEvent(event="error", data={"detail": "model crashed", "code": 500})
    sse = format_fusion_sse(ev)
    assert "event: error" in sse
    assert "model crashed" in sse


def test_format_fusion_sse_agent_event():
    """Agent events should use the existing SSE formatter without breaking fusion events."""
    import json

    ev = FusionStreamEvent(
        event="agent_tool_requested",
        data={
            "run_id": "run_123",
            "message": "Tool requested",
            "metadata": {"tool_name": "read_file"},
            "created_at": "2026-06-22T00:00:00+00:00",
        },
    )

    sse = format_fusion_sse(ev)

    assert sse.startswith("event: agent_tool_requested\n")
    data_line = [ln for ln in sse.split("\n") if ln.startswith("data: ")][0]
    payload = json.loads(data_line[len("data: "):])
    assert payload["run_id"] == "run_123"
    assert payload["metadata"] == {"tool_name": "read_file"}


def test_format_fusion_sse_still_rejects_unknown_event():
    ev = FusionStreamEvent(event="not_real", data={})

    with pytest.raises(ValueError, match="unknown fusion stream event type"):
        format_fusion_sse(ev)


# ─── Event ordering (deferred — full streaming runner test) ───────────
# Ordering of events end-to-end is tested in test_fusion_streaming_runner.py
# once the runner is implemented (C3).

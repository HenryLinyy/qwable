"""G12-1: tests for panel token-by-token streaming.

When panel_token_streaming=True (default), each panel model's analysis is
streamed token-by-token via FusionStreamEvent(panel_token, {delta}).
"""

from unittest.mock import MagicMock

import pytest

from qwable.fusion_deliberation import _run_panel_serial_streaming
from qwable.fusion_presets import FusionPreset
from qwable.streaming_events import (
    FUSION_STREAM_EVENT_PANEL_DONE,
    FUSION_STREAM_EVENT_PANEL_START,
    FUSION_STREAM_EVENT_PANEL_TOKEN,
)


def _streaming_chat_client(chunks_by_model: dict[str, list[str]]):
    """Mock that streams chunks for each model.

    chunks_by_model: {model_id: [chunk1, chunk2, ...]}
    Returns a panel client whose chat_completion_stream yields those chunks.
    """
    pc = MagicMock()
    unload_log = []

    def stream(**kw):
        model = kw.get("model")
        for chunk in chunks_by_model.get(model, ["panel ok"]):
            yield (chunk, None)
        yield ("", "stop")

    pc.chat_completion_stream = stream
    pc.chat_completion = lambda **kw: {
        "choices": [
            {
                "message": {
                    "content": "".join(chunks_by_model.get(kw.get("model"), ["ok"]))
                },
                "finish_reason": "stop",
            }
        ]
    }
    pc.unload_models = lambda models, **kw: unload_log.append(list(models))
    return pc, unload_log


@pytest.mark.asyncio
async def test_panel_tokens_yielded_per_chunk():
    """Each chat_completion_stream chunk becomes a panel_token event."""
    chunks = {
        "google/gemma-4-26b-a4b-qat": ["## ", "Analysis", "\nFor 50k"],
        "qwen/qwen3.6-35b-a3b": ["## ", "Analysis", "\nUse mergesort"],
    }
    client, _ = _streaming_chat_client(chunks)

    preset = FusionPreset(
        name="budget",
        analysis_models=("google/gemma-4-26b-a4b-qat", "qwen/qwen3.6-35b-a3b"),
        judge_model="qwen/qwen3.6-35b-a3b",
        description="test",
    )
    events = []
    async for ev in _run_panel_serial_streaming(
        preset=preset,
        original_prompt="x",
        panel_client=client,
        panel_max_tokens=500,
        temperature=0.3,
    ):
        if hasattr(ev, "event"):
            events.append(ev)

    panel_token_events = [
        e for e in events if e.event == FUSION_STREAM_EVENT_PANEL_TOKEN
    ]
    # gemma: 3 chunks, qwen3.6: 3 chunks = 6 total
    assert len(panel_token_events) == 6
    # Verify deltas preserved
    deltas = [e.data["delta"] for e in panel_token_events]
    assert deltas == [
        "## ",
        "Analysis",
        "\nFor 50k",
        "## ",
        "Analysis",
        "\nUse mergesort",
    ]


@pytest.mark.asyncio
async def test_panel_token_events_have_model_id_and_index():
    """Each panel_token event should carry model_id and index."""
    chunks = {
        "google/gemma-4-26b-a4b-qat": ["a", "b"],
        "qwen/qwen3.6-35b-a3b": ["c", "d"],
    }
    client, _ = _streaming_chat_client(chunks)

    preset = FusionPreset(
        name="budget",
        analysis_models=("google/gemma-4-26b-a4b-qat", "qwen/qwen3.6-35b-a3b"),
        judge_model="qwen/qwen3.6-35b-a3b",
        description="test",
    )
    events = []
    async for ev in _run_panel_serial_streaming(
        preset=preset,
        original_prompt="x",
        panel_client=client,
        panel_max_tokens=500,
        temperature=0.3,
    ):
        if hasattr(ev, "event") and ev.event == FUSION_STREAM_EVENT_PANEL_TOKEN:
            events.append(ev)

    # First 2 from gemma (index 0), next 2 from qwen3.6 (index 1)
    assert events[0].data["model_id"] == "google/gemma-4-26b-a4b-qat"
    assert events[0].data["index"] == 0
    assert events[0].data["delta"] == "a"
    assert events[1].data["delta"] == "b"
    assert events[2].data["model_id"] == "qwen/qwen3.6-35b-a3b"
    assert events[2].data["index"] == 1
    assert events[2].data["delta"] == "c"
    assert events[3].data["delta"] == "d"


@pytest.mark.asyncio
async def test_panel_event_order_panel_start_token_done():
    """Event order per panel: panel_start → panel_token*N → panel_done."""
    chunks = {"google/gemma-4-26b-a4b-qat": ["a", "b", "c"]}
    client, _ = _streaming_chat_client(chunks)
    preset = FusionPreset(
        name="test",
        analysis_models=("google/gemma-4-26b-a4b-qat",),
        judge_model="qwen/qwen3.6-35b-a3b",
        description="test",
    )
    events = []
    async for ev in _run_panel_serial_streaming(
        preset=preset,
        original_prompt="x",
        panel_client=client,
        panel_max_tokens=500,
        temperature=0.3,
    ):
        if hasattr(ev, "event"):
            events.append(ev.event)

    # Expected: panel_start, panel_token×3, panel_done
    assert events == [
        FUSION_STREAM_EVENT_PANEL_START,
        FUSION_STREAM_EVENT_PANEL_TOKEN,
        FUSION_STREAM_EVENT_PANEL_TOKEN,
        FUSION_STREAM_EVENT_PANEL_TOKEN,
        FUSION_STREAM_EVENT_PANEL_DONE,
    ]

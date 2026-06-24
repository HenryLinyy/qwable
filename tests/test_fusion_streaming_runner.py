"""G11: Tests for run_fusion_agent_streaming async generator.

Verifies event ordering: panel_start → panel_done × N → judge_start →
judge_token × M → judge_done → final.
"""

import asyncio
from unittest.mock import MagicMock

import pytest

from qwable.fusion_deliberation import run_fusion_agent_streaming
from qwable.fusion_presets import PRESETS
from qwable.fusion_schemas import PanelResponse
from qwable.streaming_events import (
    FUSION_STREAM_EVENT_FINAL,
    FUSION_STREAM_EVENT_JUDGE_DONE,
    FUSION_STREAM_EVENT_JUDGE_START,
    FUSION_STREAM_EVENT_JUDGE_TOKEN,
    FUSION_STREAM_EVENT_PANEL_DONE,
    FUSION_STREAM_EVENT_PANEL_START,
)


def _structured_judge_text() -> str:
    return """\
## Final Answer
Use mergesort.

## Consensus
- Stability

## Contradictions
- none

## Blind Spots
- none

## Per-model Notes
### google/gemma-4-26b-a4b-qat
notes
"""


class _StubPanelClient:
    """Stand-in for OllamaClient; returns canned panel responses."""

    def __init__(self):
        self.unload_log: list = []
        self.call_log: list = []

    def chat_completion(self, *, model, messages, **kwargs):
        self.call_log.append(model)
        return {
            "choices": [
                {"message": {"content": f"## Analysis\npanel from {model}"}, "finish_reason": "stop"}
            ]
        }

    def unload_models(self, models=None, **kwargs):
        self.unload_log.append(list(models or []))

    def chat_completion_stream(self, *, model, messages, **kwargs):
        """Yield judge tokens that build into the structured text."""
        # Yield chunks that together reconstruct _structured_judge_text()
        full = _structured_judge_text()
        # Send in 30-char chunks for testability
        chunk_size = 30
        for i in range(0, len(full), chunk_size):
            yield (full[i:i + chunk_size], None)
        yield ("", "stop")


class _StubDS4Client:
    def __init__(self):
        self.call_log: list = []

    def chat_completion_stream(self, *, model, messages, **kwargs):
        self.call_log.append(model)
        full = _structured_judge_text()
        yield (full, "stop")

    def chat_completion(self, *, model, messages, **kwargs):
        self.call_log.append(model)
        return {
            "choices": [
                {"message": {"content": _structured_judge_text()}, "finish_reason": "stop"}
            ]
        }


# ─── Order test ──────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_streaming_event_order_for_budget_preset():
    """budget preset: 2 panel events → judge_start → judge_tokens → judge_done → final."""
    panel = _StubPanelClient()
    ds4 = _StubDS4Client()

    events = []
    async for ev in run_fusion_agent_streaming(
        ollama_client=panel,
        ds4_client=ds4,
        preset=PRESETS["budget"],
        original_prompt="Compare two sort algorithms",
        panel_max_tokens=500,
        judge_max_tokens=1500,
        ds4_model="deepseek-v4-flash",
    ):
        events.append(ev)

    types = [e.event for e in events]
    # Expected: panel_start × 2, panel_done × 2, judge_start × 1,
    #           judge_token × N (≥1), judge_done × 1, final × 1
    # (G12-1: panel_token events may also appear between start and done)
    assert types[0] == FUSION_STREAM_EVENT_PANEL_START
    # First panel: start, optional panel_token*N, then done
    done_indices = [i for i, t in enumerate(types) if t == FUSION_STREAM_EVENT_PANEL_DONE]
    assert len(done_indices) == 2
    assert types[done_indices[0]] == FUSION_STREAM_EVENT_PANEL_DONE
    assert types[done_indices[1]] == FUSION_STREAM_EVENT_PANEL_DONE

    # Find judge_start position (must come after both panel_done)
    judge_start_idx = types.index(FUSION_STREAM_EVENT_JUDGE_START)
    assert judge_start_idx > done_indices[1]

    # Between judge_start and judge_done there must be at least one judge_token
    judge_done_idx = types.index(FUSION_STREAM_EVENT_JUDGE_DONE)
    token_count = types[judge_start_idx + 1:judge_done_idx].count(FUSION_STREAM_EVENT_JUDGE_TOKEN)
    assert token_count >= 1

    # Final must be last
    assert types[-1] == FUSION_STREAM_EVENT_FINAL


@pytest.mark.asyncio
async def test_streaming_panel_unloads_after_each_model():
    """Each panel model must be unloaded between calls (memory invariant)."""
    panel = _StubPanelClient()
    ds4 = _StubDS4Client()

    async for _ in run_fusion_agent_streaming(
        ollama_client=panel,
        ds4_client=ds4,
        preset=PRESETS["budget"],
        original_prompt="x",
        panel_max_tokens=500,
        judge_max_tokens=1500,
        ds4_model="deepseek-v4-flash",
    ):
        pass

    # G12-3: budget preset → only gemma unloaded (last panel + judge kept resident)
    assert len(panel.unload_log) == 1
    assert "google/gemma-4-26b-a4b-qat" in panel.unload_log[0]  # gemma unloaded
    # qwen3.6 (last panel + judge) NOT in unload_log


@pytest.mark.asyncio
async def test_streaming_judge_uses_ds4_backend_for_heavy_preset():
    """heavy preset (judge=ds4) should use ds4_client.chat_completion_stream."""
    panel = _StubPanelClient()
    ds4 = _StubDS4Client()

    events = []
    async for ev in run_fusion_agent_streaming(
        ollama_client=panel,
        ds4_client=ds4,
        preset=PRESETS["heavy"],
        original_prompt="x",
        panel_max_tokens=500,
        judge_max_tokens=1500,
        ds4_model="deepseek-v4-flash",
    ):
        events.append(ev)

    # Verify ds4 was called for judge (not ollama panel)
    assert len(ds4.call_log) == 1
    assert "deepseek-v4-flash" in ds4.call_log

    # Final event should record judge_backend=ds4
    final = events[-1]
    assert final.event == FUSION_STREAM_EVENT_FINAL
    assert final.data["trace"]["judge_backend"] == "ds4"


@pytest.mark.asyncio
async def test_streaming_final_event_has_structured_output():
    """final event should include structured output (final_answer, sections)."""
    panel = _StubPanelClient()
    ds4 = _StubDS4Client()

    events = []
    async for ev in run_fusion_agent_streaming(
        ollama_client=panel,
        ds4_client=ds4,
        preset=PRESETS["budget"],
        original_prompt="x",
        panel_max_tokens=500,
        judge_max_tokens=1500,
        ds4_model="deepseek-v4-flash",
    ):
        events.append(ev)

    final = events[-1]
    assert final.event == FUSION_STREAM_EVENT_FINAL
    assert "Use mergesort" in final.data["text"]
    assert final.data["structured"]["consensus"] == ["Stability"]
    assert final.data["structured"]["had_fallback"] is False
    assert final.data["trace"]["preset"] == "budget"
    assert final.data["trace"]["total_latency_ms"] >= 0


@pytest.mark.asyncio
async def test_streaming_judge_tokens_yielded_in_order():
    """judge_token deltas, when concatenated, should reconstruct judge text."""
    panel = _StubPanelClient()
    ds4 = _StubDS4Client()

    deltas = []
    async for ev in run_fusion_agent_streaming(
        ollama_client=panel,
        ds4_client=ds4,
        preset=PRESETS["budget"],
        original_prompt="x",
        panel_max_tokens=500,
        judge_max_tokens=1500,
        ds4_model="deepseek-v4-flash",
    ):
        if ev.event == FUSION_STREAM_EVENT_JUDGE_TOKEN:
            deltas.append(ev.data["delta"])

    # Concatenate all deltas
    full = "".join(deltas)
    assert "## Final Answer" in full
    assert "Use mergesort" in full
    assert "## Consensus" in full
    assert "## Per-model Notes" in full

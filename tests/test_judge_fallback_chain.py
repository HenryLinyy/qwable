"""G12-5: tests for judge fallback chain in fusion deliberation.

When the primary judge model fails (chat_completion raises), the runner
should try each fallback model in order. Only after all fallbacks fail does
the runner raise an exception.

Note: the existing `primary judge` is whatever `preset.judge_model` says.
The fallback chain is appended AFTER the primary judge.
"""

from unittest.mock import MagicMock

import pytest

from qwable.fusion_deliberation import run_fusion_agent
from qwable.fusion_presets import PRESETS


def _structured_judge_text() -> str:
    return """\
## Final Answer
Use mergesort for stability.

## Consensus
- Stability matters

## Contradictions
- none

## Blind Spots
- none

## Per-model Notes
### google/gemma-4-26b-a4b-qat
notes
"""


def _ok_response(text: str = None) -> dict:
    return {
        "choices": [
            {
                "message": {"content": text or _structured_judge_text()},
                "finish_reason": "stop",
            }
        ]
    }


def _make_panel_client(panel_text: str = "panel ok") -> MagicMock:
    pc = MagicMock()
    pc.chat_completion = lambda **kw: _ok_response(panel_text)
    pc.unload_models = lambda models, **kw: None
    pc.chat_completion_stream = lambda **kw: iter([(_structured_judge_text(), "stop")])
    return pc


def _make_ds4_ok() -> MagicMock:
    ds4 = MagicMock()
    ds4.chat_completion = lambda **kw: _ok_response()
    ds4.chat_completion_stream = lambda **kw: iter([(_structured_judge_text(), "stop")])
    return ds4


def _failing_client(model_id_to_fail: str) -> MagicMock:
    """Mock that fails when chat_completion is called with model_id_to_fail."""
    pc = MagicMock()

    def chat(**kw):
        if kw.get("model") == model_id_to_fail:
            raise RuntimeError(f"model {model_id_to_fail} crashed")
        return _ok_response()

    pc.chat_completion = chat
    pc.unload_models = lambda models, **kw: None
    pc.chat_completion_stream = lambda **kw: iter([(_structured_judge_text(), "stop")])
    return pc


@pytest.mark.asyncio
async def test_judge_fallback_uses_primary_when_ok():
    """Primary judge (qwen3.6) succeeds → no fallback needed."""
    panel = _make_panel_client()
    call_log = []

    def tracking_chat(**kw):
        call_log.append(kw.get("model"))
        return _ok_response()

    panel.chat_completion = tracking_chat

    # budget preset judge = qwen3.6 (in primary position)
    await run_fusion_agent(
        ollama_client=panel,
        ds4_client=_make_ds4_ok(),
        preset=PRESETS["budget"],
        original_prompt="x",
        panel_max_tokens=500,
        judge_max_tokens=1500,
        ds4_model="deepseek-v4-flash",
        keep_last_resident=True,
        judge_fallback_chain=[
            "google/gemma-4-26b-a4b-qat",
            "deepseek-r1-distill-qwen-32b",
        ],
    )
    # Judge = qwen3.6 — should be tried first and succeed
    judge_calls = [
        m for m in call_log if m != "google/gemma-4-26b-a4b-qat"
    ]  # exclude panel
    assert "qwen/qwen3.6-35b-a3b" in judge_calls
    # Fallbacks should NOT be called (primary succeeded)
    assert "deepseek-r1-distill-qwen-32b" not in call_log


@pytest.mark.asyncio
async def test_judge_fallback_tries_next_on_failure():
    """When primary judge fails, try first fallback (gemma)."""
    panel = MagicMock()
    call_log = []

    def chat(**kw):
        model = kw.get("model")
        call_log.append(model)
        if model == "qwen/qwen3.6-35b-a3b":  # primary judge
            raise RuntimeError("primary judge crashed")
        return _ok_response()

    panel.chat_completion = chat
    panel.unload_models = lambda models, **kw: None
    panel.chat_completion_stream = lambda **kw: iter(
        [(_structured_judge_text(), "stop")]
    )

    await run_fusion_agent(
        ollama_client=panel,
        ds4_client=_make_ds4_ok(),
        preset=PRESETS["budget"],  # primary judge = qwen3.6
        original_prompt="x",
        panel_max_tokens=500,
        judge_max_tokens=1500,
        ds4_model="deepseek-v4-flash",
        keep_last_resident=True,
        judge_fallback_chain=[
            "google/gemma-4-26b-a4b-qat",
            "deepseek-r1-distill-qwen-32b",
        ],
    )
    # Both qwen3.6 and gemma should appear in call log (primary failed → fallback)
    assert "qwen/qwen3.6-35b-a3b" in call_log
    assert "google/gemma-4-26b-a4b-qat" in call_log


@pytest.mark.asyncio
async def test_judge_fallback_chain_includes_ds4_first():
    """When preset judge IS ds4 (heavy preset), no fallback needed for backend."""
    panel = _make_panel_client()
    ds4 = _make_ds4_ok()
    call_log = []

    orig_ds4_chat = ds4.chat_completion

    def tracking(**kw):
        call_log.append(("ds4", kw.get("model")))
        return orig_ds4_chat(**kw)

    ds4.chat_completion = tracking

    # heavy preset judge = deepseek-v4-flash (ds4)
    await run_fusion_agent(
        ollama_client=panel,
        ds4_client=ds4,
        preset=PRESETS["heavy"],
        original_prompt="x",
        panel_max_tokens=500,
        judge_max_tokens=1500,
        ds4_model="deepseek-v4-flash",
        keep_last_resident=True,
    )
    # Only 1 ds4 call (no fallback to ollama)
    ds4_calls = [c for c in call_log if c[0] == "ds4"]
    assert len(ds4_calls) == 1


@pytest.mark.asyncio
async def test_judge_fallback_chains_through_multiple_fallbacks():
    """When primary + first fallback fail, try second fallback (r1)."""
    panel = MagicMock()
    call_log = []

    def chat(**kw):
        model = kw.get("model")
        call_log.append(model)
        if model in ("qwen/qwen3.6-35b-a3b", "google/gemma-4-26b-a4b-qat"):
            raise RuntimeError(f"{model} crashed")
        return _ok_response()

    panel.chat_completion = chat
    panel.unload_models = lambda models, **kw: None
    panel.chat_completion_stream = lambda **kw: iter(
        [(_structured_judge_text(), "stop")]
    )

    await run_fusion_agent(
        ollama_client=panel,
        ds4_client=_make_ds4_ok(),
        preset=PRESETS["budget"],
        original_prompt="x",
        panel_max_tokens=500,
        judge_max_tokens=1500,
        ds4_model="deepseek-v4-flash",
        keep_last_resident=True,
        judge_fallback_chain=[
            "google/gemma-4-26b-a4b-qat",
            "deepseek-r1-distill-qwen-32b",
        ],
    )
    # All 3 judges should appear (qwen3.6, gemma, r1)
    assert "qwen/qwen3.6-35b-a3b" in call_log
    assert "google/gemma-4-26b-a4b-qat" in call_log
    assert "deepseek-r1-distill-qwen-32b" in call_log

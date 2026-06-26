"""G10: Tests for serial deliberation runner.

Mock OllamaClient and DS4Client; verify the runner:
- calls chat_completion once per analysis_model + once for judge
- calls unload_models between panel calls
- picks ds4 backend when judge_model == ds4_model
- picks ollama backend otherwise
- captures latency per call
- wraps panel errors in PanelResponse.error
- serializes (panel N completes before panel N+1 starts)
"""

import time
from unittest.mock import MagicMock

import pytest

from qwable.fusion_deliberation import (
    run_panel_serial,
    run_fusion_agent,
)
from qwable.fusion_presets import PRESETS


def _make_panel_client(
    responses_per_model: dict[str, str], call_log: list = None, unload_log: list = None
) -> MagicMock:
    """Mock OllamaClient.

    `responses_per_model`: {model_id: response_text} returns text for chat_completion.
    `call_log`: list of model_ids appended on each chat call (for order checks).
    `unload_log`: list of model_id lists appended on each unload call.
    """
    client = MagicMock()

    def chat(model, messages, **kwargs):
        if call_log is not None:
            call_log.append(model)
        text = responses_per_model.get(model, f"default response from {model}")
        return {
            "choices": [
                {
                    "message": {"role": "assistant", "content": text},
                    "finish_reason": "stop",
                }
            ]
        }

    def unload(models, **kwargs):
        if unload_log is not None:
            unload_log.append(list(models))

    client.chat_completion = chat
    client.unload_models = unload
    return client


def _make_ds4_client(
    response_text: str = "ds4 final answer", call_log: list = None
) -> MagicMock:
    client = MagicMock()

    def chat(model, messages, **kwargs):
        if call_log is not None:
            call_log.append(("ds4", model))
        return {
            "choices": [
                {
                    "message": {"role": "assistant", "content": response_text},
                    "finish_reason": "stop",
                }
            ]
        }

    client.chat_completion = chat
    return client


# ─── run_panel_serial ─────────────────────────────────────────────────────


def test_run_panel_serial_calls_each_analysis_model_once():
    preset = PRESETS["budget"]  # 2 analysis models
    client = _make_panel_client(
        {
            "google/gemma-4-26b-a4b-qat": "gemma analysis",
            "qwen/qwen3.6-35b-a3b": "qwen3.6 analysis",
        }
    )
    responses = run_panel_serial(
        preset=preset,
        original_prompt="x",
        system_prompt="SYSTEM",
        panel_client=client,
        panel_max_tokens=500,
        temperature=0.3,
    )
    assert len(responses) == 2
    assert responses[0].model_id == "google/gemma-4-26b-a4b-qat"
    assert responses[0].text == "gemma analysis"
    assert responses[1].model_id == "qwen/qwen3.6-35b-a3b"
    assert responses[1].text == "qwen3.6 analysis"


def test_run_panel_serial_calls_unload_after_each_model():
    preset = PRESETS["budget"]
    call_log = []
    unload_log = []
    client = _make_panel_client({}, call_log=call_log, unload_log=unload_log)
    run_panel_serial(
        preset=preset,
        original_prompt="x",
        system_prompt="SYS",
        panel_client=client,
        panel_max_tokens=500,
        temperature=0.3,
    )
    # 2 panel models → 2 chat + 2 unload calls
    assert len(call_log) == 2
    assert len(unload_log) == 2
    assert unload_log[0] == ["google/gemma-4-26b-a4b-qat"]
    assert unload_log[1] == ["qwen/qwen3.6-35b-a3b"]


def test_run_panel_serial_serializes_models():
    """Each chat must complete BEFORE the next chat starts (serial, not parallel)."""
    preset = PRESETS["budget"]
    call_log = []

    def slow_chat(model, messages, **kwargs):
        call_log.append(("start", model))
        time.sleep(0.01)  # simulate latency
        call_log.append(("end", model))
        return {
            "choices": [
                {"message": {"content": f"out-{model}"}, "finish_reason": "stop"}
            ]
        }

    client = MagicMock()
    client.chat_completion = slow_chat
    client.unload_models = lambda models: None

    run_panel_serial(
        preset=preset,
        original_prompt="x",
        system_prompt="SYS",
        panel_client=client,
        panel_max_tokens=500,
        temperature=0.3,
    )
    # Expect: start m1, end m1, start m2, end m2 — NOT start m1, start m2, end m1, end m2
    assert call_log == [
        ("start", "google/gemma-4-26b-a4b-qat"),
        ("end", "google/gemma-4-26b-a4b-qat"),
        ("start", "qwen/qwen3.6-35b-a3b"),
        ("end", "qwen/qwen3.6-35b-a3b"),
    ]


def test_run_panel_serial_captures_latency_per_model():
    preset = PRESETS["budget"]
    client = _make_panel_client({})
    responses = run_panel_serial(
        preset=preset,
        original_prompt="x",
        system_prompt="SYS",
        panel_client=client,
        panel_max_tokens=500,
        temperature=0.3,
    )
    for r in responses:
        assert r.latency_ms >= 0


def test_run_panel_serial_captures_panel_error():
    """If chat_completion raises, capture error in PanelResponse and continue."""
    client = MagicMock()

    def flaky_chat(model, messages, **kwargs):
        if "bad" in model:
            raise RuntimeError("model crashed")
        return {
            "choices": [
                {"message": {"content": f"good-{model}"}, "finish_reason": "stop"}
            ]
        }

    client.chat_completion = flaky_chat
    client.unload_models = lambda models: None

    from qwable.fusion_presets import FusionPreset

    preset = FusionPreset(
        name="test",
        analysis_models=("good-model", "bad-model"),
        judge_model="qwen/qwen3.6-35b-a3b",
        description="test preset",
    )

    responses = run_panel_serial(
        preset=preset,
        original_prompt="x",
        system_prompt="SYS",
        panel_client=client,
        panel_max_tokens=500,
        temperature=0.3,
    )
    assert len(responses) == 2
    assert responses[0].error is None
    assert responses[0].text == "good-good-model"
    assert responses[1].error is not None
    assert "model crashed" in responses[1].error  # includes exception detail
    assert responses[1].text == ""
    assert responses[1].finish_reason == "error"


def test_run_panel_serial_unloads_even_after_panel_error():
    """Unload must still happen after a panel error."""
    client = MagicMock()
    client.chat_completion = MagicMock(side_effect=RuntimeError("boom"))
    unload_log = []
    client.unload_models = lambda models: unload_log.append(list(models))

    from qwable.fusion_presets import FusionPreset

    preset = FusionPreset(
        name="test",
        analysis_models=("m1", "m2"),
        judge_model="m-judge",
        description="test",
    )
    run_panel_serial(
        preset=preset,
        original_prompt="x",
        system_prompt="SYS",
        panel_client=client,
        panel_max_tokens=500,
        temperature=0.3,
    )
    assert unload_log == [["m1"], ["m2"]]


def test_run_panel_serial_continues_after_unload_failure():
    """If unload fails, runner should not crash; subsequent models still run."""
    call_log = []
    client = MagicMock()
    client.chat_completion = lambda model, messages, **kwargs: (
        call_log.append(model),
        {
            "choices": [
                {"message": {"content": f"out-{model}"}, "finish_reason": "stop"}
            ]
        },
    )[1]

    def failing_unload(models):
        raise RuntimeError("unload failed")

    client.unload_models = failing_unload

    from qwable.fusion_presets import FusionPreset

    preset = FusionPreset(
        name="test",
        analysis_models=("m1", "m2"),
        judge_model="m-judge",
        description="test",
    )
    responses = run_panel_serial(
        preset=preset,
        original_prompt="x",
        system_prompt="SYS",
        panel_client=client,
        panel_max_tokens=500,
        temperature=0.3,
    )
    # Both models should still produce responses
    assert len(responses) == 2
    assert call_log == ["m1", "m2"]


# ─── run_fusion_agent (top-level) ─────────────────────────────────────────


def _structured_judge_text() -> str:
    """A complete structured judge output for happy-path tests."""
    return """\
## Final Answer
Use mergesort for stability.

## Consensus
- Stability matters

## Contradictions
- Model A vs B

## Blind Spots
- Memory not analyzed

## Per-model Notes
### google/gemma-4-26b-a4b-qat
first notes

### qwen/qwen3.6-35b-a3b
second notes
"""


@pytest.mark.asyncio
async def test_run_fusion_agent_uses_ollama_judge_for_quality_preset():
    """Quality preset judge (qwen3.6) should use ollama backend."""
    panel_client = _make_panel_client(
        {
            "qwen/qwen3-coder-next": "## Analysis\ncoder analysis",
            "deepseek-r1-distill-qwen-32b": "## Analysis\nr1 analysis",
            # qwen3.6 is both a panelist and the quality-preset judge; this mock
            # keys by model id, so it returns the judge text for both its calls.
            "qwen/qwen3.6-35b-a3b": _structured_judge_text(),
        }
    )
    ds4_call_log = []
    ds4_client = _make_ds4_client(call_log=ds4_call_log)

    result = await run_fusion_agent(
        ollama_client=panel_client,
        ds4_client=ds4_client,
        preset=PRESETS["quality"],
        original_prompt="Compare sort algorithms",
        panel_max_tokens=500,
        judge_max_tokens=1500,
        ds4_model="deepseek-v4-flash",
    )
    assert result["trace"]["judge_backend"] == "ollama"
    assert ds4_call_log == []
    assert (
        "mergesort" in result["text"].lower() or "stability" in result["text"].lower()
    )


@pytest.mark.asyncio
async def test_run_fusion_agent_uses_ds4_judge_for_heavy_preset():
    """Heavy preset judge (deepseek-v4-flash) should use ds4 backend."""
    panel_client = _make_panel_client(
        {
            "qwen/qwen3-coder-next": "## Analysis\ncoder",
            "deepseek-r1-distill-qwen-32b": "## Analysis\nr1",
        }
    )
    ds4_client = _make_ds4_client(
        response_text=_structured_judge_text(),
    )

    result = await run_fusion_agent(
        ollama_client=panel_client,
        ds4_client=ds4_client,
        preset=PRESETS["heavy"],
        original_prompt="long context question",
        panel_max_tokens=500,
        judge_max_tokens=1500,
        ds4_model="deepseek-v4-flash",
    )
    assert result["trace"]["judge_backend"] == "ds4"


@pytest.mark.asyncio
async def test_run_fusion_agent_trace_includes_all_panel_responses():
    panel_client = _make_panel_client(
        {
            "google/gemma-4-26b-a4b-qat": "## Analysis\ngemma",
            "qwen/qwen3.6-35b-a3b": "## Analysis\nqwen",
        },
        call_log=[],
    )

    # Override judge call to return structured output

    def chat_with_judge(model, messages, **kwargs):
        if "## Final Answer" not in str(messages):
            return {
                "choices": [
                    {"message": {"content": f"panel-{model}"}, "finish_reason": "stop"}
                ]
            }
        return {
            "choices": [
                {
                    "message": {"content": _structured_judge_text()},
                    "finish_reason": "stop",
                }
            ]
        }

    panel_client.chat_completion = chat_with_judge

    result = await run_fusion_agent(
        ollama_client=panel_client,
        ds4_client=_make_ds4_client(),
        preset=PRESETS["budget"],
        original_prompt="x",
        panel_max_tokens=500,
        judge_max_tokens=1500,
        ds4_model="deepseek-v4-flash",
    )
    panel_ids = [r["model_id"] for r in result["trace"]["panel_responses"]]
    assert "google/gemma-4-26b-a4b-qat" in panel_ids
    assert "qwen/qwen3.6-35b-a3b" in panel_ids
    assert result["trace"]["judge_model"] == "qwen/qwen3.6-35b-a3b"
    assert result["trace"]["total_latency_ms"] >= 0


@pytest.mark.asyncio
async def test_run_fusion_agent_handles_unstructured_judge_output():
    """Judge that ignores the 5-section format → fallback used, had_fallback=True."""
    panel_client = MagicMock()

    def chat(model, messages, **kwargs):
        # First N calls are panel, then judge returns unstructured text
        return {
            "choices": [
                {
                    "message": {"content": "raw unstructured judge output"},
                    "finish_reason": "stop",
                }
            ]
        }

    panel_client.chat_completion = chat
    panel_client.unload_models = lambda models: None

    result = await run_fusion_agent(
        ollama_client=panel_client,
        ds4_client=_make_ds4_client(),
        preset=PRESETS["budget"],
        original_prompt="x",
        panel_max_tokens=500,
        judge_max_tokens=1500,
        ds4_model="deepseek-v4-flash",
    )
    assert result["trace"]["structured_had_fallback"] is True
    assert result["text"] == "raw unstructured judge output"


@pytest.mark.asyncio
async def test_run_fusion_agent_serial_panel_then_judge():
    """Verify trace shows N panel responses + 1 judge call in correct order."""
    call_log = []

    def chat(model, messages, **kwargs):
        # We can't easily distinguish panel vs judge from the mock side,
        # but we can verify all 3 chat calls happened (2 panel + 1 judge).
        call_log.append(model)
        # Return structured judge output on every call; runner picks the last
        # one as the judge. Panel responses are also structured (but irrelevant).
        return {
            "choices": [
                {
                    "message": {"content": _structured_judge_text()},
                    "finish_reason": "stop",
                }
            ]
        }

    panel_client = MagicMock()
    panel_client.chat_completion = chat
    panel_client.unload_models = lambda models: None

    result = await run_fusion_agent(
        ollama_client=panel_client,
        ds4_client=_make_ds4_client(),
        preset=PRESETS["budget"],  # 2 panel + 1 judge (gemma is both panel and judge)
        original_prompt="x",
        panel_max_tokens=500,
        judge_max_tokens=1500,
        ds4_model="deepseek-v4-flash",
    )

    # 3 chat calls: 2 panel + 1 judge
    assert len(call_log) == 3
    # First 2 are panel in preset order
    assert call_log[0] == "google/gemma-4-26b-a4b-qat"
    assert call_log[1] == "qwen/qwen3.6-35b-a3b"
    # Last is judge (budget preset's judge_model is qwen3.6)
    assert call_log[2] == "qwen/qwen3.6-35b-a3b"

    # Trace records 2 panel responses (the judge call does NOT appear as a panel response)
    assert len(result["trace"]["panel_responses"]) == 2
    # Judge info present
    assert result["trace"]["judge_model"] == "qwen/qwen3.6-35b-a3b"
    assert result["trace"]["judge_backend"] == "ollama"


@pytest.mark.asyncio
async def test_run_fusion_agent_unloads_judge_after_call():
    """After judge runs, judge model should also be unloaded for clean state."""
    unload_log = []

    panel_client = MagicMock()
    panel_client.chat_completion = lambda model, messages, **kwargs: (
        {
            "choices": [
                {
                    "message": {"content": _structured_judge_text()},
                    "finish_reason": "stop",
                }
            ]
        }
    )
    panel_client.unload_models = lambda models: unload_log.append(list(models))

    await run_fusion_agent(
        ollama_client=panel_client,
        ds4_client=_make_ds4_client(),
        preset=PRESETS["budget"],
        original_prompt="x",
        panel_max_tokens=500,
        judge_max_tokens=1500,
        ds4_model="deepseek-v4-flash",
    )
    # G12-3: budget preset → 2 panel models unloads, last panel kept resident;
    # judge (qwen3.6) is also kept resident by keep_last_resident=True default.
    # So only 1 panel unload (gemma), nothing for judge.
    assert len(unload_log) == 1
    assert "google/gemma-4-26b-a4b-qat" in unload_log[0]  # only gemma unloaded


@pytest.mark.asyncio
async def test_run_fusion_agent_panel_with_error_still_runs_judge():
    """If a panel model errors, runner continues and judge still runs."""
    panel_client = MagicMock()
    call_count = {"n": 0}

    def chat(model, messages, **kwargs):
        call_count["n"] += 1
        if call_count["n"] == 1:
            raise RuntimeError("panel-1 failed")
        if model == "qwen/qwen3.6-35b-a3b":
            return {
                "choices": [
                    {"message": {"content": "panel 2 ok"}, "finish_reason": "stop"}
                ]
            }
        # judge call
        return {
            "choices": [
                {
                    "message": {"content": _structured_judge_text()},
                    "finish_reason": "stop",
                }
            ]
        }

    panel_client.chat_completion = chat
    panel_client.unload_models = lambda models: None

    result = await run_fusion_agent(
        ollama_client=panel_client,
        ds4_client=_make_ds4_client(),
        preset=PRESETS["budget"],
        original_prompt="x",
        panel_max_tokens=500,
        judge_max_tokens=1500,
        ds4_model="deepseek-v4-flash",
    )
    # Trace should show one panel error
    panel_errors = [r for r in result["trace"]["panel_responses"] if r["error"]]
    assert len(panel_errors) >= 1
    # Judge still produced a final answer
    assert result["text"] is not None
    assert len(result["text"]) > 0

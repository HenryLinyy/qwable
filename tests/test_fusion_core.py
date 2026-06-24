"""G10: FusionCore dispatch + fusion-agent integration tests (mocked).

Uses a fake OllamaClient/DS4Client pair to exercise the full real runner
without touching real services.
"""

from unittest.mock import MagicMock

import pytest

from qwable.config import FusionConfig
from qwable.fusion_core import FusionCore
from qwable.models import DS4Client, OllamaClient
from qwable.schemas import FusionAction, ParsedAgentTask


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
### model-a
notes
"""


def _install_mock_clients(monkeypatch, judge_text: str = None) -> dict:
    """Replace OllamaClient/DS4Client classes with mocks that auto-respond.

    All chat_completion calls (panel + judge) return judge_text so trace
    assertions are deterministic regardless of which call we're inspecting.
    Returns a dict with call_log, unload_log for assertions.
    """
    if judge_text is None:
        judge_text = _structured_judge_text()

    state = {"call_log": [], "unload_log": []}

    def fake_chat(self_or_None=None, *, model, messages, **kwargs):
        state["call_log"].append(model)
        return {
            "choices": [
                {"message": {"role": "assistant", "content": judge_text},
                 "finish_reason": "stop"}
            ]
        }

    def fake_ds4_chat(self_or_None=None, *, model, messages, **kwargs):
        state["call_log"].append(("ds4", model))
        return {
            "choices": [
                {"message": {"role": "assistant", "content": judge_text},
                 "finish_reason": "stop"}
            ]
        }

    def fake_unload(self_or_None=None, models=None, **kwargs):
        state["unload_log"].append(list(models or []))

    # Patch the chat_completion method on both classes
    monkeypatch.setattr(OllamaClient, "chat_completion", fake_chat)
    monkeypatch.setattr(OllamaClient, "unload_models", fake_unload)
    monkeypatch.setattr(DS4Client, "chat_completion", fake_ds4_chat)

    return state


def _make_config(**overrides) -> FusionConfig:
    defaults = dict(
        qwable_host="127.0.0.1",
        qwable_port=8088,
        ollama_base_url="http://127.0.0.1:1234/v1",
        qwable_timeout_seconds=60,
        qwable_queue_timeout_seconds=5,
        qwable_max_concurrent_requests=1,
        ds4_base_url="http://127.0.0.1:8000/v1",
        ds4_timeout_seconds=60,
        local_model_backend="lmstudio",
        lmstudio_cli_path="/bin/echo",
        fusion_default_preset="quality",
    )
    defaults.update(overrides)
    return FusionConfig(**defaults)


def _make_task(raw_body: dict, profile: str = "fusion-agent") -> ParsedAgentTask:
    messages = raw_body.get("messages") or []
    text = ""
    if messages and isinstance(messages[0], dict):
        text = messages[0].get("content", "") or ""
    elif "input" in raw_body:
        text = str(raw_body["input"])
    return ParsedAgentTask(
        text=text,
        tools=[],
        tool_results=[],
        profile=profile,
        source_protocol="openai_chat",
        stream=False,
        raw_request=raw_body,
    )


# ─── End-to-end runner integration ────────────────────────────────────────


@pytest.mark.asyncio
async def test_fusion_agent_dispatches_to_run_fusion_agent(monkeypatch):
    """FusionCore.execute routes fusion-agent to the real runner."""
    state = _install_mock_clients(monkeypatch)
    core = FusionCore(_make_config())
    task = _make_task({
        "model": "qwable-fusion",
        "messages": [{"role": "user", "content": "hi"}],
        "fusion": {"preset": "quality"},
    })
    action = await core.execute(task)
    assert action.type == "final_answer"
    assert action.text is not None
    assert action.trace["fusion"]["preset"] == "quality"
    # Real runner was called: 3 panel + 1 judge = 4 chat calls
    assert len(state["call_log"]) == 4


@pytest.mark.asyncio
async def test_fusion_agent_uses_default_preset_when_no_override(monkeypatch):
    """Empty raw body → default preset (quality) is used."""
    _install_mock_clients(monkeypatch)
    core = FusionCore(_make_config())
    task = _make_task({"model": "qwable-fusion", "messages": []})
    action = await core.execute(task)
    assert action.type == "final_answer"
    assert action.trace["fusion"]["preset"] == "quality"


@pytest.mark.asyncio
async def test_fusion_agent_respects_custom_preset(monkeypatch):
    _install_mock_clients(monkeypatch)
    core = FusionCore(_make_config())
    task = _make_task({
        "model": "qwable-fusion",
        "messages": [{"role": "user", "content": "x"}],
        "fusion": {"preset": "budget"},
    })
    action = await core.execute(task)
    assert action.trace["fusion"]["preset"] == "budget"


@pytest.mark.asyncio
async def test_fusion_agent_respects_plugins_shape(monkeypatch):
    """OpenRouter plugins shape should also be honored."""
    _install_mock_clients(monkeypatch)
    core = FusionCore(_make_config())
    task = _make_task({
        "model": "qwable-fusion",
        "messages": [{"role": "user", "content": "x"}],
        "plugins": [{"id": "fusion", "preset": "coding"}],
    })
    action = await core.execute(task)
    assert action.trace["fusion"]["preset"] == "coding"


@pytest.mark.asyncio
async def test_fusion_agent_custom_panel_override(monkeypatch):
    _install_mock_clients(monkeypatch)
    core = FusionCore(_make_config())
    task = _make_task({
        "model": "qwable-fusion",
        "messages": [{"role": "user", "content": "x"}],
        "fusion": {
            "analysis_models": ["m1", "m2"],
            "judge_model": "m-judge",
        },
    })
    action = await core.execute(task)
    assert action.trace["fusion"]["preset"] == "custom"
    assert action.trace["fusion"]["analysis_models"] == ["m1", "m2"]
    assert action.trace["fusion"]["judge_model"] == "m-judge"


@pytest.mark.asyncio
async def test_fusion_agent_bad_preset_returns_error_action(monkeypatch):
    """Bad preset returns error action without invoking any model."""
    state = _install_mock_clients(monkeypatch)
    core = FusionCore(_make_config())
    task = _make_task({
        "model": "qwable-fusion",
        "messages": [{"role": "user", "content": "x"}],
        "fusion": {"preset": "does-not-exist"},
    })
    action = await core.execute(task)
    assert action.type == "final_answer"
    assert "does-not-exist" in action.text
    assert action.confidence == 0.0
    assert action.rationale_summary == "fusion_preset_error"
    # No models were called (preset resolution failed before runner)
    assert state["call_log"] == []


@pytest.mark.asyncio
async def test_fusion_agent_custom_panel_empty_returns_error(monkeypatch):
    state = _install_mock_clients(monkeypatch)
    core = FusionCore(_make_config())
    task = _make_task({
        "model": "qwable-fusion",
        "messages": [{"role": "user", "content": "x"}],
        "fusion": {"analysis_models": []},
    })
    action = await core.execute(task)
    assert action.type == "final_answer"
    assert action.confidence == 0.0
    assert state["call_log"] == []


@pytest.mark.asyncio
async def test_fusion_agent_does_not_affect_other_profiles(monkeypatch):
    """fast-agent profile should NOT go through fusion dispatch."""
    _install_mock_clients(monkeypatch)
    core = FusionCore(_make_config())
    assert hasattr(core, "_run_fusion_agent")


@pytest.mark.asyncio
async def test_fusion_agent_uses_ds4_judge_for_heavy_preset(monkeypatch):
    """Heavy preset should route judge call to ds4 backend."""
    state = _install_mock_clients(monkeypatch)
    core = FusionCore(_make_config())
    task = _make_task({
        "model": "qwable-fusion",
        "messages": [{"role": "user", "content": "x"}],
        "fusion": {"preset": "heavy"},
    })
    action = await core.execute(task)
    assert action.trace["fusion"]["judge_backend"] == "ds4"
    # Verify ds4 was actually called (judge call)
    ds4_calls = [c for c in state["call_log"] if isinstance(c, tuple) and c[0] == "ds4"]
    assert len(ds4_calls) == 1


@pytest.mark.asyncio
async def test_fusion_agent_returns_structured_sections(monkeypatch):
    """Successful structured judge should populate all 5 sections in trace."""
    _install_mock_clients(monkeypatch)
    core = FusionCore(_make_config())
    task = _make_task({
        "model": "qwable-fusion",
        "messages": [{"role": "user", "content": "x"}],
        "fusion": {"preset": "budget"},
    })
    action = await core.execute(task)
    structured = action.trace["fusion"]["structured"]
    assert "final_answer" in structured
    assert "consensus" in structured
    assert "contradictions" in structured
    assert "blind_spots" in structured
    assert "per_model_notes" in structured
    assert structured["had_fallback"] is False
    assert action.confidence == 0.85


@pytest.mark.asyncio
async def test_fusion_agent_unstructured_judge_uses_fallback(monkeypatch):
    """Judge that ignores format → had_fallback=True, lower confidence."""
    _install_mock_clients(monkeypatch, judge_text="raw unstructured judge output")
    core = FusionCore(_make_config())
    task = _make_task({
        "model": "qwable-fusion",
        "messages": [{"role": "user", "content": "x"}],
        "fusion": {"preset": "budget"},
    })
    action = await core.execute(task)
    assert action.trace["fusion"]["structured_had_fallback"] is True
    assert action.confidence == 0.5
    assert action.text == "raw unstructured judge output"


@pytest.mark.asyncio
async def test_fusion_agent_unloads_each_panel_model(monkeypatch):
    """Each panel model is unloaded between calls (LM Studio invariant)."""
    state = _install_mock_clients(monkeypatch)
    core = FusionCore(_make_config())
    task = _make_task({
        "model": "qwable-fusion",
        "messages": [{"role": "user", "content": "x"}],
        "fusion": {"preset": "quality"},
    })
    await core.execute(task)
    # G12-3: 3 panel models → 2 unloads (last kept resident for fast follow-up)
    assert len(state["unload_log"]) == 2
    # First 2 unloads are coder + qwen3.6
    assert "qwen/qwen3-coder-next" in state["unload_log"][0]
    assert "qwen/qwen3.6-35b-a3b" in state["unload_log"][1]
    # Last panel (r1) NOT in unload_log
    assert not any("deepseek-r1-distill" in str(u) for u in state["unload_log"])

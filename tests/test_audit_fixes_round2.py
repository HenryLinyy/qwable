"""Regression tests for the round-2 audit fixes (5 confirmed + 3 split HIGH)."""

import pytest

from qwable.agent_orchestrator import _parse_json_object, _extract_tool_call
from qwable.config import FusionConfig
from qwable.conversation_store import ConversationStore, ConversationMessage
from qwable.model_roles import WorkflowStage
from qwable.model_selector import ModelSelector


# ── #1 _parse_json_object: prose-with-brace + plain JSON, and nested objects ──

def test_parse_json_object_recovers_plan_after_brace_prose():
    # An earlier brace in the prose must not make the slice span two objects.
    out = _parse_json_object('Thinking about {} options, final: {"steps": ["s1"]}')
    assert out == {"steps": ["s1"]}


def test_parse_json_object_returns_outer_not_nested():
    # Nested braces must not cause an inner object to be returned.
    out = _parse_json_object('{"tool_call": {"name": "x", "input": {"a": 1}}}')
    assert "tool_call" in out


# ── #5 _extract_tool_call: unrecognized tool-intent shape is surfaced ──

def test_extract_tool_call_raises_on_unrecognized_tool_intent():
    with pytest.raises(ValueError):
        _extract_tool_call({"tool": "browser", "arguments": {"url": "x"}})


def test_extract_tool_call_none_when_no_tool_intent():
    # A plain completed step (no tool/action/arguments) still returns None.
    assert _extract_tool_call({"step_result": {"status": "done"}}) is None


# ── #7 select_for_stage: explicit temperature=0.0 must be preserved ──

def test_select_for_stage_preserves_zero_temperature():
    sel = ModelSelector(FusionConfig())
    chosen = sel.select_for_stage(WorkflowStage.EXECUTE_PATCH, temperature=0.0)
    assert chosen.generation_config.get("temperature") == 0.0


# ── #8 ConversationStore: lock + per-write temp keep both appends ──

def test_conversation_store_append_persists_sequentially(tmp_path):
    store = ConversationStore(store_dir=tmp_path)
    conv = store.create()
    store.append(conv.id, ConversationMessage(role="user", content="hello"))
    store.append(conv.id, ConversationMessage(role="assistant", content="hi"))
    loaded = store.get(conv.id)
    assert loaded is not None
    assert [m.content for m in loaded.messages] == ["hello", "hi"]
    assert hasattr(store, "_lock")

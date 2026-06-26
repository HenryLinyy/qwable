"""Tests for OpenAI Chat renderer."""

import json

from qwable.render_openai_chat import render_final_answer, render_tool_call
from qwable.schemas import FusionAction


def test_render_final_answer():
    """Render final_answer to chat message."""
    action = FusionAction(
        type="final_answer",
        text="Hello",
        tool_name=None,
        tool_input=None,
        confidence=1.0,
        rationale_summary=None,
    )
    result = render_final_answer(action)
    assert result["role"] == "assistant"
    assert result["content"] == "Hello"


def test_render_tool_call():
    """Render tool_call to chat message with tool_calls."""
    action = FusionAction(
        type="tool_call",
        text=None,
        tool_name="read_file",
        tool_input={"path": "test.txt"},
        confidence=0.9,
        rationale_summary=None,
    )
    result = render_tool_call(action)
    assert result["role"] == "assistant"
    assert result["content"] is None
    assert len(result["tool_calls"]) == 1
    assert result["tool_calls"][0]["function"]["name"] == "read_file"
    arguments = result["tool_calls"][0]["function"]["arguments"]
    assert isinstance(arguments, str)
    assert json.loads(arguments) == {"path": "test.txt"}

"""Tests for OpenAI Responses renderer."""

import json

from qwable.render_openai_responses import render_final_answer, render_tool_call
from qwable.schemas import FusionAction


def test_render_final_answer():
    """Render final_answer to output_text."""
    action = FusionAction(
        type="final_answer",
        text="Hello",
        tool_name=None,
        tool_input=None,
        confidence=1.0,
        rationale_summary=None,
    )
    result = render_final_answer(action)
    assert result["type"] == "output_text"
    assert result["text"] == "Hello"


def test_render_tool_call():
    """Render tool_call to function_call."""
    action = FusionAction(
        type="tool_call",
        text=None,
        tool_name="read_file",
        tool_input={"path": "test.txt"},
        confidence=0.9,
        rationale_summary=None,
    )
    result = render_tool_call(action)
    assert result["type"] == "function_call"
    assert result["name"] == "read_file"
    assert isinstance(result["arguments"], str)
    assert json.loads(result["arguments"]) == {"path": "test.txt"}

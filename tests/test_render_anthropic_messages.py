"""Tests for Anthropic Messages renderer."""

from qwable.render_anthropic_messages import render_final_answer, render_tool_use
from qwable.schemas import FusionAction


def test_render_final_answer():
    """Render final_answer to Anthropic text block."""
    action = FusionAction(type="final_answer", text="Hello", tool_name=None, tool_input=None, confidence=1.0, rationale_summary=None)
    result = render_final_answer(action)
    assert result["type"] == "text"
    assert result["text"] == "Hello"


def test_render_tool_use():
    """Render tool_call to Anthropic tool_use block."""
    action = FusionAction(type="tool_call", text=None, tool_name="Bash", tool_input={"command": "ls"}, confidence=0.9, rationale_summary=None)
    result = render_tool_use(action)
    assert result["type"] == "tool_use"
    assert result["name"] == "Bash"
    assert result["input"] == {"command": "ls"}

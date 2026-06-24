"""Render FusionAction back to Anthropic Messages API format."""

from qwable.schemas import FusionAction


def render_final_answer(action: FusionAction) -> dict:
    """Render a final_answer action to Anthropic content block."""
    text = action.text or ""
    return {
        "type": "text",
        "text": text,
    }


def render_tool_use(action: FusionAction, tool_use_id: str = "toolu_local_1") -> dict:
    """Render a tool_call action to Anthropic tool_use content block."""
    name = action.tool_name or ""
    inp = action.tool_input or {}

    return {
        "type": "tool_use",
        "id": tool_use_id,
        "name": name,
        "input": inp,
    }

"""Render FusionAction back to OpenAI Responses API format."""

import json

from qwable.schemas import FusionAction


def render_final_answer(action: FusionAction) -> dict:
    """Render a final_answer action to OpenAI Responses output."""
    text = action.text or ""
    return {
        "type": "output_text",
        "text": text,
    }


def render_tool_call(action: FusionAction, call_id: str = "call_local_1") -> dict:
    """Render a tool_call action to OpenAI Responses output."""
    name = action.tool_name or ""
    arguments = _arguments_json(action.tool_input or {})

    return {
        "type": "function_call",
        "call_id": call_id,
        "name": name,
        "arguments": arguments,
    }


def _arguments_json(arguments: dict | str) -> str:
    if isinstance(arguments, str):
        return arguments
    return json.dumps(arguments, ensure_ascii=False, separators=(",", ":"))

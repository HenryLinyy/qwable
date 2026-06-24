"""Render FusionAction back to OpenAI Chat Completions format."""

import json

from qwable.schemas import FusionAction


def render_final_answer(action: FusionAction) -> dict:
    """Render a final_answer action to Chat Completions message."""
    text = action.text or ""
    return {
        "role": "assistant",
        "content": text,
    }


def render_tool_call(action: FusionAction, call_id: str = "call_local_1") -> dict:
    """Render a tool_call action to Chat Completions message with tool_calls."""
    name = action.tool_name or ""
    arguments = _arguments_json(action.tool_input or {})

    return {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": call_id,
                "type": "function",
                "function": {
                    "name": name,
                    "arguments": arguments,
                },
            }
        ],
    }


def _arguments_json(arguments: dict | str) -> str:
    if isinstance(arguments, str):
        return arguments
    return json.dumps(arguments, ensure_ascii=False, separators=(",", ":"))

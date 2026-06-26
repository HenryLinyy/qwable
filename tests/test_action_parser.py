"""Tests for action parser."""

from qwable.action_parser import parse_action_from_text


def test_parse_json_final_answer():
    """Parse JSON final_answer action."""
    text = '{"type": "final_answer", "text": "Hello world", "confidence": 0.95}'
    action = parse_action_from_text(text)
    assert action.type == "final_answer"
    assert action.text == "Hello world"
    assert action.confidence == 0.95


def test_parse_json_tool_call():
    """Parse JSON tool_call action."""
    text = '{"type": "tool_call", "tool_name": "read_file", "tool_input": {"path": "test.txt"}, "confidence": 0.9}'
    action = parse_action_from_text(text)
    assert action.type == "tool_call"
    assert action.tool_name == "read_file"
    assert action.tool_input == {"path": "test.txt"}


def test_parse_plain_text_fallback():
    """Fall back to final_answer for plain text."""
    text = "Hello world"
    action = parse_action_from_text(text)
    assert action.type == "final_answer"
    assert action.text == "Hello world"
    assert action.confidence == 1.0


def test_parse_with_think_block():
    """Strip think blocks before parsing JSON."""
    text = '<thinking>reasoning</thinking>{"type": "final_answer", "text": "Hello"}'
    action = parse_action_from_text(text)
    assert action.type == "final_answer"
    assert action.text == "Hello"


def test_parse_native_tool_call():
    """Parse native tool call from Ollama response."""
    raw = {
        "choices": [
            {
                "message": {
                    "tool_calls": [
                        {
                            "function": {
                                "name": "run_shell",
                                "arguments": {"command": "ls"},
                            }
                        }
                    ],
                }
            }
        ]
    }
    tools = [
        {
            "type": "function",
            "function": {
                "name": "run_shell",
                "parameters": {"type": "object"},
            },
        }
    ]
    action = parse_action_from_text(raw, tools=tools)
    assert action.type == "tool_call"
    assert action.tool_name == "run_shell"


def test_parse_native_tool_call_json_string_arguments():
    """Parse OpenAI-compatible tool arguments strings into dicts."""
    raw = {
        "choices": [
            {
                "message": {
                    "tool_calls": [
                        {
                            "function": {
                                "name": "run_shell",
                                "arguments": '{"command":"ls"}',
                            }
                        }
                    ],
                }
            }
        ]
    }
    tools = [
        {
            "type": "function",
            "function": {
                "name": "run_shell",
                "parameters": {
                    "type": "object",
                    "properties": {"command": {"type": "string"}},
                    "required": ["command"],
                },
            },
        }
    ]
    action = parse_action_from_text(raw, tools=tools)
    assert action.type == "tool_call"
    assert action.tool_input == {"command": "ls"}


def test_parse_rejects_unknown_tool():
    """Reject model-selected tools that were not provided by the client."""
    raw = {
        "tool_calls": [{"function": {"name": "delete_everything", "arguments": "{}"}}]
    }
    tools = [{"type": "function", "function": {"name": "run_shell", "parameters": {}}}]
    action = parse_action_from_text(raw, tools=tools)
    assert action.type == "final_answer"
    assert action.confidence == 0.0
    assert "not provided" in action.rationale_summary


def test_parse_rejects_invalid_tool_input_schema():
    """Reject tool inputs that do not satisfy the declared schema."""
    raw = {"tool_calls": [{"function": {"name": "run_shell", "arguments": "{}"}}]}
    tools = [
        {
            "type": "function",
            "function": {
                "name": "run_shell",
                "parameters": {"type": "object", "required": ["command"]},
            },
        }
    ]
    action = parse_action_from_text(raw, tools=tools)
    assert action.type == "final_answer"
    assert action.confidence == 0.0
    assert "required" in action.rationale_summary


def test_parse_unclosed_think_block_fails_closed():
    """Unclosed think blocks should not leak raw model text as a final answer."""
    action = parse_action_from_text("<think>hidden")
    assert action.type == "final_answer"
    assert action.text == ""
    assert action.confidence == 0.0

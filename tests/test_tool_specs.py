"""Tests for tool spec normalization."""

from qwable.tool_specs import normalize_openai_tools, normalize_anthropic_tools


def test_normalize_openai_tools():
    """OpenAI tools should normalize to ToolSpec list."""
    tools = [
        {
            "type": "function",
            "function": {
                "name": "read_file",
                "description": "Read a file",
                "parameters": {"type": "object", "properties": {"path": {"type": "string"}}},
            },
        }
    ]
    result = normalize_openai_tools(tools, "openai_responses")
    assert len(result) == 1
    assert result[0].name == "read_file"
    assert result[0].description == "Read a file"
    assert result[0].source_protocol == "openai_responses"


def test_normalize_openai_tools_flat():
    """OpenAI tools in flat format (name at top level)."""
    tools = [
        {
            "type": "function",
            "name": "run_shell",
            "description": "Run a command",
            "parameters": {"type": "object", "properties": {"cmd": {"type": "string"}}},
        }
    ]
    result = normalize_openai_tools(tools, "openai_chat")
    assert len(result) == 1
    assert result[0].name == "run_shell"


def test_normalize_anthropic_tools():
    """Anthropic tools should normalize to ToolSpec list."""
    tools = [
        {
            "name": "Bash",
            "description": "Run a shell command",
            "input_schema": {
                "type": "object",
                "properties": {"command": {"type": "string"}},
                "required": ["command"],
            },
        }
    ]
    result = normalize_anthropic_tools(tools)
    assert len(result) == 1
    assert result[0].name == "Bash"
    assert result[0].source_protocol == "anthropic_messages"


def test_empty_tools():
    """Empty tools list should return empty."""
    assert normalize_openai_tools([], "openai_responses") == []
    assert normalize_anthropic_tools([]) == []

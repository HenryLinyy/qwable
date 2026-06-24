"""Tests for Anthropic Messages input parsing."""

from qwable.message_parsing import parse_anthropic_messages_input


def test_parse_simple_text():
    """Parse a simple text message."""
    body = {
        "model": "claude-qwable-fast",
        "max_tokens": 1024,
        "messages": [{"role": "user", "content": "Hello"}],
        "stream": False,
    }
    task = parse_anthropic_messages_input(body)
    assert "[user]: Hello" in task.text
    assert task.profile == "fast-agent"
    assert task.source_protocol == "anthropic_messages"


def test_parse_with_tools():
    """Parse input with Anthropic tools."""
    body = {
        "model": "claude-qwable-full",
        "max_tokens": 1024,
        "tools": [
            {
                "name": "Bash",
                "description": "Run a shell command",
                "input_schema": {
                    "type": "object",
                    "properties": {"command": {"type": "string"}},
                    "required": ["command"],
                },
            }
        ],
        "messages": [{"role": "user", "content": "List directory"}],
        "stream": False,
    }
    task = parse_anthropic_messages_input(body)
    assert task.profile == "full-agent"
    assert len(task.tools) == 1
    assert task.tools[0].name == "Bash"


def test_parse_with_tool_result():
    """Parse input with tool_result block."""
    body = {
        "model": "claude-qwable-fast",
        "max_tokens": 1024,
        "tools": [{"name": "Bash", "description": "Shell", "input_schema": {"type": "object"}}],
        "messages": [
            {"role": "user", "content": "List directory"},
            {
                "role": "assistant",
                "content": [{"type": "tool_use", "id": "toolu_1", "name": "Bash", "input": {"command": "ls"}}],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "toolu_1",
                        "content": "file1.txt\nfile2.txt",
                    }
                ],
            },
        ],
        "stream": False,
    }
    task = parse_anthropic_messages_input(body)
    assert len(task.tool_results) == 1
    assert task.tool_results[0].tool_call_id == "toolu_1"
    assert task.tool_results[0].content == "file1.txt\nfile2.txt"


def test_parse_with_system():
    """Parse input with system prompt."""
    body = {
        "model": "claude-qwable-heavy",
        "max_tokens": 1024,
        "system": "You are an expert.",
        "messages": [{"role": "user", "content": "Analyze"}],
        "stream": False,
    }
    task = parse_anthropic_messages_input(body)
    assert "[system]: You are an expert." in task.text
    assert task.profile == "heavy-agent"

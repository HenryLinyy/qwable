"""Tests for OpenAI Responses input parsing."""

from qwable.message_parsing import parse_openai_responses_input


def test_parse_simple_input():
    """Parse a simple string input."""
    body = {
        "model": "qwable-fast",
        "input": "Hello world",
        "stream": False,
    }
    task = parse_openai_responses_input(body)
    assert task.text == "Hello world"
    assert task.profile == "fast-agent"
    assert task.source_protocol == "openai_responses"
    assert not task.stream
    assert len(task.tools) == 0
    assert len(task.tool_results) == 0


def test_parse_input_with_tools():
    """Parse input with tools."""
    body = {
        "model": "qwable-full",
        "input": "Read a file",
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "read_file",
                    "description": "Read a file",
                    "parameters": {"type": "object", "properties": {"path": {"type": "string"}}},
                },
            }
        ],
        "stream": True,
    }
    task = parse_openai_responses_input(body)
    assert task.profile == "full-agent"
    assert task.stream
    assert len(task.tools) == 1
    assert task.tools[0].name == "read_file"


def test_parse_function_call_output():
    """Parse function_call_output from previous step."""
    body = {
        "model": "qwable-fast",
        "input": [
            {"role": "user", "content": "List directory"},
            {"type": "function_call_output", "call_id": "call_1", "name": "run_shell", "output": "file1.txt\nfile2.txt"},
        ],
        "stream": False,
    }
    task = parse_openai_responses_input(body)
    assert len(task.tool_results) == 1
    assert task.tool_results[0].tool_call_id == "call_1"
    assert task.tool_results[0].name == "run_shell"
    assert task.tool_results[0].content == "file1.txt\nfile2.txt"


def test_parse_heavy_model():
    """Parse heavy-agent model name."""
    body = {
        "model": "qwable-heavy",
        "input": "Analyze large repo",
        "stream": False,
    }
    task = parse_openai_responses_input(body)
    assert task.profile == "heavy-agent"

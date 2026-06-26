"""Tests for fusion core with mocked clients."""

import pytest
from unittest.mock import MagicMock
from qwable.fusion_core import FusionCore
from qwable.config import FusionConfig
from qwable.schemas import ParsedAgentTask, ToolSpec, ToolResult


@pytest.fixture
def mock_config():
    cfg = FusionConfig()
    return cfg


@pytest.fixture
def mock_ollama():
    ollama = MagicMock()
    ollama.chat_completion.return_value = {
        "choices": [{"message": {"content": "Hello world"}}]
    }
    return ollama


@pytest.fixture
def mock_ds4():
    ds4 = MagicMock()
    ds4.health.return_value = True
    ds4.chat_completion.return_value = {
        "choices": [{"message": {"content": "ds4 response"}}]
    }
    return ds4


@pytest.mark.asyncio
async def test_fast_agent_final_answer(mock_config, mock_ollama, mock_ds4):
    """Fast-agent should return a final_answer from Ollama."""
    core = FusionCore(mock_config)
    core.ollama = mock_ollama
    core.ds4 = mock_ds4

    task = ParsedAgentTask(
        text="Hello",
        tools=[],
        tool_results=[],
        profile="fast-agent",
        source_protocol="openai_responses",
        stream=False,
        raw_request={},
    )
    action = await core.execute(task)
    assert action.type == "final_answer"
    assert action.text == "Hello world"


@pytest.mark.asyncio
async def test_fast_agent_with_tools(mock_config, mock_ollama, mock_ds4):
    """Fast-agent with tools should pass them to Ollama."""
    core = FusionCore(mock_config)
    core.ollama = mock_ollama
    core.ds4 = mock_ds4

    task = ParsedAgentTask(
        text="Read file",
        tools=[
            ToolSpec(
                name="read_file",
                description="Read",
                input_schema={},
                source_protocol="openai_responses",
                raw={},
            )
        ],
        tool_results=[],
        profile="fast-agent",
        source_protocol="openai_responses",
        stream=False,
        raw_request={},
    )
    action = await core.execute(task)
    assert action.type == "final_answer"


@pytest.mark.asyncio
async def test_fast_agent_respects_request_options_and_tool_choice_none(
    mock_config, mock_ollama, mock_ds4
):
    """Fast-agent should honor request max_tokens, temperature, and tool_choice=none."""
    core = FusionCore(mock_config)
    core.ollama = mock_ollama
    core.ds4 = mock_ds4

    task = ParsedAgentTask(
        text="Read file",
        tools=[
            ToolSpec(
                name="read_file",
                description="Read",
                input_schema={},
                source_protocol="openai_responses",
                raw={},
            )
        ],
        tool_results=[],
        profile="fast-agent",
        source_protocol="openai_responses",
        stream=False,
        raw_request={"max_tokens": 42, "temperature": 0.2, "tool_choice": "none"},
    )
    action = await core.execute(task)
    assert action.type == "final_answer"
    kwargs = mock_ollama.chat_completion.call_args.kwargs
    assert kwargs["max_tokens"] == 42
    assert kwargs["temperature"] == 0.2
    assert kwargs["tools"] is None


@pytest.mark.asyncio
async def test_fast_agent_rejects_context_over_limit_before_model_call(
    mock_config, mock_ollama, mock_ds4
):
    """Fast-agent should return an explicit context error instead of silently overfeeding the model."""
    cfg = FusionConfig(fast_max_input_chars=8)
    core = FusionCore(cfg)
    core.ollama = mock_ollama
    core.ds4 = mock_ds4

    task = ParsedAgentTask(
        text="this input is too long",
        tools=[],
        tool_results=[],
        profile="fast-agent",
        source_protocol="openai_responses",
        stream=False,
        raw_request={},
    )
    action = await core.execute(task)
    assert action.type == "final_answer"
    assert "context limit exceeded" in action.text
    assert action.trace["error"] == "context_limit_exceeded"
    mock_ollama.chat_completion.assert_not_called()


@pytest.mark.asyncio
async def test_fast_agent_sends_tool_results_as_authoritative_user_evidence(
    mock_config, mock_ollama, mock_ds4
):
    """function_call_output evidence should be prominent user evidence, not assistant prose."""
    core = FusionCore(mock_config)
    core.ollama = mock_ollama
    core.ds4 = mock_ds4

    task = ParsedAgentTask(
        text="請根據工具結果回答有哪些項目。",
        tools=[],
        tool_results=[
            ToolResult(
                tool_call_id="call_local_1",
                name="run_shell",
                content="README.md\nqwable\nscripts\ntests\n",
                is_error=False,
                source_protocol="openai_responses",
                raw={},
            )
        ],
        profile="fast-agent",
        source_protocol="openai_responses",
        stream=False,
        raw_request={"tool_choice": "none"},
    )
    action = await core.execute(task)
    assert action.type == "final_answer"
    messages = mock_ollama.chat_completion.call_args.kwargs["messages"]
    assert messages[-1]["role"] == "user"
    evidence = messages[-1]["content"]
    assert "AUTHORITATIVE_TOOL_RESULT" in evidence
    assert "tool_call_id=call_local_1" in evidence
    assert "tool_name=run_shell" in evidence
    assert "README.md\nqwable\nscripts\ntests" in evidence
    assert "只根據上述工具結果" in evidence


@pytest.mark.asyncio
async def test_full_agent_with_tools_returns_coder_tool_call_without_panel(
    mock_config, mock_ds4
):
    """Full-agent tool loop should use one coder native-tool step before any panel."""
    core = FusionCore(mock_config)
    core.ollama = MagicMock()
    core.ds4 = mock_ds4
    core.ollama.chat_completion.return_value = {
        "choices": [
            {
                "message": {
                    "tool_calls": [
                        {
                            "function": {
                                "name": "read_file",
                                "arguments": {"path": "README.md"},
                            }
                        }
                    ]
                }
            }
        ]
    }

    task = ParsedAgentTask(
        text="Read README",
        tools=[
            ToolSpec(
                name="read_file",
                description="Read",
                input_schema={
                    "type": "object",
                    "properties": {"path": {"type": "string"}},
                    "required": ["path"],
                },
                source_protocol="openai_responses",
                raw={},
            )
        ],
        tool_results=[],
        profile="full-agent",
        source_protocol="openai_responses",
        stream=False,
        raw_request={},
    )
    action = await core.execute(task)
    assert action.type == "tool_call"
    assert action.tool_name == "read_file"
    assert action.tool_input == {"path": "README.md"}
    assert core.ollama.chat_completion.call_count == 1


@pytest.mark.asyncio
async def test_chat_agent_final_answer(mock_config, mock_ollama, mock_ds4):
    """Chat-agent should return a final_answer."""
    core = FusionCore(mock_config)
    core.ollama = mock_ollama
    core.ds4 = mock_ds4

    task = ParsedAgentTask(
        text="Hello",
        tools=[],
        tool_results=[],
        profile="chat-agent",
        source_protocol="openai_chat",
        stream=False,
        raw_request={},
    )
    action = await core.execute(task)
    assert action.type == "final_answer"
    assert action.text == "Hello world"

"""Tests for heavy-agent with mocked ds4 and Ollama."""

import pytest
from unittest.mock import MagicMock
from qwable.fusion_core import FusionCore
from qwable.config import FusionConfig
from qwable.schemas import ParsedAgentTask, ToolSpec


@pytest.fixture
def mock_config():
    return FusionConfig()


@pytest.fixture
def fit_config():
    return FusionConfig(
        est_model_heavy_gb=20,
        est_model_coder_gb=10,
        est_model_critic_gb=10,
        est_model_judge_gb=10,
    )


@pytest.mark.asyncio
async def test_heavy_agent_ds4_online(fit_config):
    """Heavy-agent should use ds4 when online."""
    core = FusionCore(fit_config)
    core.ollama = MagicMock()
    core.ds4 = MagicMock()
    core.ds4.health.return_value = True
    core.ds4.chat_completion.return_value = {
        "choices": [
            {
                "message": {
                    "content": '{"type": "final_answer", "text": "ds4 analysis complete"}'
                }
            }
        ]
    }

    # Use side_effect to simulate 3 sequential ollama calls (checker, critic, judge)
    ollama_responses = [
        {"choices": [{"message": {"content": "checker ok"}}]},
        {"choices": [{"message": {"content": "critic ok"}}]},
        {
            "choices": [
                {
                    "message": {
                        "content": '{"type": "final_answer", "text": "ds4 analysis complete\\n\\n[checker/critic approved]", "confidence": 0.95}'
                    }
                }
            ]
        },
    ]
    core.ollama.chat_completion.side_effect = ollama_responses

    task = ParsedAgentTask(
        text="Analyze this repo",
        tools=[],
        tool_results=[],
        profile="heavy-agent",
        source_protocol="openai_responses",
        stream=False,
        raw_request={},
    )
    action = await core.execute(task)
    assert action.type == "final_answer"
    assert "ds4 analysis" in action.text


@pytest.mark.asyncio
async def test_heavy_agent_resource_guard_skips_ollama_panel_when_unfit(mock_config):
    """Heavy-agent should not hard-run Ollama panel when ds4 + panel exceeds memory estimate."""
    core = FusionCore(mock_config)
    core.ollama = MagicMock()
    core.ds4 = MagicMock()
    core.ds4.health.return_value = True
    core.ds4.chat_completion.return_value = {
        "choices": [{"message": {"content": "ds4 primary answer"}}]
    }

    task = ParsedAgentTask(
        text="Analyze this repo",
        tools=[],
        tool_results=[],
        profile="heavy-agent",
        source_protocol="openai_responses",
        stream=False,
        raw_request={},
    )
    action = await core.execute(task)
    assert action.type == "final_answer"
    assert action.text == "ds4 primary answer"
    assert "heavy_resource_guard" in (action.rationale_summary or "")
    assert action.trace["profile"] == "heavy-agent"
    assert action.trace["heavy_backend"] == "ds4"
    assert action.trace["fallback"] is None
    assert action.trace["resource_guard"] is True
    # Memory estimate: est_model_heavy=90 + LM Studio coder=65 + critic=66 + judge=66 + KV=5.
    assert "292.0GB exceeds limit" in action.trace["reason"]
    core.ollama.chat_completion.assert_not_called()


@pytest.mark.asyncio
async def test_heavy_agent_unloads_ollama_models_before_ds4_primary(mock_config):
    """Heavy-agent should release resident Ollama models before starting ds4 primary."""
    core = FusionCore(mock_config)
    core.ollama = MagicMock()
    core.ds4 = MagicMock()
    core.ds4.health.return_value = True

    def ds4_primary_response(**kwargs):
        assert core.ollama.unload_models.called
        return {"choices": [{"message": {"content": "ds4 primary answer"}}]}

    core.ds4.chat_completion.side_effect = ds4_primary_response

    task = ParsedAgentTask(
        text="Analyze this repo",
        tools=[],
        tool_results=[],
        profile="heavy-agent",
        source_protocol="openai_responses",
        stream=False,
        raw_request={},
    )
    action = await core.execute(task)
    assert action.type == "final_answer"
    assert action.text == "ds4 primary answer"

    core.ollama.unload_models.assert_called_once()
    unload_models = core.ollama.unload_models.call_args.args[0]
    # Order mirrors _ollama_models_for_unload() in fusion_core.py, with dupes
    # removed via dict.fromkeys. With LM Studio defaults:
    #   - model_tooler == model_coder → deduped
    #   - model_critic == model_judge → deduped
    #   - formatter/vision-fast/formatter-mlx == fast → deduped
    #   - hermes-pro/agentic-mlx == agentic-pro → deduped
    assert unload_models == [
        mock_config.model_fast,  # google/gemma-4-26b-a4b-qat
        mock_config.model_coder,  # qwen/qwen3-coder-next
        mock_config.model_critic,  # deepseek-r1-distill-qwen-32b
        mock_config.model_vision_pro,  # qwen/qwen3-vl-30b
        mock_config.model_agentic_pro,  # qwen/qwen3.6-35b-a3b
    ]
    assert mock_config.model_vision_fast == mock_config.model_formatter


@pytest.mark.asyncio
async def test_heavy_agent_falls_back_to_ds4_primary_when_judge_content_empty(
    fit_config,
):
    """Heavy-agent should not return empty text when reasoning-only judge output has no content."""
    core = FusionCore(fit_config)
    core.ollama = MagicMock()
    core.ds4 = MagicMock()
    core.ds4.health.return_value = True
    core.ds4.chat_completion.return_value = {
        "choices": [{"message": {"content": "ds4 primary answer"}}]
    }

    ollama_responses = [
        {"choices": [{"message": {"content": "checker ok"}}]},
        {"choices": [{"message": {"content": "critic ok"}}]},
        {
            "choices": [
                {"message": {"content": "", "reasoning": "internal reasoning only"}}
            ]
        },
    ]
    core.ollama.chat_completion.side_effect = ollama_responses

    task = ParsedAgentTask(
        text="Analyze this repo",
        tools=[],
        tool_results=[],
        profile="heavy-agent",
        source_protocol="openai_responses",
        stream=False,
        raw_request={},
    )
    action = await core.execute(task)
    assert action.type == "final_answer"
    assert action.text == "ds4 primary answer"
    assert "judge_empty" in (action.rationale_summary or "")
    assert action.trace["profile"] == "heavy-agent"
    assert action.trace["heavy_backend"] == "ds4"
    assert action.trace["judge_empty"] is True
    assert "internal reasoning" not in action.text


@pytest.mark.asyncio
async def test_heavy_agent_fallback_empty_output_returns_explicit_error(mock_config):
    """Heavy-agent fallback should not report success with empty full-agent output."""
    core = FusionCore(mock_config)
    core.ollama = MagicMock()
    core.ds4 = MagicMock()
    core.ds4.health.return_value = True
    core.ds4.chat_completion.side_effect = Exception("Metal OOM")
    core.ollama.chat_completion.side_effect = [
        {"choices": [{"message": {"content": "coder proposal"}}]},
        {"choices": [{"message": {"content": "tooler review"}}]},
        {"choices": [{"message": {"content": "critic review"}}]},
        {"choices": [{"message": {"content": "", "reasoning": "internal only"}}]},
    ]

    task = ParsedAgentTask(
        text="Analyze",
        tools=[],
        tool_results=[],
        profile="heavy-agent",
        source_protocol="openai_responses",
        stream=False,
        raw_request={},
    )
    action = await core.execute(task)
    assert action.type == "final_answer"
    assert action.text
    assert "ds4_primary_failed" in action.text
    assert action.trace["error"] == "fallback_empty"
    assert action.trace["fallback_reason"] == "ds4_primary_failed"


@pytest.mark.asyncio
async def test_heavy_agent_ds4_offline_fallback(mock_config):
    """Heavy-agent should fall back to full-agent when ds4 is offline."""
    core = FusionCore(mock_config)
    core.ollama = MagicMock()
    core.ds4 = MagicMock()
    core.ds4.health.return_value = False

    # Full-agent calls ollama 4 times (coder, tooler, critic, judge)
    ollama_responses = [
        {"choices": [{"message": {"content": "coder proposal"}}]},
        {"choices": [{"message": {"content": "tooler review"}}]},
        {"choices": [{"message": {"content": "critic review"}}]},
        {
            "choices": [
                {
                    "message": {
                        "content": '{"type": "final_answer", "text": "fallback answer", "confidence": 0.9}'
                    }
                }
            ]
        },
    ]
    core.ollama.chat_completion.side_effect = ollama_responses

    task = ParsedAgentTask(
        text="Analyze",
        tools=[],
        tool_results=[],
        profile="heavy-agent",
        source_protocol="openai_responses",
        stream=False,
        raw_request={},
    )
    action = await core.execute(task)
    assert action.type == "final_answer"
    assert action.trace["profile"] == "heavy-agent"
    assert action.trace["heavy_backend"] is None
    assert action.trace["fallback"] == "full-agent"
    assert action.trace["fallback_reason"] == "ds4_offline"


@pytest.mark.asyncio
async def test_heavy_agent_ds4_timeout_fallback(mock_config):
    """Heavy-agent should fall back when ds4 times out."""
    core = FusionCore(mock_config)
    core.ollama = MagicMock()
    core.ds4 = MagicMock()
    core.ds4.health.return_value = True
    core.ds4.chat_completion.side_effect = Exception("Timeout")

    # Full-agent calls ollama 4 times (coder, tooler, critic, judge)
    ollama_responses = [
        {"choices": [{"message": {"content": "coder proposal"}}]},
        {"choices": [{"message": {"content": "tooler review"}}]},
        {"choices": [{"message": {"content": "critic review"}}]},
        {
            "choices": [
                {
                    "message": {
                        "content": '{"type": "final_answer", "text": "timeout fallback", "confidence": 0.9}'
                    }
                }
            ]
        },
    ]
    core.ollama.chat_completion.side_effect = ollama_responses

    task = ParsedAgentTask(
        text="Analyze",
        tools=[],
        tool_results=[],
        profile="heavy-agent",
        source_protocol="openai_responses",
        stream=False,
        raw_request={},
    )
    action = await core.execute(task)
    assert action.type == "final_answer"
    assert action.trace["profile"] == "heavy-agent"
    assert action.trace["heavy_backend"] is None
    assert action.trace["fallback"] == "full-agent"
    assert action.trace["fallback_reason"] == "ds4_primary_failed"


@pytest.mark.asyncio
async def test_heavy_agent_with_tools_returns_coder_tool_call_before_ds4(mock_config):
    """Heavy-agent tool loop should ask coder for native tool calls before touching ds4."""
    core = FusionCore(mock_config)
    core.ollama = MagicMock()
    core.ds4 = MagicMock()
    core.ollama.chat_completion.return_value = {
        "choices": [
            {
                "message": {
                    "tool_calls": [
                        {
                            "function": {
                                "name": "run_shell",
                                "arguments": {"command": "pwd"},
                            }
                        }
                    ]
                }
            }
        ]
    }

    task = ParsedAgentTask(
        text="Check working directory",
        tools=[
            ToolSpec(
                name="run_shell",
                description="Run shell",
                input_schema={
                    "type": "object",
                    "properties": {"command": {"type": "string"}},
                    "required": ["command"],
                },
                source_protocol="openai_responses",
                raw={},
            )
        ],
        tool_results=[],
        profile="heavy-agent",
        source_protocol="openai_responses",
        stream=False,
        raw_request={},
    )
    action = await core.execute(task)
    assert action.type == "tool_call"
    assert action.tool_name == "run_shell"
    assert action.tool_input == {"command": "pwd"}
    assert action.trace["profile"] == "heavy-agent"
    assert action.trace["tool_loop"] == "coder"
    core.ds4.health.assert_not_called()
    core.ds4.chat_completion.assert_not_called()


@pytest.mark.asyncio
async def test_heavy_agent_ds4_bad_response_fallback(fit_config):
    """Heavy-agent should fallback with trace when ds4 returns no primary content."""
    core = FusionCore(fit_config)
    core.ollama = MagicMock()
    core.ds4 = MagicMock()
    core.ds4.health.return_value = True
    core.ds4.chat_completion.return_value = {"choices": [{"message": {}}]}
    core.ollama.chat_completion.side_effect = [
        {"choices": [{"message": {"content": "coder proposal"}}]},
        {"choices": [{"message": {"content": "tooler review"}}]},
        {"choices": [{"message": {"content": "critic review"}}]},
        {
            "choices": [
                {
                    "message": {
                        "content": '{"type": "final_answer", "text": "bad response fallback"}'
                    }
                }
            ]
        },
    ]

    task = ParsedAgentTask(
        text="Analyze",
        tools=[],
        tool_results=[],
        profile="heavy-agent",
        source_protocol="openai_responses",
        stream=False,
        raw_request={},
    )
    action = await core.execute(task)
    assert action.type == "final_answer"
    assert action.text == "bad response fallback"
    assert action.trace["fallback"] == "full-agent"
    assert action.trace["fallback_reason"] == "ds4_bad_response"


@pytest.mark.asyncio
async def test_heavy_agent_fallback_failure_returns_explicit_error(mock_config):
    """Heavy-agent should return an explicit error action if the full-agent fallback also fails."""
    core = FusionCore(mock_config)
    core.ollama = MagicMock()
    core.ds4 = MagicMock()
    core.ds4.health.return_value = False
    core.ollama.chat_completion.side_effect = Exception("ollama down")

    task = ParsedAgentTask(
        text="Analyze",
        tools=[],
        tool_results=[],
        profile="heavy-agent",
        source_protocol="openai_responses",
        stream=False,
        raw_request={},
    )
    action = await core.execute(task)
    assert action.type == "final_answer"
    assert "heavy-agent fallback failed" in action.text
    assert action.trace["error"] == "fallback_failed"
    assert action.trace["fallback_reason"] == "ds4_offline"

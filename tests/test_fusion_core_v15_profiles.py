"""Tests for v1.5 vision and pro profile execution contracts."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from qwable.config import FusionConfig
from qwable.fusion_core import FusionCore
from qwable.schemas import ParsedAgentTask, ToolResult, ToolSpec
from qwable.vision import ImageInput, VisionEvidence


def _image() -> ImageInput:
    return ImageInput(
        source_protocol="openai_responses",
        mime_type="image/png",
        data_base64="aW1hZ2U=",
        url=None,
        local_path=None,
        detail="auto",
        raw={},
    )


def _task(
    profile: str, *, images=None, tools=None, tool_results=None
) -> ParsedAgentTask:
    return ParsedAgentTask(
        text="Please inspect the screenshot",
        tools=tools or [],
        tool_results=tool_results or [],
        profile=profile,
        source_protocol="openai_responses",
        stream=False,
        raw_request={},
        images=images or [],
    )


def _evidence(profile: str = "vision-pro") -> VisionEvidence:
    return VisionEvidence(
        model="qwen3-vl:30b-a3b-instruct" if profile == "vision-pro" else "gemma4:12b",
        profile=profile,
        summary="A screenshot with a Save button.",
        visible_text="Save",
        ui_elements=[{"type": "button", "text": "Save"}],
        tables=[],
        charts=[],
        warnings=[],
        confidence=0.8,
        raw_text="Summary: A screenshot with a Save button.\nVisible Text: Save",
    )


@pytest.mark.asyncio
async def test_vision_evidence_is_injected_before_tool_results():
    core = FusionCore(FusionConfig())
    core.ollama = MagicMock()
    core.ds4 = MagicMock()
    core.ollama.chat_completion.return_value = {
        "choices": [{"message": {"content": "ok"}}]
    }

    task = _task(
        "fast-agent",
        tool_results=[
            ToolResult(
                tool_call_id="call_1",
                name="run_shell",
                content="README.md",
                is_error=False,
                source_protocol="openai_responses",
                raw={},
            )
        ],
    )
    task.vision_evidence.append(_evidence())

    action = await core.execute(task)

    assert action.type == "final_answer"
    messages = core.ollama.chat_completion.call_args.kwargs["messages"]
    assert "[VisionEvidence #1]" in messages[2]["content"]
    assert "AUTHORITATIVE_TOOL_RESULT" in messages[3]["content"]


@pytest.mark.asyncio
async def test_vision_pro_without_tools_returns_evidence_only():
    core = FusionCore(FusionConfig())
    core.ollama = MagicMock()
    core.ds4 = MagicMock()
    core.vision.extract_evidence = AsyncMock(return_value=[_evidence()])

    action = await core.execute(_task("vision-pro", images=[_image()]))

    assert action.type == "final_answer"
    assert "A screenshot with a Save button" in action.text
    core.ollama.chat_completion.assert_not_called()


@pytest.mark.asyncio
async def test_vision_pro_with_tools_routes_evidence_to_coder_agent():
    core = FusionCore(FusionConfig())
    core.ollama = MagicMock()
    core.ds4 = MagicMock()
    core.vision.extract_evidence = AsyncMock(return_value=[_evidence()])
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

    task = _task(
        "vision-pro",
        images=[_image()],
        tools=[
            ToolSpec(
                name="read_file",
                description="Read",
                input_schema={"type": "object"},
                source_protocol="openai_responses",
                raw={},
            )
        ],
    )

    action = await core.execute(task)

    assert action.type == "tool_call"
    assert action.tool_name == "read_file"
    assert (
        "[VisionEvidence #1]"
        in core.ollama.chat_completion.call_args.kwargs["messages"][2]["content"]
    )


@pytest.mark.asyncio
async def test_vision_heavy_extracts_evidence_and_unloads_before_ds4():
    cfg = FusionConfig(
        est_model_heavy_gb=20,
        est_model_coder_gb=10,
        est_model_critic_gb=10,
        est_model_judge_gb=10,
    )
    core = FusionCore(cfg)
    core.ollama = MagicMock()
    core.ds4 = MagicMock()
    core.vision.extract_evidence = AsyncMock(return_value=[_evidence()])
    core.ds4.health.return_value = True
    core.ds4.chat_completion.return_value = {
        "choices": [{"message": {"content": "ds4 primary answer"}}]
    }
    core.ollama.chat_completion.side_effect = [
        {"choices": [{"message": {"content": "checker ok"}}]},
        {"choices": [{"message": {"content": "critic ok"}}]},
        {
            "choices": [
                {"message": {"content": '{"type": "final_answer", "text": "approved"}'}}
            ]
        },
    ]

    action = await core.execute(_task("vision-heavy", images=[_image()]))

    assert action.type == "final_answer"
    assert core.vision.extract_evidence.await_count == 1
    core.ollama.unload_models.assert_called()
    unload_models = core.ollama.unload_models.call_args.args[0]
    assert cfg.model_vision_pro in unload_models
    assert cfg.model_agentic_pro in unload_models
    ds4_messages = core.ds4.chat_completion.call_args.kwargs["messages"]
    assert any("[VisionEvidence #1]" in message["content"] for message in ds4_messages)


@pytest.mark.asyncio
async def test_agentic_and_hermes_pro_use_qwen36_profiles():
    cfg = FusionConfig()
    core = FusionCore(cfg)
    core.ollama = MagicMock()
    core.ds4 = MagicMock()
    core.ollama.chat_completion.return_value = {
        "choices": [{"message": {"content": "pro answer"}}]
    }

    agentic = await core.execute(_task("agentic-pro"))
    hermes = await core.execute(_task("hermes-pro"))

    assert agentic.type == "final_answer"
    assert hermes.type == "final_answer"
    calls = core.ollama.chat_completion.call_args_list
    assert calls[0].kwargs["model"] == cfg.model_agentic_pro
    assert calls[1].kwargs["model"] == cfg.model_hermes_pro


@pytest.mark.asyncio
async def test_optional_mlx_profiles_use_optional_models_without_changing_defaults():
    cfg = FusionConfig()
    core = FusionCore(cfg)
    core.ollama = MagicMock()
    core.ds4 = MagicMock()
    core.ollama.chat_completion.return_value = {
        "choices": [{"message": {"content": "mlx answer"}}]
    }

    agentic_mlx = await core.execute(_task("agentic-mlx"))
    formatter_mlx = await core.execute(_task("formatter-mlx"))

    assert cfg.model_agentic_pro == "qwen/qwen3.6-35b-a3b"
    assert cfg.model_formatter == "google/gemma-4-26b-a4b-qat"
    assert agentic_mlx.trace["profile"] == "agentic-mlx"
    assert formatter_mlx.trace["profile"] == "formatter-mlx"
    calls = core.ollama.chat_completion.call_args_list
    assert calls[0].kwargs["model"] == cfg.model_agentic_mlx
    assert calls[1].kwargs["model"] == cfg.model_formatter_mlx


def test_agentic_mlx_has_dedicated_256k_context_limit():
    """agentic-mlx must use its own 256K context ceiling, not shared with vision-pro (96K).

    qwen3.6:35b-a3b-nvfp4 (Ollama) reports context_length=262144, so 256000 chars
    is a safe upper bound for input characters.
    """
    cfg = FusionConfig()
    core = FusionCore(cfg)
    # New dedicated setting
    assert cfg.agentic_mlx_max_input_chars == 256000
    # Must exceed vision-pro / full-agent ceilings
    assert cfg.agentic_mlx_max_input_chars > cfg.vision_pro_max_input_chars
    assert cfg.agentic_mlx_max_input_chars > cfg.full_max_input_chars
    # Routing must use the dedicated value
    assert core._context_limit_for_profile("agentic-mlx") == 256000


def test_agentic_mlx_max_tokens_default_is_600_for_thinking_model():
    """qwen3.6:35b-a3b-nvfp4 is a reasoning/thinking model; chain-of-thought
    eats ~500 tokens, so the default output budget must be >= 600 to leave
    room for actual content. The previous default of 1800 was safe but wasteful
    (most of the budget went to thinking). 600 is the minimum for a non-empty
    content reply on a trivial PONG-style prompt.
    """
    cfg = FusionConfig()
    assert cfg.agentic_mlx_max_tokens == 600
    assert cfg.agentic_mlx_max_tokens >= 600  # hard floor for thinking model


def test_fast_max_tokens_default_keeps_thinking_reserve():
    """fast/chat/formatter-mlx profiles now route to MLX thinking models
    (gemma4:12b-mlx, qwen3.6:35b-a3b-nvfp4). Each eats ~500 tokens for
    chain-of-thought before producing any content. The default output
    budget must leave at least ~1000 tokens for actual content even if a
    client forgets to set max_tokens.
    """
    cfg = FusionConfig()
    # Bumped from 1200 -> 1500: gives 500 thinking + 1000 content margin.
    assert cfg.fast_max_tokens == 1500
    # Sanity: must clear the thinking-model floor by a comfortable margin
    assert cfg.fast_max_tokens >= 1500  # ~500 thinking + ~1000 content
    # formatter-mlx profile should also be safe (it shares fast_max_tokens)
    assert cfg.fast_max_tokens >= 600  # never below the agentic-mlx floor


@pytest.mark.asyncio
async def test_agentic_mlx_accepts_inputs_up_to_256k_chars():
    """Inputs at the new 256K ceiling must NOT trip the context_limit_exceeded path."""
    cfg = FusionConfig()
    core = FusionCore(cfg)
    core.ollama = MagicMock()
    core.ds4 = MagicMock()
    core.ollama.chat_completion.return_value = {
        "choices": [{"message": {"content": "ok"}}]
    }

    # 200K chars - would have been rejected under the old 96K shared limit
    big_text = "a" * 200_000
    task = ParsedAgentTask(
        text=big_text,
        tools=[],
        tool_results=[],
        profile="agentic-mlx",
        source_protocol="openai_responses",
        stream=False,
        raw_request={},
    )

    action = await core.execute(task)

    assert action.trace is None or action.trace.get("error") != "context_limit_exceeded"
    assert action.type == "final_answer"
    assert "context limit exceeded" not in action.text
    core.ollama.chat_completion.assert_called_once()
    # It must have routed to qwen3.6 NVFP4
    assert (
        core.ollama.chat_completion.call_args.kwargs["model"] == cfg.model_agentic_mlx
    )


@pytest.mark.asyncio
async def test_agentic_mlx_rejects_inputs_above_256k():
    """Inputs above the new ceiling must still trip the context_limit_exceeded path
    with a clear, profile-specific error message."""
    cfg = FusionConfig(agentic_mlx_max_input_chars=1000)  # tight override for test
    core = FusionCore(cfg)
    core.ollama = MagicMock()
    core.ds4 = MagicMock()

    task = ParsedAgentTask(
        text="x" * 2000,
        tools=[],
        tool_results=[],
        profile="agentic-mlx",
        source_protocol="openai_responses",
        stream=False,
        raw_request={},
    )

    action = await core.execute(task)

    assert action.type == "final_answer"
    assert "context limit exceeded" in action.text
    assert "agentic-mlx" in action.text
    assert action.trace["error"] == "context_limit_exceeded"
    assert action.trace["limit_chars"] == 1000
    core.ollama.chat_completion.assert_not_called()

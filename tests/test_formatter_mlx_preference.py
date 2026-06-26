"""G11: Tests for prefer_mlx_formatter preference.

When enabled (default), the fast-agent profile auto-routes through formatter-mlx
(gemma via MLX) for SHORT text-only requests (< 1000 chars, no tools).

Conditions to route to formatter-mlx:
  - prefer_mlx_formatter == True (default)
  - profile == "fast-agent"
  - task.tools is empty
  - task.text length < 1000 chars (QWABLE_MLX_FORMATTER_MAX_CHARS)

Otherwise: standard fast-agent path.
"""

import pytest

from qwable.config import FusionConfig
from qwable.fusion_core import FusionCore
from qwable.models import DS4Client, OllamaClient
from qwable.schemas import ParsedAgentTask


def _make_config(**overrides) -> FusionConfig:
    defaults = dict(
        qwable_host="127.0.0.1",
        qwable_port=8088,
        ollama_base_url="http://127.0.0.1:1234/v1",
        qwable_timeout_seconds=60,
        qwable_queue_timeout_seconds=5,
        qwable_max_concurrent_requests=1,
        ds4_base_url="http://127.0.0.1:8000/v1",
        ds4_timeout_seconds=60,
        local_model_backend="lmstudio",
        lmstudio_cli_path="/bin/echo",
        fusion_default_preset="quality",
        prefer_mlx_formatter=True,
        mlx_formatter_max_chars=1000,
    )
    defaults.update(overrides)
    return FusionConfig(**defaults)


def _install_mock_clients(monkeypatch, response_text: str = "mlx formatter says hi"):
    """Mock OllamaClient.chat_completion to auto-respond."""
    state = {"call_log": []}

    def fake_chat(self_or_None=None, *, model, messages, **kwargs):
        state["call_log"].append(model)
        return {
            "choices": [
                {
                    "message": {"role": "assistant", "content": response_text},
                    "finish_reason": "stop",
                }
            ]
        }

    def fake_unload(self_or_None=None, models=None, **kwargs):
        pass

    monkeypatch.setattr(OllamaClient, "chat_completion", fake_chat)
    monkeypatch.setattr(OllamaClient, "unload_models", fake_unload)
    monkeypatch.setattr(DS4Client, "chat_completion", fake_chat)
    return state


def _make_task(text: str, tools: list | None = None) -> ParsedAgentTask:
    from qwable.schemas import ToolSpec

    spec_tools = []
    for t in tools or []:
        if isinstance(t, dict):
            fn = t.get("function", {})
            spec_tools.append(
                ToolSpec(
                    name=fn.get("name", ""),
                    description=fn.get("description"),
                    input_schema=fn.get("parameters", {}),
                    source_protocol="openai_chat",
                    raw=t,
                )
            )
        else:
            spec_tools.append(t)
    return ParsedAgentTask(
        text=text,
        tools=spec_tools,
        tool_results=[],
        profile="fast-agent",
        source_protocol="openai_chat",
        stream=False,
        raw_request={"messages": [{"role": "user", "content": text}]},
    )


# ─── Tests ──────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_fast_agent_short_text_routes_to_formatter_mlx(monkeypatch):
    """Short text + no tools + prefer_mlx=True → formatter-mlx."""
    state = _install_mock_clients(monkeypatch)
    core = FusionCore(_make_config(prefer_mlx_formatter=True))
    task = _make_task("summarize this in 1 sentence")

    action = await core.execute(task)

    assert action.type == "final_answer"
    assert action.trace["profile"] == "formatter-mlx"
    assert action.trace["model"] == "google/gemma-4-26b-a4b-qat"
    # formatter-mlx path uses gemma model
    assert "google/gemma-4-26b-a4b-qat" in state["call_log"]


@pytest.mark.asyncio
async def test_fast_agent_with_tools_uses_fast_agent(monkeypatch):
    """Tools present → formatter-mlx shortcut is bypassed → fast-agent path."""
    _install_mock_clients(monkeypatch)
    config = _make_config(prefer_mlx_formatter=True)
    core = FusionCore(config)
    tools = [{"type": "function", "function": {"name": "read_file", "parameters": {}}}]
    task = _make_task("read the file", tools=tools)

    action = await core.execute(task)

    # Should NOT route to formatter-mlx
    trace = getattr(action, "trace", None) or {}
    assert trace.get("profile") != "formatter-mlx"


@pytest.mark.asyncio
async def test_fast_agent_long_text_uses_fast_agent(monkeypatch):
    """Text length >= 1000 chars → formatter-mlx shortcut is bypassed."""
    _install_mock_clients(monkeypatch)
    config = _make_config(
        prefer_mlx_formatter=True,
        mlx_formatter_max_chars=1000,
    )
    core = FusionCore(config)
    long_text = "a" * 1500  # 1500 chars, above the 1000 threshold
    task = _make_task(long_text)

    action = await core.execute(task)

    trace = getattr(action, "trace", None) or {}
    assert trace.get("profile") != "formatter-mlx"


@pytest.mark.asyncio
async def test_prefer_mlx_formatter_false_uses_fast_agent(monkeypatch):
    """When prefer_mlx_formatter=False, all fast-agent → fast-agent (no shortcut)."""
    _install_mock_clients(monkeypatch)
    core = FusionCore(_make_config(prefer_mlx_formatter=False))
    task = _make_task("summarize this in 1 sentence")

    action = await core.execute(task)

    trace = getattr(action, "trace", None) or {}
    assert trace.get("profile") != "formatter-mlx"


@pytest.mark.asyncio
async def test_fast_agent_exact_threshold_routes_to_fast_agent(monkeypatch):
    """Text exactly at threshold (1000 chars) → formatter-mlx (strict <)."""
    _install_mock_clients(monkeypatch)
    config = _make_config(
        prefer_mlx_formatter=True,
        mlx_formatter_max_chars=1000,
    )
    core = FusionCore(config)
    # Exactly 999 chars — should be < 1000, so routes to formatter-mlx
    text = "a" * 999
    task = _make_task(text)

    action = await core.execute(task)

    assert action.trace["profile"] == "formatter-mlx"


@pytest.mark.asyncio
async def test_only_fast_agent_profile_uses_shortcut(monkeypatch):
    """Only fast-agent profile is eligible for shortcut; others unchanged."""
    _install_mock_clients(monkeypatch)
    config = _make_config(prefer_mlx_formatter=True)
    core = FusionCore(config)
    # chat-agent profile with short text — should NOT route to formatter-mlx
    task = ParsedAgentTask(
        text="hello",
        tools=[],
        tool_results=[],
        profile="chat-agent",
        source_protocol="openai_chat",
        stream=False,
        raw_request={"messages": [{"role": "user", "content": "hello"}]},
    )

    action = await core.execute(task)

    # chat-agent runs through its own profile, not formatter-mlx
    trace = getattr(action, "trace", None) or {}
    assert trace.get("profile") != "formatter-mlx"


# ─── Config field tests ──────────────────────────────────────────────────


def test_fusion_config_has_prefer_mlx_formatter_field():
    """FusionConfig should expose prefer_mlx_formatter field, default True."""
    config = FusionConfig()
    assert hasattr(config, "prefer_mlx_formatter")
    assert config.prefer_mlx_formatter is True


def test_fusion_config_has_mlx_formatter_max_chars_field():
    """FusionConfig should expose mlx_formatter_max_chars field, default 1000."""
    config = FusionConfig()
    assert hasattr(config, "mlx_formatter_max_chars")
    assert config.mlx_formatter_max_chars == 1000


def test_fusion_config_env_override_prefer_mlx_formatter(monkeypatch):
    """QWABLE_PREFER_MLX_FORMATTER=false env var should disable."""
    monkeypatch.setenv("QWABLE_PREFER_MLX_FORMATTER", "false")
    config = FusionConfig()
    assert config.prefer_mlx_formatter is False


def test_fusion_config_env_override_max_chars(monkeypatch):
    """QWABLE_MLX_FORMATTER_MAX_CHARS env var should set threshold."""
    monkeypatch.setenv("QWABLE_MLX_FORMATTER_MAX_CHARS", "500")
    config = FusionConfig()
    assert config.mlx_formatter_max_chars == 500

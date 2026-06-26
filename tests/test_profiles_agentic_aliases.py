"""Tests for v1.7 workflow profile aliases."""

from qwable.profiles import resolve_profile


def test_openai_responses_agent_aliases():
    assert resolve_profile("qwable-agent", "openai_responses") == "agentic-workflow"
    assert resolve_profile("qwable-code-agent", "openai_responses") == "coding-workflow"
    assert (
        resolve_profile("qwable-review-agent", "openai_responses") == "review-workflow"
    )


def test_anthropic_agent_aliases():
    assert (
        resolve_profile("claude-qwable-agent", "anthropic_messages")
        == "agentic-workflow"
    )
    assert (
        resolve_profile("claude-qwable-code-agent", "anthropic_messages")
        == "coding-workflow"
    )
    assert (
        resolve_profile("claude-qwable-review-agent", "anthropic_messages")
        == "review-workflow"
    )


def test_openai_chat_agent_aliases():
    assert resolve_profile("qwable-agent", "openai_chat") == "agentic-workflow"
    assert resolve_profile("qwable-code-agent", "openai_chat") == "coding-workflow"
    assert resolve_profile("qwable-review-agent", "openai_chat") == "review-workflow"


def test_existing_mlx_aliases_remain_explicit_profiles():
    assert resolve_profile("qwable-agentic-mlx", "openai_responses") == "agentic-mlx"
    assert (
        resolve_profile("qwable-formatter-mlx", "openai_responses") == "formatter-mlx"
    )
    assert (
        resolve_profile("claude-qwable-agentic-mlx", "anthropic_messages")
        == "agentic-mlx"
    )
    assert (
        resolve_profile("claude-qwable-formatter-mlx", "anthropic_messages")
        == "formatter-mlx"
    )
    assert resolve_profile("qwable-agentic-mlx", "openai_chat") == "agentic-mlx"
    assert resolve_profile("qwable-formatter-mlx", "openai_chat") == "formatter-mlx"

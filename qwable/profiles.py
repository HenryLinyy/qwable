"""Agent profile definitions and mapping."""

from typing import Literal

ProfileName = Literal[
    "fast-agent",
    "full-agent",
    "heavy-agent",
    "chat-agent",
    "vision-fast",
    "vision-pro",
    "vision-heavy",
    "agentic-pro",
    "hermes-pro",
    "agentic-mlx",
    "formatter-mlx",
    "fusion-agent",
    "agentic-workflow",
    "coding-workflow",
    "review-workflow",
]

PROFILE_MAP_OPENAI_RESPONSES: dict[str, ProfileName] = {
    "qwable": "fast-agent",
    "qwable-fast": "fast-agent",
    "qwable-full": "full-agent",
    "qwable-heavy": "heavy-agent",
    "qwable-vision-fast": "vision-fast",
    "qwable-vision-pro": "vision-pro",
    "qwable-vision-heavy": "vision-heavy",
    "qwable-agentic-pro": "agentic-pro",
    "qwable-hermes-pro": "hermes-pro",
    "qwable-agentic-mlx": "agentic-mlx",
    "qwable-formatter-mlx": "formatter-mlx",
    "qwable-fusion": "fusion-agent",
    "qwable-fusion-budget": "fusion-agent",
    "qwable-fusion-quality": "fusion-agent",
    "qwable-fusion-coding": "fusion-agent",
    "qwable-fusion-heavy": "fusion-agent",
    "qwable-agent": "agentic-workflow",
    "qwable-code-agent": "coding-workflow",
    "qwable-review-agent": "review-workflow",
}

PROFILE_MAP_ANTHROPIC_MESSAGES: dict[str, ProfileName] = {
    "claude-qwable": "fast-agent",
    "claude-qwable-fast": "fast-agent",
    "claude-qwable-full": "full-agent",
    "claude-qwable-heavy": "heavy-agent",
    "claude-qwable-vision-fast": "vision-fast",
    "claude-qwable-vision-pro": "vision-pro",
    "claude-qwable-vision-heavy": "vision-heavy",
    "claude-qwable-agentic-pro": "agentic-pro",
    "claude-qwable-hermes-pro": "hermes-pro",
    "claude-qwable-agentic-mlx": "agentic-mlx",
    "claude-qwable-formatter-mlx": "formatter-mlx",
    "claude-qwable-fusion": "fusion-agent",
    "claude-qwable-fusion-budget": "fusion-agent",
    "claude-qwable-fusion-quality": "fusion-agent",
    "claude-qwable-fusion-coding": "fusion-agent",
    "claude-qwable-fusion-heavy": "fusion-agent",
    "claude-qwable-agent": "agentic-workflow",
    "claude-qwable-code-agent": "coding-workflow",
    "claude-qwable-review-agent": "review-workflow",
}

PROFILE_MAP_OPENAI_CHAT: dict[str, ProfileName] = {
    "qwable-chat": "chat-agent",
    "qwable-fast": "fast-agent",
    "qwable-full": "full-agent",
    "qwable-heavy": "heavy-agent",
    "qwable-vision-fast": "vision-fast",
    "qwable-vision-pro": "vision-pro",
    "qwable-vision-heavy": "vision-heavy",
    "qwable-agentic-pro": "agentic-pro",
    "qwable-hermes-pro": "hermes-pro",
    "qwable-agentic-mlx": "agentic-mlx",
    "qwable-formatter-mlx": "formatter-mlx",
    "qwable-fusion": "fusion-agent",
    "qwable-fusion-budget": "fusion-agent",
    "qwable-fusion-quality": "fusion-agent",
    "qwable-fusion-coding": "fusion-agent",
    "qwable-fusion-heavy": "fusion-agent",
    "qwable-agent": "agentic-workflow",
    "qwable-code-agent": "coding-workflow",
    "qwable-review-agent": "review-workflow",
}


def resolve_profile(
    model_name: str,
    protocol: Literal["openai_responses", "anthropic_messages", "openai_chat"],
) -> ProfileName:
    """Map a model name to the corresponding agent profile."""
    if protocol == "openai_responses":
        return PROFILE_MAP_OPENAI_RESPONSES.get(model_name, "fast-agent")
    elif protocol == "anthropic_messages":
        return PROFILE_MAP_ANTHROPIC_MESSAGES.get(model_name, "fast-agent")
    elif protocol == "openai_chat":
        return PROFILE_MAP_OPENAI_CHAT.get(model_name, "chat-agent")
    return "fast-agent"

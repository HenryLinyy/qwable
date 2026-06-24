"""G10: Verify raw_request is preserved across all 3 protocol parsers.

Fusion mode reads fusion config from task.raw_request, so this flow must
work for OpenAI Chat, OpenAI Responses, and Anthropic Messages.

NOTE: All three parsers take only `body` — the profile is resolved later by
`resolve_profile(body["model"], protocol)`.
"""

from qwable.message_parsing import (
    parse_anthropic_messages_input,
    parse_openai_chat_input,
    parse_openai_responses_input,
)


def test_openai_chat_preserves_raw_request():
    body = {
        "model": "qwable-fusion",
        "messages": [{"role": "user", "content": "hello"}],
        "fusion": {"preset": "quality"},
    }
    task = parse_openai_chat_input(body)
    assert task.raw_request is body


def test_openai_responses_preserves_raw_request():
    body = {
        "model": "qwable-fusion",
        "input": "hello",
        "fusion": {"preset": "budget"},
    }
    task = parse_openai_responses_input(body)
    assert task.raw_request is body


def test_anthropic_messages_preserves_raw_request():
    body = {
        "model": "claude-qwable-fusion",
        "messages": [{"role": "user", "content": "hello"}],
        "max_tokens": 1024,
        "fusion": {"preset": "coding"},
    }
    task = parse_anthropic_messages_input(body)
    assert task.raw_request is body


def test_openai_chat_preserves_plugins_shape():
    """OpenRouter-style plugins shape must survive in raw_request."""
    body = {
        "model": "qwable-fusion",
        "messages": [{"role": "user", "content": "hello"}],
        "plugins": [{"id": "fusion", "preset": "quality"}],
    }
    task = parse_openai_chat_input(body)
    assert task.raw_request["plugins"] == [{"id": "fusion", "preset": "quality"}]


def test_openai_responses_preserves_plugins_shape():
    body = {
        "model": "qwable-fusion",
        "input": "hello",
        "plugins": [{"id": "fusion", "preset": "coding"}],
    }
    task = parse_openai_responses_input(body)
    assert task.raw_request["plugins"] == [{"id": "fusion", "preset": "coding"}]

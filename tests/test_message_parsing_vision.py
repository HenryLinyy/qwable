"""Tests for multimodal image input parsing."""

from qwable.message_parsing import (
    parse_anthropic_messages_input,
    parse_openai_chat_input,
    parse_openai_responses_input,
)


DATA_URL = "data:image/png;base64,aW1hZ2UtYnl0ZXM="


def test_openai_responses_parses_input_image_and_preserves_tool_output():
    body = {
        "model": "qwable-vision-pro",
        "input": [
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": "OCR this screenshot"},
                    {"type": "input_image", "image_url": DATA_URL, "detail": "high"},
                ],
            },
            {
                "type": "function_call_output",
                "call_id": "call_1",
                "name": "run_shell",
                "output": "README.md",
            },
        ],
    }

    task = parse_openai_responses_input(body)

    assert task.profile == "vision-pro"
    assert "OCR this screenshot" in task.text
    assert len(task.images) == 1
    assert task.images[0].mime_type == "image/png"
    assert task.images[0].data_base64 == "aW1hZ2UtYnl0ZXM="
    assert task.images[0].detail == "high"
    assert len(task.tool_results) == 1
    assert task.tool_results[0].content == "README.md"


def test_openai_chat_parses_image_url_block():
    body = {
        "model": "qwable-vision-fast",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe the UI"},
                    {"type": "image_url", "image_url": {"url": DATA_URL, "detail": "low"}},
                ],
            }
        ],
    }

    task = parse_openai_chat_input(body)

    assert task.profile == "vision-fast"
    assert "[user]: Describe the UI" in task.text
    assert len(task.images) == 1
    assert task.images[0].source_protocol == "openai_chat"
    assert task.images[0].detail == "low"


def test_anthropic_messages_parses_base64_image_block():
    body = {
        "model": "claude-qwable-vision-pro",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Find button labels"},
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": "aW1hZ2UtYnl0ZXM=",
                        },
                    },
                ],
            }
        ],
    }

    task = parse_anthropic_messages_input(body)

    assert task.profile == "vision-pro"
    assert "[user]: Find button labels" in task.text
    assert len(task.images) == 1
    assert task.images[0].source_protocol == "anthropic_messages"
    assert task.images[0].mime_type == "image/png"
    assert task.images[0].data_base64 == "aW1hZ2UtYnl0ZXM="

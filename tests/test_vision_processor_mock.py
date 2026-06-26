"""Tests for VisionProcessor request shape."""

from unittest.mock import MagicMock

import pytest

from qwable.config import FusionConfig
from qwable.schemas import ParsedAgentTask
from qwable.vision import ImageInput
from qwable.vision_processor import VisionProcessor


def _task() -> ParsedAgentTask:
    return ParsedAgentTask(
        text="OCR this image",
        tools=[],
        tool_results=[],
        profile="vision-pro",
        source_protocol="openai_responses",
        stream=False,
        raw_request={},
        images=[
            ImageInput(
                source_protocol="openai_responses",
                mime_type="image/png",
                data_base64="aW1hZ2U=",
                url=None,
                local_path=None,
                detail="auto",
                raw={},
            )
        ],
    )


@pytest.mark.asyncio
async def test_vision_processor_uses_native_chat_with_thinking_disabled():
    cfg = FusionConfig()
    ollama = MagicMock()
    ollama.native_chat_completion.return_value = {
        "message": {
            "content": "Summary: UI screenshot\nVisible Text: Save\nConfidence: 0.8"
        }
    }
    processor = VisionProcessor(cfg, ollama)

    evidence = await processor.extract_evidence(_task(), "vision-pro")

    assert len(evidence) == 1
    assert evidence[0].model == cfg.model_vision_pro
    assert evidence[0].profile == "vision-pro"
    assert "UI screenshot" in evidence[0].raw_text
    kwargs = ollama.native_chat_completion.call_args.kwargs
    assert kwargs["model"] == cfg.model_vision_pro
    assert kwargs["think"] is False
    assert kwargs["messages"][1]["images"] == ["aW1hZ2U="]


@pytest.mark.asyncio
async def test_vision_processor_warns_without_inline_base64_download():
    cfg = FusionConfig()
    ollama = MagicMock()
    processor = VisionProcessor(cfg, ollama)
    task = _task()
    task.images[0].data_base64 = None
    task.images[0].url = "https://example.com/image.png"

    evidence = await processor.extract_evidence(task, "vision-pro")

    assert len(evidence) == 1
    assert evidence[0].warnings
    assert "inline base64" in evidence[0].warnings[0]
    ollama.native_chat_completion.assert_not_called()

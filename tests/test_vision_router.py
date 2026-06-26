"""Tests for vision profile routing."""

from qwable.schemas import ParsedAgentTask
from qwable.vision import ImageInput
from qwable.vision_router import select_vision_profile


def _task(text: str, profile: str = "fast-agent", images: int = 1) -> ParsedAgentTask:
    return ParsedAgentTask(
        text=text,
        tools=[],
        tool_results=[],
        profile=profile,
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
            for _ in range(images)
        ],
    )


def test_no_images_returns_none():
    assert select_vision_profile(_task("hello", images=0)) is None


def test_explicit_vision_profile_is_respected():
    assert select_vision_profile(_task("hello", profile="vision-pro")) == "vision-pro"


def test_heavy_with_images_uses_two_stage_vision_heavy():
    assert (
        select_vision_profile(_task("analyze", profile="heavy-agent")) == "vision-heavy"
    )


def test_ui_ocr_keywords_use_vision_pro():
    assert (
        select_vision_profile(_task("OCR this UI screenshot and list buttons"))
        == "vision-pro"
    )


def test_many_images_use_vision_pro():
    assert select_vision_profile(_task("summarize", images=5)) == "vision-pro"


def test_simple_image_prompt_uses_vision_fast():
    assert select_vision_profile(_task("what is in this image")) == "vision-fast"

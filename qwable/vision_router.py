"""Routing helpers for image-bearing requests."""

from __future__ import annotations

from qwable.schemas import ParsedAgentTask


VISION_PRO_KEYWORDS = (
    "ui",
    "screenshot",
    "screen shot",
    "截圖",
    "ocr",
    "表格",
    "圖表",
    "chart",
    "diagram",
    "mockup",
    "html",
    "css",
    "visual coding",
    "設計稿",
    "看錯誤畫面",
    "錯誤畫面",
)

VISION_PROFILES = {"vision-fast", "vision-pro", "vision-heavy"}


def select_vision_profile(task: ParsedAgentTask) -> str | None:
    """Select a vision profile for a task, or None when no vision pass is needed."""
    if not task.images:
        return None

    if task.profile in VISION_PROFILES:
        return task.profile

    if task.profile == "heavy-agent":
        return "vision-heavy"

    text = (task.text or "").lower()
    if len(task.images) > 4:
        return "vision-pro"
    if any(keyword in text for keyword in VISION_PRO_KEYWORDS):
        return "vision-pro"
    return "vision-fast"

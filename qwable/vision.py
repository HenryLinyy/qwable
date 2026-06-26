"""Vision input and evidence helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal
import re


SourceProtocol = Literal["openai_responses", "anthropic_messages", "openai_chat"]


@dataclass
class ImageInput:
    """Normalized image input from supported client protocols."""

    source_protocol: SourceProtocol
    mime_type: str | None
    data_base64: str | None
    url: str | None
    local_path: str | None
    detail: str | None
    raw: dict


@dataclass
class VisionEvidence:
    """Auditable visual evidence extracted before tool or heavy reasoning."""

    model: str
    profile: str
    summary: str
    visible_text: str | None = None
    ui_elements: list[dict] = field(default_factory=list)
    tables: list[dict] = field(default_factory=list)
    charts: list[dict] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    confidence: float | None = None
    raw_text: str = ""


# Accept inline base64 data URLs with an empty media type (data:;base64,...) and
# optional parameters (data:image/png;charset=utf-8;base64,...), per RFC 2397.
DATA_URL_RE = re.compile(
    r"^data:(?P<mime>[-\w.+/]*)(?:;[-\w.=]+)*;base64,(?P<data>.+)$", re.DOTALL
)


def image_from_url_value(
    source_protocol: SourceProtocol,
    value: str | None,
    *,
    detail: str | None = None,
    raw: dict | None = None,
) -> ImageInput | None:
    """Convert a data URL or remote URL into an ImageInput."""
    if not isinstance(value, str) or not value:
        return None

    match = DATA_URL_RE.match(value)
    if match:
        return ImageInput(
            source_protocol=source_protocol,
            mime_type=match.group("mime") or "application/octet-stream",
            data_base64=match.group("data").strip(),
            url=None,
            local_path=None,
            detail=detail or "auto",
            raw=raw or {},
        )

    return ImageInput(
        source_protocol=source_protocol,
        mime_type=None,
        data_base64=None,
        url=value,
        local_path=None,
        detail=detail or "auto",
        raw=raw or {},
    )


def image_from_base64_source(
    source_protocol: SourceProtocol,
    *,
    data_base64: str | None,
    mime_type: str | None,
    detail: str | None = None,
    raw: dict | None = None,
) -> ImageInput | None:
    """Convert a protocol-native base64 image source into an ImageInput."""
    if not isinstance(data_base64, str) or not data_base64:
        return None
    return ImageInput(
        source_protocol=source_protocol,
        mime_type=mime_type,
        data_base64=data_base64.strip(),
        url=None,
        local_path=None,
        detail=detail or "auto",
        raw=raw or {},
    )


def estimate_base64_size_mb(data_base64: str) -> float:
    """Estimate decoded image bytes from base64 without decoding large payloads."""
    compact = "".join(data_base64.split())
    padding = compact.count("=")
    decoded_bytes = max(0, (len(compact) * 3 // 4) - padding)
    return decoded_bytes / (1024 * 1024)


def format_vision_evidence(evidence: VisionEvidence, index: int) -> str:
    """Format evidence as an auditable text block for downstream agents."""
    parts = [
        f"[VisionEvidence #{index}]",
        f"model: {evidence.model}",
        f"profile: {evidence.profile}",
        "Summary:",
        evidence.summary or "",
    ]
    if evidence.visible_text:
        parts.extend(["Visible Text:", evidence.visible_text])
    if evidence.ui_elements:
        parts.extend(["UI Elements:", str(evidence.ui_elements)])
    if evidence.tables:
        parts.extend(["Tables:", str(evidence.tables)])
    if evidence.charts:
        parts.extend(["Charts:", str(evidence.charts)])
    if evidence.warnings:
        parts.extend(["Warnings:", str(evidence.warnings)])
    if evidence.confidence is not None:
        parts.extend(["Confidence:", str(evidence.confidence)])
    if (
        evidence.raw_text
        and evidence.raw_text.strip() != (evidence.summary or "").strip()
    ):
        parts.extend(["Raw Evidence:", evidence.raw_text.strip()])
    parts.append("[/VisionEvidence]")
    return "\n".join(parts)

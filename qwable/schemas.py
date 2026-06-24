"""Core data structures for Qwable Agent Gateway."""

from dataclasses import dataclass
from typing import Literal, Optional
from qwable.vision import ImageInput, VisionEvidence


@dataclass
class ToolSpec:
    """Normalized tool specification from any protocol."""
    name: str
    description: str | None
    input_schema: dict
    source_protocol: Literal["openai_responses", "anthropic_messages", "openai_chat"]
    raw: dict


@dataclass
class ToolResult:
    """Normalized tool result from any protocol."""
    tool_call_id: str | None
    name: str | None
    content: str
    is_error: bool
    source_protocol: Literal["openai_responses", "anthropic_messages", "openai_chat"]
    raw: dict


@dataclass
class ParsedAgentTask:
    """Parsed incoming request into a unified internal format."""
    text: str
    tools: list[ToolSpec]
    tool_results: list[ToolResult]
    profile: str
    source_protocol: Literal["openai_responses", "anthropic_messages", "openai_chat"]
    stream: bool
    raw_request: dict
    images: list[ImageInput] = None
    vision_evidence: list[VisionEvidence] = None

    def __post_init__(self):
        if self.images is None:
            self.images = []
        if self.vision_evidence is None:
            self.vision_evidence = []


@dataclass
class FusionAction:
    """Unified action output from fusion core."""
    type: Literal["final_answer", "tool_call"]
    text: str | None
    tool_name: str | None
    tool_input: dict | None
    confidence: float | None
    rationale_summary: str | None
    trace: dict | None = None

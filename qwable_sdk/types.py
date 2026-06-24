"""SDK public types: preset enum + result dataclass."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class FusionPresetName(str, Enum):
    """Built-in fusion deliberation presets.

    Matches the four built-in presets on the gateway side:
    - quality: 3 large models + qwen3.6 judge (~66GB peak)
    - budget: 2 light models + qwen3.6 judge (~38GB peak)
    - coding: 3 large models + coder judge (~66GB peak)
    - heavy: 2 large models + ds4 judge (~66GB + ds4)
    - custom: caller provides analysis_models + judge_model
    """

    QUALITY = "quality"
    BUDGET = "budget"
    CODING = "coding"
    HEAVY = "heavy"
    CUSTOM = "custom"


# Convenience alias
FusionPreset = FusionPresetName


@dataclass
class FusionResult:
    """Result of a non-streaming fusion deliberation call.

    Attributes:
        text: Final answer text (from parsed Final Answer section, or raw judge text)
        preset: Preset name used
        panel_models: List of panel model ids (in execution order)
        judge_model: Judge model id
        judge_backend: "ollama" or "ds4"
        structured: 5-section structured output (may be empty on fallback)
        trace: Full runner trace (latency_ms, panel responses, etc.)
        total_latency_ms: Total deliberation time
        had_fallback: True if judge returned non-structured output
    """

    text: str
    preset: str
    panel_models: list[str] = field(default_factory=list)
    judge_model: str = ""
    judge_backend: str = ""
    structured: dict = field(default_factory=dict)
    trace: dict = field(default_factory=dict)
    total_latency_ms: int = 0
    had_fallback: bool = False

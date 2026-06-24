"""G10: Fusion deliberation router schemas.

Dataclasses shared across the fusion deliberation pipeline:
  - FusionRequest: parsed caller override (preset / panel / judge)
  - PanelResponse: one analysis model response (success or error)
  - SynthesisInput: inputs to the judge synthesis step
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class FusionRequest:
    """Parsed fusion mode request from the caller.

    All fields optional; resolution order is:
    1. analysis_models override (custom panel)
    2. preset name
    3. config default preset
    """

    preset: Optional[str] = None
    analysis_models: Optional[list[str]] = None
    judge_model: Optional[str] = None


@dataclass
class PanelResponse:
    """Single analysis model response from the deliberation panel.

    `error` is set when the model call failed; `text` may be empty in that case.
    """

    model_id: str
    text: str
    finish_reason: str = "stop"
    latency_ms: int = 0
    error: Optional[str] = None


@dataclass
class SynthesisInput:
    """Inputs to the judge synthesis step."""

    original_prompt: str
    panel_responses: list[PanelResponse]
    preset_name: str

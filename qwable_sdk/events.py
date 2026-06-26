"""Fusion deliberation streaming event types (SDK side).

These dataclasses are returned by `QwableClient.fusion_chat_stream()`.
Each event corresponds to a `FusionStreamEvent` emitted by the gateway.

Event types (mirrors `qwable.streaming_events`):
  - panel_start  : panel model is about to be invoked
  - panel_token  : token (or chunk) emitted by a panel model (G12-1)
  - panel_done   : panel model finished
  - judge_start  : judge model is about to be invoked
  - judge_token  : token (or chunk) emitted by the judge
  - judge_done   : judge finished
  - final        : full structured synthesis + trace ready
  - error        : runner failed
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class PanelEvent:
    """Panel lifecycle event (start/done/token)."""

    event: str  # "panel_start" | "panel_token" | "panel_done"
    model_id: str
    index: int
    latency_ms: int = 0
    delta: str = ""  # for panel_token
    text_preview: str = ""  # for panel_done
    error: Optional[str] = None
    finish_reason: str = ""


@dataclass
class JudgeEvent:
    """Judge lifecycle event (start/done/token)."""

    event: str  # "judge_start" | "judge_token" | "judge_done"
    judge_model: str = ""
    judge_backend: str = ""  # "ollama" | "ds4"
    delta: str = ""  # for judge_token
    latency_ms: int = 0
    finish_reason: str = ""
    structured_had_fallback: bool = False


@dataclass
class FinalEvent:
    """Final synthesis event — the only event that carries full structured output."""

    text: str
    structured: dict  # {final_answer, consensus, contradictions, blind_spots, per_model_notes, had_fallback}
    trace: dict  # {preset, panel_responses, judge_model, judge_backend, total_latency_ms, ...}


@dataclass
class ErrorEvent:
    """Runner error event."""

    event: str  # always "error"
    detail: str = ""
    code: int = 0


@dataclass
class FusionEvent:
    """Union of all streaming event types.

    Discriminate on `event`:
      - panel_start/panel_token/panel_done → use `panel` attribute (PanelEvent)
      - judge_start/judge_token/judge_done → use `judge` attribute (JudgeEvent)
      - final → use `final` attribute (FinalEvent)
      - error → use `error` attribute (ErrorEvent)
    """

    event: str
    data: dict = field(default_factory=dict)
    panel: Optional[PanelEvent] = None
    judge: Optional[JudgeEvent] = None
    final: Optional[FinalEvent] = None
    error: Optional[ErrorEvent] = None

"""G11: Fusion deliberation streaming events.

SSE-style events emitted by `run_fusion_agent_streaming()` so the caller
sees deliberation progress live instead of waiting for the full response.

Event types:
  - panel_start  : a panel model is about to be invoked
  - panel_token  : a token (or chunk) was emitted by a panel model (G12-1)
  - panel_done   : a panel model finished (success or error)
  - judge_start  : judge model is about to be invoked
  - judge_token  : a token (or chunk) was emitted by the judge
  - judge_done   : judge finished, full text is in the next 'final' event
  - final        : full structured synthesis + trace ready
  - error        : runner failed (preset error, runner error, etc.)
  - agent_*      : v1.7 agent runtime progress events
"""

import json
from dataclasses import dataclass
from typing import Any

from qwable.agent_events import AGENT_EVENT_TYPES


# ─── Event type constants ────────────────────────────────────────────────

FUSION_STREAM_EVENT_PANEL_START = "panel_start"
FUSION_STREAM_EVENT_PANEL_DONE = "panel_done"
FUSION_STREAM_EVENT_PANEL_TOKEN = "panel_token"  # G12-1
FUSION_STREAM_EVENT_JUDGE_START = "judge_start"
FUSION_STREAM_EVENT_JUDGE_TOKEN = "judge_token"
FUSION_STREAM_EVENT_JUDGE_DONE = "judge_done"
FUSION_STREAM_EVENT_FINAL = "final"
FUSION_STREAM_EVENT_ERROR = "error"

FUSION_STREAM_EVENT_TYPES: list[str] = [
    FUSION_STREAM_EVENT_PANEL_START,
    FUSION_STREAM_EVENT_PANEL_DONE,
    FUSION_STREAM_EVENT_PANEL_TOKEN,
    FUSION_STREAM_EVENT_JUDGE_START,
    FUSION_STREAM_EVENT_JUDGE_TOKEN,
    FUSION_STREAM_EVENT_JUDGE_DONE,
    FUSION_STREAM_EVENT_FINAL,
    FUSION_STREAM_EVENT_ERROR,
]

AGENT_STREAM_EVENT_TYPES: list[str] = list(AGENT_EVENT_TYPES)
FUSION_STREAM_EVENT_TYPES.extend(AGENT_STREAM_EVENT_TYPES)


# ─── Event dataclass ─────────────────────────────────────────────────────


@dataclass
class FusionStreamEvent:
    """Single SSE event in the fusion deliberation stream.

    `event`: one of FUSION_STREAM_EVENT_TYPES
    `data`: free-form dict — caller decides schema per event type
    """

    event: str
    data: dict[str, Any]


# ─── SSE formatter ──────────────────────────────────────────────────────


def format_fusion_sse(ev: FusionStreamEvent) -> str:
    """Format a FusionStreamEvent as a Server-Sent Events chunk.

    Output shape (per SSE spec):
        event: <event>\\n
        data: <json>\\n
        \\n

    Returns the formatted chunk (caller is responsible for streaming it).
    """
    if ev.event not in FUSION_STREAM_EVENT_TYPES:
        raise ValueError(f"unknown fusion stream event type: {ev.event!r}")
    payload = json.dumps(ev.data, ensure_ascii=False)
    return f"event: {ev.event}\ndata: {payload}\n\n"

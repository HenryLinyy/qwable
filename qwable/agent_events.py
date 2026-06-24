"""Agent runtime event schema."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from qwable.agent_state import utc_now_iso


AGENT_EVENT_TYPES = [
    "agent_run_created",
    "agent_context_pack_ready",
    "agent_plan_started",
    "agent_plan_ready",
    "agent_plan_review_started",
    "agent_plan_review_done",
    "agent_step_started",
    "agent_tool_requested",
    "agent_tool_result_received",
    "agent_test_started",
    "agent_test_failed",
    "agent_test_passed",
    "agent_repair_started",
    "agent_repair_done",
    "agent_finalizing",
    "agent_completed",
    "agent_failed",
]

AgentEventType = Literal[
    "agent_run_created",
    "agent_context_pack_ready",
    "agent_plan_started",
    "agent_plan_ready",
    "agent_plan_review_started",
    "agent_plan_review_done",
    "agent_step_started",
    "agent_tool_requested",
    "agent_tool_result_received",
    "agent_test_started",
    "agent_test_failed",
    "agent_test_passed",
    "agent_repair_started",
    "agent_repair_done",
    "agent_finalizing",
    "agent_completed",
    "agent_failed",
]


@dataclass
class AgentEvent:
    type: AgentEventType
    run_id: str
    message: str
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=utc_now_iso)


def agent_event_to_stream_event(agent_event: AgentEvent):
    """Convert an AgentEvent into the existing SSE stream event shape."""

    from qwable.streaming_events import FusionStreamEvent

    return FusionStreamEvent(
        event=agent_event.type,
        data={
            "run_id": agent_event.run_id,
            "message": agent_event.message,
            "metadata": agent_event.metadata,
            "created_at": agent_event.created_at,
        },
    )

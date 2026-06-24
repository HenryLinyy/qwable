"""Tests for agent runtime event schema."""

from datetime import datetime


def test_agent_event_types_match_phase_9_contract():
    from qwable.agent_events import AGENT_EVENT_TYPES

    assert AGENT_EVENT_TYPES == [
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


def test_agent_event_constructs_with_timestamp_and_independent_metadata():
    from qwable.agent_events import AgentEvent

    first = AgentEvent(
        type="agent_run_created",
        run_id="run_123",
        message="Run created",
        metadata={"workflow": "coding-workflow"},
    )
    second = AgentEvent(type="agent_completed", run_id="run_456", message="Done")
    second.metadata["status"] = "completed"

    assert first.type == "agent_run_created"
    assert first.run_id == "run_123"
    assert first.message == "Run created"
    assert first.metadata == {"workflow": "coding-workflow"}
    assert second.metadata == {"status": "completed"}
    assert datetime.fromisoformat(first.created_at).tzinfo is not None
    assert datetime.fromisoformat(second.created_at).tzinfo is not None


def test_agent_event_to_stream_event_shape():
    from qwable.agent_events import AgentEvent, agent_event_to_stream_event
    from qwable.streaming_events import FusionStreamEvent

    agent_event = AgentEvent(
        type="agent_step_started",
        run_id="run_123",
        message="Starting step",
        metadata={"step_id": "step_1"},
        created_at="2026-06-22T00:00:00+00:00",
    )

    stream_event = agent_event_to_stream_event(agent_event)

    assert isinstance(stream_event, FusionStreamEvent)
    assert stream_event.event == "agent_step_started"
    assert stream_event.data == {
        "run_id": "run_123",
        "message": "Starting step",
        "metadata": {"step_id": "step_1"},
        "created_at": "2026-06-22T00:00:00+00:00",
    }

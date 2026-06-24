"""Tests for agent run state containers."""

from datetime import datetime
import re


def test_agent_run_create_assigns_id_and_defaults():
    from qwable.agent_state import AgentRun

    run = AgentRun.create(goal="Ship v1.7", workflow="coding-workflow")

    assert re.fullmatch(r"run_[0-9a-f]{16}", run.run_id)
    assert run.goal == "Ship v1.7"
    assert run.workflow == "coding-workflow"
    assert run.status == "planning"
    assert run.plan == []
    assert run.current_step_index == 0
    assert run.artifacts == []
    assert run.failures == []
    assert run.repair_count == 0
    assert run.tool_call_count == 0
    assert run.trace == {}
    assert datetime.fromisoformat(run.created_at).tzinfo is not None
    assert datetime.fromisoformat(run.updated_at).tzinfo is not None


def test_current_step_empty_plan_returns_none():
    from qwable.agent_state import AgentRun

    run = AgentRun.create(goal="No plan yet", workflow="agentic-workflow")

    assert run.current_step() is None


def test_current_step_returns_valid_index_and_rejects_out_of_range():
    from qwable.agent_state import AgentRun, AgentStep

    first = AgentStep(step_id="step_1", title="Inspect", intent="Read files")
    second = AgentStep(step_id="step_2", title="Patch", intent="Apply change")
    run = AgentRun.create(goal="Patch safely", workflow="coding-workflow")
    run.plan = [first, second]

    assert run.current_step() is first
    run.current_step_index = 1
    assert run.current_step() is second
    run.current_step_index = -1
    assert run.current_step() is None
    run.current_step_index = 2
    assert run.current_step() is None


def test_touch_updates_updated_at():
    from qwable.agent_state import AgentRun

    run = AgentRun.create(goal="Touch timestamp", workflow="review-workflow")
    run.updated_at = "2000-01-01T00:00:00+00:00"

    run.touch()

    assert run.updated_at != "2000-01-01T00:00:00+00:00"
    assert datetime.fromisoformat(run.updated_at).tzinfo is not None


def test_step_artifact_and_failure_defaults_are_not_shared():
    from qwable.agent_state import AgentArtifact, AgentFailure, AgentStep

    first = AgentStep(step_id="step_1", title="One", intent="First")
    second = AgentStep(step_id="step_2", title="Two", intent="Second")
    first.required_tools.append("read_file")
    first.evidence.append("found target")

    assert first.status == "pending"
    assert first.attempt_count == 0
    assert second.required_tools == []
    assert second.evidence == []

    artifact = AgentArtifact(
        artifact_id="artifact_1",
        run_id="run_1",
        kind="tool_result",
        content="ok",
    )
    failure = AgentFailure(stage="executor", message="failed")

    artifact.metadata["tool"] = "read_file"
    failure.metadata["recoverable"] = True

    assert artifact.metadata == {"tool": "read_file"}
    assert failure.metadata == {"recoverable": True}

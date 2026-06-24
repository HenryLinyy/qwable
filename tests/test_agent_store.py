"""Tests for SQLite-backed agent run storage."""

import sqlite3


def _store_path(tmp_path):
    return tmp_path / "agent_runs.sqlite3"


def test_init_schema_idempotent_creates_expected_tables(tmp_path):
    from qwable.agent_store import AgentStore

    db_path = _store_path(tmp_path)
    store = AgentStore(str(db_path))

    store.init_schema()
    store.init_schema()

    with sqlite3.connect(db_path) as conn:
        tables = {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            )
        }

    assert {
        "agent_runs",
        "agent_steps",
        "agent_artifacts",
        "agent_failures",
    }.issubset(tables)


def test_save_load_run_roundtrip(tmp_path):
    from qwable.agent_state import (
        AgentArtifact,
        AgentFailure,
        AgentRun,
        AgentStep,
    )
    from qwable.agent_store import AgentStore

    store = AgentStore(str(_store_path(tmp_path)))
    store.init_schema()
    run = AgentRun.create(goal="Implement store", workflow="coding-workflow")
    run.status = "waiting_for_tool"
    run.current_step_index = 1
    run.repair_count = 2
    run.tool_call_count = 3
    run.trace = {"workflow": "coding-workflow", "selected_model": "qwen"}
    run.plan = [
        AgentStep(
            step_id="step_1",
            title="Inspect",
            intent="Read files",
            status="done",
            required_tools=["read_file"],
            success_criteria=["files read"],
            evidence=["config.py"],
            output="ok",
            attempt_count=1,
        ),
        AgentStep(
            step_id="step_2",
            title="Patch",
            intent="Apply change",
            status="waiting_for_tool",
            failure_criteria=["patch rejected"],
            error="needs tool",
            attempt_count=2,
        ),
    ]
    run.artifacts = [
        AgentArtifact(
            artifact_id="artifact_1",
            run_id=run.run_id,
            kind="tool_result",
            content="pytest passed",
            metadata={"command": "pytest"},
        )
    ]
    run.failures = [
        AgentFailure(
            stage="executor",
            message="tool failed",
            metadata={"recoverable": True},
        )
    ]

    store.save_run(run)
    loaded = store.load_run(run.run_id)

    assert loaded is not None
    assert loaded.run_id == run.run_id
    assert loaded.workflow == "coding-workflow"
    assert loaded.goal == "Implement store"
    assert loaded.status == "waiting_for_tool"
    assert loaded.current_step_index == 1
    assert loaded.repair_count == 2
    assert loaded.tool_call_count == 3
    assert loaded.trace == {"workflow": "coding-workflow", "selected_model": "qwen"}
    assert [step.step_id for step in loaded.plan] == ["step_1", "step_2"]
    assert loaded.plan[0].required_tools == ["read_file"]
    assert loaded.plan[0].success_criteria == ["files read"]
    assert loaded.plan[0].evidence == ["config.py"]
    assert loaded.plan[1].failure_criteria == ["patch rejected"]
    assert loaded.plan[1].error == "needs tool"
    assert loaded.artifacts[0].metadata == {"command": "pytest"}
    assert loaded.failures[0].metadata == {"recoverable": True}


def test_append_artifact_and_failure_roundtrip(tmp_path):
    from qwable.agent_state import AgentArtifact, AgentFailure, AgentRun
    from qwable.agent_store import AgentStore

    store = AgentStore(str(_store_path(tmp_path)))
    store.init_schema()
    run = AgentRun.create(goal="Append later", workflow="agentic-workflow")
    store.save_run(run)

    store.append_artifact(
        run.run_id,
        AgentArtifact(
            artifact_id="artifact_late",
            run_id=run.run_id,
            kind="final_report",
            content="done",
            metadata={"ok": True},
        ),
    )
    store.append_failure(
        run.run_id,
        AgentFailure(stage="planner", message="bad json", metadata={"raw": "{"}),
    )

    loaded = store.load_run(run.run_id)

    assert loaded is not None
    assert [(a.kind, a.content, a.metadata) for a in loaded.artifacts] == [
        ("final_report", "done", {"ok": True})
    ]
    assert [(f.stage, f.message, f.metadata) for f in loaded.failures] == [
        ("planner", "bad json", {"raw": "{"})
    ]


def test_list_runs_returns_updated_desc_with_limit(tmp_path):
    from qwable.agent_state import AgentRun
    from qwable.agent_store import AgentStore

    store = AgentStore(str(_store_path(tmp_path)))
    store.init_schema()
    old = AgentRun.create(goal="old", workflow="agentic-workflow")
    middle = AgentRun.create(goal="middle", workflow="agentic-workflow")
    newest = AgentRun.create(goal="newest", workflow="agentic-workflow")
    old.updated_at = "2026-01-01T00:00:00+00:00"
    middle.updated_at = "2026-01-02T00:00:00+00:00"
    newest.updated_at = "2026-01-03T00:00:00+00:00"
    for run in [old, middle, newest]:
        store.save_run(run)

    assert [run.run_id for run in store.list_runs(limit=2)] == [
        newest.run_id,
        middle.run_id,
    ]


def test_save_run_replaces_steps_without_stale_rows(tmp_path):
    from qwable.agent_state import AgentRun, AgentStep
    from qwable.agent_store import AgentStore

    store = AgentStore(str(_store_path(tmp_path)))
    store.init_schema()
    run = AgentRun.create(goal="Replace plan", workflow="coding-workflow")
    run.plan = [
        AgentStep(step_id="step_old", title="Old", intent="Remove me"),
        AgentStep(step_id="step_keep", title="Keep", intent="Keep me"),
    ]
    store.save_run(run)
    run.plan = [AgentStep(step_id="step_keep", title="Keep", intent="Keep me")]
    store.save_run(run)

    loaded = store.load_run(run.run_id)

    assert loaded is not None
    assert [step.step_id for step in loaded.plan] == ["step_keep"]

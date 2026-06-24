"""E2E mock tests for v1.7 agent workflow runtime."""

import json

import pytest

from qwable.agent_state import AgentRun, AgentStep
from qwable.config import FusionConfig
from qwable.fusion_core import FusionCore
from qwable.profiles import resolve_profile
from qwable.schemas import ParsedAgentTask, ToolResult
from tests.agent_mocks import (
    FakeAgentModelClient,
    fake_chat_response,
    fake_executor_tool_call_response,
    fake_planner_response,
)


def _task(profile: str, text: str, *, raw_request=None, tool_results=None) -> ParsedAgentTask:
    return ParsedAgentTask(
        text=text,
        tools=[],
        tool_results=tool_results or [],
        profile=profile,
        source_protocol="openai_responses",
        stream=False,
        raw_request=raw_request or {"input": text},
    )


def _tool_result(name: str, content: str, *, is_error=False) -> ToolResult:
    return ToolResult(
        tool_call_id="call_1",
        name=name,
        content=content,
        is_error=is_error,
        source_protocol="openai_responses",
        raw={"content": content},
    )


def _core(tmp_path, scripted_outputs, **config_overrides):
    config = FusionConfig(
        agent_store_path=str(tmp_path / "agent_runs.sqlite3"),
        **config_overrides,
    )
    core = FusionCore(config)
    fake_client = FakeAgentModelClient(scripted_outputs)
    core.ollama = fake_client
    core.agent_orchestrator.model_client = fake_client
    return core, fake_client


def _critic_response(*, fatal=False):
    return fake_chat_response(
        json.dumps(
            {
                "fatal_blocker": fatal,
                "blockers": ["missing evidence"] if fatal else [],
                "risks": [],
            }
        )
    )


@pytest.mark.asyncio
async def test_code_agent_model_runs_planner_critic_executor_and_returns_tool_call(tmp_path):
    profile = resolve_profile("qwable-code-agent", "openai_responses")
    core, fake_client = _core(
        tmp_path,
        [
            fake_planner_response(),
            _critic_response(),
            fake_executor_tool_call_response(),
        ],
    )

    action = await core.execute(_task(profile, "Implement agent runtime v1.7"))

    assert profile == "coding-workflow"
    assert action.type == "tool_call"
    assert action.tool_name == "search_files"
    assert action.trace["agent_run_id"].startswith("run_")
    assert action.trace["workflow"] == "coding-workflow"
    assert action.trace["model_role"] == "executor"
    assert [call["model"] for call in fake_client.calls] == [
        core.config.model_role_planner,
        core.config.model_role_critic,
        core.config.model_qwable,  # v1.8: executor stage now routes through select_for_stage -> Qwable
    ]


@pytest.mark.asyncio
async def test_code_agent_executor_falls_back_to_coder_when_qwable_disabled(tmp_path):
    """Backward compat: disabling Qwable reverts the executor stage to the
    v1.7 default (qwen3-coder-next) — proves the rewire is opt-out, not forced.
    """
    core, fake_client = _core(
        tmp_path,
        [
            fake_planner_response(),
            _critic_response(),
            fake_executor_tool_call_response(),
        ],
        enable_qwable_executor=False,
    )

    await core.execute(_task("coding-workflow", "Implement agent runtime v1.7"))

    assert [call["model"] for call in fake_client.calls] == [
        core.config.model_role_planner,
        core.config.model_role_critic,
        core.config.model_coder,  # Qwable disabled -> executor falls back to qwen3-coder-next
    ]


@pytest.mark.asyncio
async def test_tool_result_continuation_appends_evidence_and_finalizes(tmp_path):
    core, fake_client = _core(
        tmp_path,
        [
            fake_planner_response(),
            _critic_response(),
            fake_executor_tool_call_response(),
            fake_chat_response("Final report after search result"),
        ],
    )
    first = await core.execute(_task("coding-workflow", "Implement agent runtime v1.7"))

    continuation = await core.execute(
        _task(
            "coding-workflow",
            "Implement agent runtime v1.7",
            raw_request={"metadata": {"agent_run_id": first.trace["agent_run_id"]}},
            tool_results=[_tool_result("search_files", "agent_orchestrator.py found")],
        )
    )

    assert continuation.type == "final_answer"
    assert continuation.text == "Final report after search result"
    assert continuation.trace["stage"] == "finalizer"
    assert [call["model"] for call in fake_client.calls] == [
        core.config.model_role_planner,
        core.config.model_role_critic,
        core.config.model_qwable,  # v1.8: executor stage now routes through select_for_stage -> Qwable
        core.config.model_role_judge,
    ]

    loaded = core.agent_store.load_run(first.trace["agent_run_id"])
    assert loaded is not None
    assert loaded.current_step().status == "done"
    assert "search_files:agent_orchestrator.py found" in loaded.current_step().evidence
    assert any(artifact.kind == "final_report" for artifact in loaded.artifacts)


@pytest.mark.asyncio
async def test_test_failure_uses_repair_model_and_returns_patch_tool_call(tmp_path):
    core, fake_client = _core(
        tmp_path,
        [
            fake_executor_tool_call_response(
                name="apply_patch",
                tool_input={
                    "path": "qwable/agent_orchestrator.py",
                    "patch": "*** Begin Patch\n*** End Patch\n",
                },
            )
        ],
    )
    run = _saved_testing_run(core, repair_count=0)

    action = await core.execute(
        _task(
            "coding-workflow",
            "Repair failed tests",
            raw_request={"metadata": {"agent_run_id": run.run_id}},
            tool_results=[_tool_result("run_tests", "AssertionError", is_error=True)],
        )
    )

    assert action.type == "tool_call"
    assert action.tool_name == "apply_patch"
    assert action.trace["stage"] == "repair"
    assert action.trace["model_role"] == "repair"
    assert [call["model"] for call in fake_client.calls] == [core.config.model_qwable]  # v1.8: repair -> Qwable

    loaded = core.agent_store.load_run(run.run_id)
    assert loaded is not None
    assert loaded.repair_count == 1


@pytest.mark.asyncio
async def test_repair_limit_blocks_without_calling_repair_model(tmp_path):
    core, fake_client = _core(tmp_path, [])
    run = _saved_testing_run(core, repair_count=core.config.agent_max_repair_attempts)

    action = await core.execute(
        _task(
            "coding-workflow",
            "Repair failed tests",
            raw_request={"metadata": {"agent_run_id": run.run_id}},
            tool_results=[_tool_result("run_tests", "AssertionError", is_error=True)],
        )
    )

    assert action.type == "final_answer"
    assert "repair_blocked:max_repair_attempts_exceeded" in action.text
    assert action.trace["agent_status"] == "blocked"
    assert action.trace["stage"] == "repair"
    assert fake_client.calls == []


def _saved_testing_run(core: FusionCore, *, repair_count: int) -> AgentRun:
    run = AgentRun.create(goal="Fix tests", workflow="coding-workflow")
    run.status = "testing"
    run.repair_count = repair_count
    run.plan = [
        AgentStep(
            step_id="step_1",
            title="Patch",
            intent="Apply change",
            status="done",
            attempt_count=1,
        )
    ]
    core.agent_store.save_run(run)
    return run

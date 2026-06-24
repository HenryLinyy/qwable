"""Tests for agent runtime prompt builders."""


def _context_pack():
    from qwable.context_pack import ContextFileSummary, ContextPack

    return ContextPack(
        goal="Implement prompts",
        workflow="coding-workflow",
        files=[
            ContextFileSummary(
                path="qwable/agent_prompts.py",
                reason="target file",
                summary="Prompt builders live here",
                symbols=["build_planner_messages"],
            )
        ],
        raw_evidence=["request: Implement qwable/agent_prompts.py"],
        constraints=["No model calls in prompt builders"],
        risks=["tests not run yet"],
    )


def _task():
    from qwable.schemas import ParsedAgentTask

    return ParsedAgentTask(
        text="Implement prompt builders",
        tools=[],
        tool_results=[],
        profile="coding-workflow",
        source_protocol="openai_responses",
        stream=False,
        raw_request={"input": "Implement prompt builders"},
    )


def _run():
    from qwable.agent_state import AgentArtifact, AgentFailure, AgentRun, AgentStep

    run = AgentRun.create(goal="Implement prompt builders", workflow="coding-workflow")
    run.plan = [
        AgentStep(
            step_id="step_1",
            title="Add tests",
            intent="Write prompt tests",
            required_tools=["search_files"],
            success_criteria=["tests fail first"],
        )
    ]
    run.artifacts = [
        AgentArtifact(
            artifact_id="artifact_1",
            run_id=run.run_id,
            kind="test_result",
            content="pytest failed",
        )
    ]
    run.failures = [
        AgentFailure(stage="executor", message="missing builder")
    ]
    return run


def test_system_prompts_include_required_contracts():
    from qwable.agent_prompts import (
        CRITIC_SYSTEM,
        EXECUTOR_SYSTEM,
        FINALIZER_SYSTEM,
        PLANNER_SYSTEM,
        REPAIR_SYSTEM,
    )

    assert "Qwable Agent Planner" in PLANNER_SYSTEM
    assert "JSON" in PLANNER_SYSTEM
    assert '"steps"' in PLANNER_SYSTEM
    assert "Qwable Critic" in CRITIC_SYSTEM
    assert "missing evidence" in CRITIC_SYSTEM
    assert "不得加入未提供的事實" in CRITIC_SYSTEM
    assert "Qwable Executor" in EXECUTOR_SYSTEM
    assert "一次只執行目前 step" in EXECUTOR_SYSTEM
    assert "tool_call" in EXECUTOR_SYSTEM
    assert "Qwable Repair Agent" in REPAIR_SYSTEM
    assert "不得擴大修改範圍" in REPAIR_SYSTEM
    assert "Qwable Finalizer" in FINALIZER_SYSTEM
    assert "不得宣稱未執行的測試已通過" in FINALIZER_SYSTEM


def test_build_planner_messages_include_task_and_context():
    from qwable.agent_prompts import build_planner_messages

    messages = build_planner_messages(_task(), _context_pack())

    assert [message["role"] for message in messages] == ["system", "user"]
    assert "Qwable Agent Planner" in messages[0]["content"]
    assert "USER_GOAL" in messages[1]["content"]
    assert "Implement prompt builders" in messages[1]["content"]
    assert "CONTEXT_PACK" in messages[1]["content"]
    assert "qwable/agent_prompts.py" in messages[1]["content"]


def test_build_plan_critic_messages_include_run_plan_and_context():
    from qwable.agent_prompts import build_plan_critic_messages

    run = _run()
    messages = build_plan_critic_messages(run, _context_pack())

    assert messages[0]["role"] == "system"
    assert "Qwable Critic" in messages[0]["content"]
    assert "AGENT_RUN" in messages[1]["content"]
    assert run.run_id in messages[1]["content"]
    assert "Add tests" in messages[1]["content"]
    assert "CONTEXT_PACK" in messages[1]["content"]


def test_build_executor_messages_include_current_step_only():
    from qwable.agent_prompts import build_executor_messages

    run = _run()
    current_step = run.plan[0]
    messages = build_executor_messages(run, _context_pack(), current_step)

    assert "Qwable Executor" in messages[0]["content"]
    assert "CURRENT_STEP" in messages[1]["content"]
    assert "Add tests" in messages[1]["content"]
    assert "test_result" in messages[1]["content"]
    assert "CONTEXT_PACK" in messages[1]["content"]


def test_build_repair_messages_include_failure_text():
    from qwable.agent_prompts import build_repair_messages

    messages = build_repair_messages(_run(), _context_pack(), "pytest failed at test_agent_prompts")

    assert "Qwable Repair Agent" in messages[0]["content"]
    assert "FAILURE_TEXT" in messages[1]["content"]
    assert "pytest failed at test_agent_prompts" in messages[1]["content"]
    assert "AGENT_RUN" in messages[1]["content"]


def test_build_finalizer_messages_include_trace_and_honesty_contract():
    from qwable.agent_prompts import build_finalizer_messages

    messages = build_finalizer_messages(_run(), _context_pack())

    assert "Qwable Finalizer" in messages[0]["content"]
    assert "不得宣稱未執行的測試已通過" in messages[0]["content"]
    assert "FINAL_REPORT_INPUT" in messages[1]["content"]
    assert "pytest failed" in messages[1]["content"]
    assert "missing builder" in messages[1]["content"]

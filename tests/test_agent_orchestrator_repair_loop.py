"""Acceptance split tests for agent orchestrator repair and review flow."""

from tests import test_agent_orchestrator as orchestrator_tests


async def test_successful_test_result_runs_finalizer(tmp_path):
    await orchestrator_tests.test_successful_test_result_runs_finalizer(tmp_path)


async def test_failed_test_result_uses_repair_loop_and_requests_repair_tool(tmp_path):
    await orchestrator_tests.test_failed_test_result_uses_repair_loop_and_requests_repair_tool(
        tmp_path
    )


async def test_unrepairable_test_failure_returns_blocked_final_answer(tmp_path):
    await (
        orchestrator_tests.test_unrepairable_test_failure_returns_blocked_final_answer(
            tmp_path
        )
    )


async def test_review_workflow_uses_reviewer_judge_finalizer_without_patch_loop(
    tmp_path,
):
    await orchestrator_tests.test_review_workflow_uses_reviewer_judge_finalizer_without_patch_loop(
        tmp_path
    )


async def test_review_workflow_suppresses_apply_patch_tool_call(tmp_path):
    await orchestrator_tests.test_review_workflow_suppresses_apply_patch_tool_call(
        tmp_path
    )

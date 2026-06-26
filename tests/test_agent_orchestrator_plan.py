"""Acceptance split tests for agent orchestrator planning flow."""

from tests import test_agent_orchestrator as orchestrator_tests


def test_extract_agent_run_id_accepts_valid_metadata_only():
    orchestrator_tests.test_extract_agent_run_id_accepts_valid_metadata_only()


async def test_run_uses_model_selector_for_planner_and_executor(tmp_path):
    await orchestrator_tests.test_run_uses_model_selector_for_planner_and_executor(
        tmp_path
    )


async def test_invalid_planner_json_returns_failed_final_answer(tmp_path):
    await orchestrator_tests.test_invalid_planner_json_returns_failed_final_answer(
        tmp_path
    )


async def test_agentic_workflow_runs_plan_critic_before_executor(tmp_path):
    await orchestrator_tests.test_agentic_workflow_runs_plan_critic_before_executor(
        tmp_path
    )


async def test_agentic_workflow_blocks_on_fatal_plan_critic(tmp_path):
    await orchestrator_tests.test_agentic_workflow_blocks_on_fatal_plan_critic(tmp_path)

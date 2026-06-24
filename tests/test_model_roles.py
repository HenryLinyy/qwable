"""Tests for agent model role value contracts."""


def test_model_roles_are_stable():
    from qwable.model_roles import ModelRole

    assert ModelRole.SIMPLE_FORMATTER.value == "simple_formatter"
    assert ModelRole.PLANNER.value == "planner"
    assert ModelRole.EXECUTOR.value == "executor"
    assert ModelRole.REPAIR.value == "repair"
    assert ModelRole.CRITIC.value == "critic"
    assert ModelRole.JUDGE.value == "judge"
    assert ModelRole.HEAVY_PRIMARY.value == "heavy_primary"
    assert ModelRole.VISION.value == "vision"


def test_workflow_names_and_stages_are_stable():
    from qwable.model_roles import WorkflowName, WorkflowStage

    assert WorkflowName.AGENTIC.value == "agentic-workflow"
    assert WorkflowName.CODING.value == "coding-workflow"
    assert WorkflowName.REVIEW.value == "review-workflow"
    assert WorkflowStage.CONTEXT.value == "context"
    assert WorkflowStage.PLANNER.value == "planner"
    assert WorkflowStage.PLAN_CRITIC.value == "plan_critic"
    assert WorkflowStage.EXECUTOR.value == "executor"
    assert WorkflowStage.TEST.value == "test"
    assert WorkflowStage.REPAIR.value == "repair"
    assert WorkflowStage.REVIEWER.value == "reviewer"
    assert WorkflowStage.JUDGE.value == "judge"
    assert WorkflowStage.FINALIZER.value == "finalizer"

"""Tests for agent workflow stage role maps."""


def test_coding_workflow_maps_mutating_stages_to_executor_roles():
    from qwable.model_roles import ModelRole, WorkflowName, WorkflowStage
    from qwable.workflows import WORKFLOW_STAGE_ROLE_MAP

    stages = WORKFLOW_STAGE_ROLE_MAP[WorkflowName.CODING.value]

    assert stages[WorkflowStage.CONTEXT.value] == ModelRole.PLANNER
    assert stages[WorkflowStage.PLANNER.value] == ModelRole.PLANNER
    assert stages[WorkflowStage.PLAN_CRITIC.value] == ModelRole.CRITIC
    assert stages[WorkflowStage.EXECUTOR.value] == ModelRole.EXECUTOR
    assert stages[WorkflowStage.TEST.value] == ModelRole.EXECUTOR
    assert stages[WorkflowStage.REPAIR.value] == ModelRole.REPAIR
    assert stages[WorkflowStage.JUDGE.value] == ModelRole.JUDGE
    assert stages[WorkflowStage.FINALIZER.value] == ModelRole.JUDGE


def test_review_workflow_does_not_map_executor_or_repair_stages():
    from qwable.model_roles import ModelRole, WorkflowName, WorkflowStage
    from qwable.workflows import WORKFLOW_STAGE_ROLE_MAP

    stages = WORKFLOW_STAGE_ROLE_MAP[WorkflowName.REVIEW.value]

    assert stages[WorkflowStage.CONTEXT.value] == ModelRole.PLANNER
    assert stages[WorkflowStage.REVIEWER.value] == ModelRole.CRITIC
    assert stages[WorkflowStage.JUDGE.value] == ModelRole.JUDGE
    assert stages[WorkflowStage.FINALIZER.value] == ModelRole.JUDGE
    assert WorkflowStage.EXECUTOR.value not in stages
    assert WorkflowStage.REPAIR.value not in stages


def test_workflow_default_max_tokens_cover_mapped_generation_stages():
    from qwable.workflows import WORKFLOW_DEFAULT_MAX_TOKENS, WORKFLOW_STAGE_ROLE_MAP

    for workflow, stages in WORKFLOW_DEFAULT_MAX_TOKENS.items():
        assert workflow in WORKFLOW_STAGE_ROLE_MAP
        for stage, max_tokens in stages.items():
            assert stage in WORKFLOW_STAGE_ROLE_MAP[workflow]
            assert max_tokens > 0

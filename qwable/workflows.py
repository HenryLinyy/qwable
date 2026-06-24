"""Static workflow stage-to-model-role maps."""


from qwable.model_roles import ModelRole, WorkflowName, WorkflowStage


# v1.7 stage map (kept for backward compat — used by ModelSelector.select()).
WORKFLOW_STAGE_ROLE_MAP: dict[str, dict[str, ModelRole]] = {
    WorkflowName.AGENTIC.value: {
        WorkflowStage.CONTEXT.value: ModelRole.PLANNER,
        WorkflowStage.PLANNER.value: ModelRole.PLANNER,
        WorkflowStage.PLAN_CRITIC.value: ModelRole.CRITIC,
        WorkflowStage.EXECUTOR.value: ModelRole.EXECUTOR,
        WorkflowStage.JUDGE.value: ModelRole.JUDGE,
        WorkflowStage.FINALIZER.value: ModelRole.JUDGE,
    },
    WorkflowName.CODING.value: {
        WorkflowStage.CONTEXT.value: ModelRole.PLANNER,
        WorkflowStage.PLANNER.value: ModelRole.PLANNER,
        WorkflowStage.PLAN_CRITIC.value: ModelRole.CRITIC,
        WorkflowStage.EXECUTOR.value: ModelRole.EXECUTOR,
        WorkflowStage.TEST.value: ModelRole.EXECUTOR,
        WorkflowStage.REPAIR.value: ModelRole.REPAIR,
        WorkflowStage.JUDGE.value: ModelRole.JUDGE,
        WorkflowStage.FINALIZER.value: ModelRole.JUDGE,
    },
    WorkflowName.REVIEW.value: {
        WorkflowStage.CONTEXT.value: ModelRole.PLANNER,
        WorkflowStage.REVIEWER.value: ModelRole.CRITIC,
        WorkflowStage.JUDGE.value: ModelRole.JUDGE,
        WorkflowStage.FINALIZER.value: ModelRole.JUDGE,
    },
}


# v1.8: finer-grained stage map (per plan §8). Each stage picks exactly one
# ModelRole. Selection of the *model* is handled by ModelSelector based on
# the role + Qwable/Qwythos enable flags.
STAGE_ROLE_MAP: dict[WorkflowStage, ModelRole] = {
    # Long-context stages — Qwythos optional primary, qwen3.6 default.
    WorkflowStage.CONTEXT_ACQUISITION: ModelRole.LONG_CONTEXT_WORKER,
    WorkflowStage.REPO_INDEX: ModelRole.LONG_CONTEXT_WORKER,
    WorkflowStage.CONTEXT_COMPACTION: ModelRole.LONG_CONTEXT_WORKER,
    # Planning — unchanged from v1.7.
    WorkflowStage.PLAN_REVIEW: ModelRole.CRITIC,
    WorkflowStage.PLAN_REVISION: ModelRole.PLANNER,
    # Execution / repair — Qwable primary, qwen3-coder-next fallback.
    WorkflowStage.EXECUTE_PATCH: ModelRole.EXECUTOR,
    WorkflowStage.REPAIR_PATCH: ModelRole.REPAIR,
    # Failure analysis — long context (read tool result + error tail).
    WorkflowStage.FAILURE_ANALYSIS: ModelRole.LONG_CONTEXT_WORKER,
    # Final stages — judge (unchanged).
    WorkflowStage.FINAL_REVIEW: ModelRole.JUDGE,
    WorkflowStage.FINAL_REPORT: ModelRole.JUDGE,
}


WORKFLOW_DEFAULT_MAX_TOKENS: dict[str, dict[str, int]] = {
    WorkflowName.AGENTIC.value: {
        WorkflowStage.PLANNER.value: 1200,
        WorkflowStage.PLAN_CRITIC.value: 1200,
        WorkflowStage.EXECUTOR.value: 1200,
        WorkflowStage.JUDGE.value: 1600,
        WorkflowStage.FINALIZER.value: 1600,
    },
    WorkflowName.CODING.value: {
        WorkflowStage.PLANNER.value: 1600,
        WorkflowStage.PLAN_CRITIC.value: 1200,
        WorkflowStage.EXECUTOR.value: 1800,
        WorkflowStage.REPAIR.value: 1600,
        WorkflowStage.JUDGE.value: 1600,
        WorkflowStage.FINALIZER.value: 1600,
    },
    WorkflowName.REVIEW.value: {
        WorkflowStage.REVIEWER.value: 1800,
        WorkflowStage.JUDGE.value: 1600,
        WorkflowStage.FINALIZER.value: 1600,
    },
}

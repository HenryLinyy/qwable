"""Agent workflow model roles and selection result types."""

from dataclasses import dataclass
from enum import Enum


class ModelRole(str, Enum):
    SIMPLE_FORMATTER = "simple_formatter"
    PLANNER = "planner"
    EXECUTOR = "executor"
    REPAIR = "repair"
    LONG_CONTEXT_WORKER = "long_context_worker"  # v1.8
    CRITIC = "critic"
    JUDGE = "judge"
    HEAVY_PRIMARY = "heavy_primary"
    VISION = "vision"


class WorkflowName(str, Enum):
    AGENTIC = "agentic-workflow"
    CODING = "coding-workflow"
    REVIEW = "review-workflow"


class WorkflowStage(str, Enum):
    # v1.7 stages (kept for backward compat with WORKFLOW_STAGE_ROLE_MAP)
    CONTEXT = "context"
    PLANNER = "planner"
    PLAN_CRITIC = "plan_critic"
    EXECUTOR = "executor"
    TEST = "test"
    REPAIR = "repair"
    REVIEWER = "reviewer"
    JUDGE = "judge"
    FINALIZER = "finalizer"
    # v1.8 finer-grained stages (per plan §8)
    CONTEXT_ACQUISITION = "context_acquisition"
    REPO_INDEX = "repo_index"
    CONTEXT_COMPACTION = "context_compaction"
    EXECUTE_PATCH = "execute_patch"
    REPAIR_PATCH = "repair_patch"
    FAILURE_ANALYSIS = "failure_analysis"
    PLAN_REVIEW = "plan_review"
    PLAN_REVISION = "plan_revision"
    FINAL_REVIEW = "final_review"
    FINAL_REPORT = "final_report"


@dataclass(frozen=True)
class RoleSelection:
    workflow: str
    stage: str
    role: ModelRole
    model: str
    fallback_chain: list[str]
    max_tokens: int
    temperature: float
    reason: str

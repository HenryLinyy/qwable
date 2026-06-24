"""Agent run state dataclasses for the v1.7 runtime."""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Literal
import uuid


AgentRunStatus = Literal[
    "planning",
    "reviewing_plan",
    "executing",
    "waiting_for_tool",
    "testing",
    "repairing",
    "reviewing",
    "finalizing",
    "completed",
    "failed",
    "blocked",
]
AgentStepStatus = Literal[
    "pending",
    "running",
    "waiting_for_tool",
    "done",
    "failed",
    "blocked",
    "skipped",
]
ArtifactKind = Literal[
    "context_pack",
    "plan",
    "plan_review",
    "tool_call",
    "tool_result",
    "patch",
    "test_result",
    "repair",
    "review",
    "final_report",
]


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:16]}"


@dataclass
class AgentStep:
    step_id: str
    title: str
    intent: str
    status: AgentStepStatus = "pending"
    required_tools: list[str] = field(default_factory=list)
    success_criteria: list[str] = field(default_factory=list)
    failure_criteria: list[str] = field(default_factory=list)
    evidence: list[str] = field(default_factory=list)
    output: str | None = None
    error: str | None = None
    attempt_count: int = 0


@dataclass
class AgentArtifact:
    artifact_id: str
    run_id: str
    kind: ArtifactKind
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=utc_now_iso)


@dataclass
class AgentFailure:
    stage: str
    message: str
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=utc_now_iso)


@dataclass
class AgentRun:
    run_id: str
    workflow: str
    goal: str
    status: AgentRunStatus = "planning"
    plan: list[AgentStep] = field(default_factory=list)
    current_step_index: int = 0
    artifacts: list[AgentArtifact] = field(default_factory=list)
    failures: list[AgentFailure] = field(default_factory=list)
    repair_count: int = 0
    tool_call_count: int = 0
    created_at: str = field(default_factory=utc_now_iso)
    updated_at: str = field(default_factory=utc_now_iso)
    trace: dict[str, Any] = field(default_factory=dict)

    @staticmethod
    def create(goal: str, workflow: str) -> "AgentRun":
        return AgentRun(run_id=new_id("run"), workflow=workflow, goal=goal)

    def touch(self) -> None:
        self.updated_at = utc_now_iso()

    def current_step(self) -> AgentStep | None:
        if not self.plan:
            return None
        if self.current_step_index < 0 or self.current_step_index >= len(self.plan):
            return None
        return self.plan[self.current_step_index]

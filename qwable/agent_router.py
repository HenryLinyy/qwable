"""Route generic agent workflow requests to specialized workflows."""

from __future__ import annotations

from qwable.config import FusionConfig
from qwable.schemas import ParsedAgentTask


CODING_KEYWORDS = [
    "implement",
    "patch",
    "bug",
    "fix",
    "repo",
    "test",
    "pytest",
    "修改",
    "修",
    "實作",
    "施工",
]

REVIEW_KEYWORDS = [
    "review",
    "audit",
    "risk",
    "security",
    "審查",
    "風險",
    "安全",
    "漏洞",
]

PATCH_KEYWORDS = [
    "implement",
    "patch",
    "bug",
    "fix",
    "test",
    "pytest",
    "修改",
    "修",
    "實作",
    "施工",
]

WORKFLOW_PROFILES = {"agentic-workflow", "coding-workflow", "review-workflow"}


class AgentRouter:
    def __init__(self, config: FusionConfig):
        self.config = config

    def resolve_workflow(self, task: ParsedAgentTask, default_workflow: str) -> str:
        forced = _forced_workflow(task)
        if (
            forced == "coding-workflow"
            and (task.profile in WORKFLOW_PROFILES or default_workflow in WORKFLOW_PROFILES)
        ):
            return "coding-workflow"

        if task.profile == "coding-workflow" or default_workflow == "coding-workflow":
            return "coding-workflow"
        if task.profile == "review-workflow" or default_workflow == "review-workflow":
            return "review-workflow"
        if default_workflow != "agentic-workflow":
            return default_workflow

        text = (task.text or "").lower()
        if _contains_keyword(text, CODING_KEYWORDS):
            return "coding-workflow"
        if _contains_keyword(text, REVIEW_KEYWORDS) and not _contains_keyword(text, PATCH_KEYWORDS):
            return "review-workflow"
        return default_workflow


def _forced_workflow(task: ParsedAgentTask) -> str | None:
    metadata = task.raw_request.get("metadata") if isinstance(task.raw_request, dict) else None
    if not isinstance(metadata, dict):
        return None
    value = metadata.get("force_workflow")
    if isinstance(value, str) and value in WORKFLOW_PROFILES:
        return value
    return None


def _contains_keyword(text: str, keywords: list[str]) -> bool:
    return any(keyword in text for keyword in keywords)

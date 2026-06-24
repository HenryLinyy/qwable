"""Tests for v1.7 AgentRouter workflow upgrades."""

from qwable.config import FusionConfig
from qwable.schemas import ParsedAgentTask


def _task(profile: str, text: str, *, metadata=None) -> ParsedAgentTask:
    raw_request = {"input": text}
    if metadata is not None:
        raw_request["metadata"] = metadata
    return ParsedAgentTask(
        text=text,
        tools=[],
        tool_results=[],
        profile=profile,
        source_protocol="openai_responses",
        stream=False,
        raw_request=raw_request,
    )


def test_agent_router_upgrades_generic_agentic_tasks_by_keywords():
    from qwable.agent_router import AgentRouter

    router = AgentRouter(FusionConfig())

    assert (
        router.resolve_workflow(
            _task("agentic-workflow", "Please implement the repository fix"),
            "agentic-workflow",
        )
        == "coding-workflow"
    )
    assert (
        router.resolve_workflow(
            _task("agentic-workflow", "Please run a security audit"),
            "agentic-workflow",
        )
        == "review-workflow"
    )
    assert (
        router.resolve_workflow(
            _task("agentic-workflow", "整理這份資料並提出方案"),
            "agentic-workflow",
        )
        == "agentic-workflow"
    )


def test_agent_router_keeps_explicit_workflow_profiles_stable():
    from qwable.agent_router import AgentRouter

    router = AgentRouter(FusionConfig())

    assert (
        router.resolve_workflow(
            _task("coding-workflow", "security audit only"),
            "coding-workflow",
        )
        == "coding-workflow"
    )
    assert (
        router.resolve_workflow(
            _task("review-workflow", "fix this implementation bug"),
            "review-workflow",
        )
        == "review-workflow"
    )


def test_agent_router_allows_explicit_force_to_coding_from_review():
    from qwable.agent_router import AgentRouter

    router = AgentRouter(FusionConfig())

    assert (
        router.resolve_workflow(
            _task(
                "review-workflow",
                "review this patch",
                metadata={"force_workflow": "coding-workflow"},
            ),
            "review-workflow",
        )
        == "coding-workflow"
    )

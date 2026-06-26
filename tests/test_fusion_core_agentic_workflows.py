"""Tests for FusionCore v1.7 agent workflow dispatch."""

from unittest.mock import MagicMock

import pytest

from qwable.config import FusionConfig
from qwable.fusion_core import FusionCore
from qwable.schemas import FusionAction, ParsedAgentTask


def _task(profile: str, text: str = "Implement agent runtime") -> ParsedAgentTask:
    return ParsedAgentTask(
        text=text,
        tools=[],
        tool_results=[],
        profile=profile,
        source_protocol="openai_responses",
        stream=False,
        raw_request={"input": text},
    )


class FakeAgentOrchestrator:
    def __init__(self):
        self.calls = []

    async def run(self, task, workflow):
        self.calls.append((task.profile, workflow, task.text))
        return FusionAction(
            type="final_answer",
            text=f"workflow:{workflow}",
            tool_name=None,
            tool_input=None,
            confidence=0.8,
            rationale_summary="fake agent runtime",
            trace={
                "agent_runtime": True,
                "agent_run_id": "run_test",
                "workflow": workflow,
                "stage": "executor",
                "model_role": "executor",
                "selected_model": "model/executor",
            },
        )


@pytest.mark.asyncio
async def test_agentic_workflow_dispatches_to_agent_orchestrator():
    core = FusionCore(FusionConfig())
    core.agent_orchestrator = FakeAgentOrchestrator()

    action = await core.execute(
        _task("agentic-workflow", "Organize agent runtime notes")
    )

    assert action.text == "workflow:agentic-workflow"
    assert core.agent_orchestrator.calls == [
        ("agentic-workflow", "agentic-workflow", "Organize agent runtime notes")
    ]
    assert action.trace["agent_runtime"] is True
    assert action.trace["workflow"] == "agentic-workflow"
    assert action.trace["last_used_model"] == "model/executor"
    assert core.last_used_model == "model/executor"


@pytest.mark.asyncio
async def test_generic_agentic_workflow_routes_coding_keywords_to_coding_workflow():
    core = FusionCore(FusionConfig())
    core.agent_orchestrator = FakeAgentOrchestrator()

    action = await core.execute(_task("agentic-workflow", "Please implement a bug fix"))

    assert action.text == "workflow:coding-workflow"
    assert core.agent_orchestrator.calls == [
        ("coding-workflow", "coding-workflow", "Please implement a bug fix")
    ]
    assert action.trace["workflow"] == "coding-workflow"
    assert action.trace["routed_from_workflow"] == "agentic-workflow"


@pytest.mark.asyncio
async def test_generic_agentic_workflow_routes_review_keywords_to_review_workflow():
    core = FusionCore(FusionConfig())
    core.agent_orchestrator = FakeAgentOrchestrator()

    action = await core.execute(_task("agentic-workflow", "Please do a security audit"))

    assert action.text == "workflow:review-workflow"
    assert core.agent_orchestrator.calls == [
        ("review-workflow", "review-workflow", "Please do a security audit")
    ]
    assert action.trace["workflow"] == "review-workflow"
    assert action.trace["routed_from_workflow"] == "agentic-workflow"


@pytest.mark.asyncio
async def test_coding_and_review_workflows_dispatch_to_agent_orchestrator():
    core = FusionCore(FusionConfig())
    core.agent_orchestrator = FakeAgentOrchestrator()

    coding = await core.execute(_task("coding-workflow"))
    review = await core.execute(_task("review-workflow", "Review this diff"))

    assert coding.text == "workflow:coding-workflow"
    assert review.text == "workflow:review-workflow"
    assert core.agent_orchestrator.calls == [
        ("coding-workflow", "coding-workflow", "Implement agent runtime"),
        ("review-workflow", "review-workflow", "Review this diff"),
    ]
    assert coding.trace["last_used_model"] == "model/executor"
    assert review.trace["last_used_model"] == "model/executor"


def test_agent_workflow_context_limits_are_dedicated_config_values():
    cfg = FusionConfig(
        agentic_workflow_max_input_chars=111,
        coding_workflow_max_input_chars=222,
        review_workflow_max_input_chars=333,
    )
    core = FusionCore(cfg)

    assert core._context_limit_for_profile("agentic-workflow") == 111
    assert core._context_limit_for_profile("coding-workflow") == 222
    assert core._context_limit_for_profile("review-workflow") == 333


@pytest.mark.asyncio
async def test_agent_workflow_context_limit_blocks_before_orchestrator():
    cfg = FusionConfig(coding_workflow_max_input_chars=5)
    core = FusionCore(cfg)
    core.agent_orchestrator = FakeAgentOrchestrator()

    action = await core.execute(_task("coding-workflow", text="too long for limit"))

    assert action.type == "final_answer"
    assert action.rationale_summary == "context_limit_exceeded"
    assert action.trace["profile"] == "coding-workflow"
    assert action.trace["limit_chars"] == 5
    assert core.agent_orchestrator.calls == []


@pytest.mark.asyncio
async def test_existing_fast_agent_dispatch_is_unchanged():
    cfg = FusionConfig(prefer_mlx_formatter=False)
    core = FusionCore(cfg)
    core.ollama = MagicMock()
    core.ollama.chat_completion.return_value = {
        "choices": [{"message": {"content": "fast answer"}}]
    }

    action = await core.execute(_task("fast-agent", "hello"))

    assert action.type == "final_answer"
    assert action.text == "fast answer"
    core.ollama.chat_completion.assert_called_once()
    assert core.ollama.chat_completion.call_args.kwargs["model"] == cfg.model_fast

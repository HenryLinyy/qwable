"""Core agent runtime orchestration without HTTP route integration."""

from __future__ import annotations
import inspect
import json
import re
from typing import Any

from qwable.agent_prompts import (
    build_executor_messages,
    build_finalizer_messages,
    build_plan_critic_messages,
    build_planner_messages,
    build_repair_messages,
)
from qwable.agent_state import (
    AgentArtifact,
    AgentFailure,
    AgentRun,
    AgentStep,
    new_id,
)
from qwable.agent_store import AgentStore
from qwable.config import FusionConfig
from qwable.context_compactor import ContextCompactor
from qwable.model_roles import RoleSelection, WorkflowStage
from qwable.model_selector import ModelSelector
from qwable.patch_protocol import validate_tool_call
from qwable.repair_loop import RepairLoop
from qwable.schemas import FusionAction, ParsedAgentTask, ToolResult
from qwable.test_runner import TestCommandPlanner


def extract_agent_run_id(task: ParsedAgentTask) -> str | None:
    metadata = task.raw_request.get("metadata") or {}
    if isinstance(metadata, dict):
        value = metadata.get("agent_run_id")
        if isinstance(value, str) and value.startswith("run_"):
            return value
    return None


class AgentOrchestrator:
    def __init__(
        self,
        config: FusionConfig,
        model_client,
        model_selector: ModelSelector,
        store: AgentStore,
        compactor: ContextCompactor,
    ):
        self.config = config
        self.model_client = model_client
        self.model_selector = model_selector
        self.store = store
        self.compactor = compactor
        self.test_command_planner = TestCommandPlanner()
        self.repair_loop = RepairLoop(config)
        self.core_last_used_model: str | None = None

    async def run(self, task: ParsedAgentTask, workflow: str) -> FusionAction:
        run = self._load_or_create_run(task, workflow)
        context_pack = self.compactor.build_pack(task, workflow)
        run.artifacts.append(
            AgentArtifact(
                artifact_id=new_id("artifact"),
                run_id=run.run_id,
                kind="context_pack",
                content=context_pack.to_prompt_text(self.config.agent_context_pack_max_chars),
                metadata={"workflow": workflow},
            )
        )

        if workflow == "review-workflow":
            return await self._run_review_workflow(run, context_pack, task)

        if task.tool_results:
            self._append_tool_results(run, task.tool_results)
            latest_result = task.tool_results[-1]
            if _is_test_result(latest_result):
                if latest_result.is_error:
                    return await self._handle_test_failure(run, context_pack, latest_result, workflow)
                return await self._finalize(run, context_pack, workflow)

            self._mark_current_step_done(run, latest_result)
            if latest_result.is_error:
                return self._tool_result_error_action(run, workflow, latest_result)
            if workflow == "coding-workflow":
                if _is_mutating_tool_result(latest_result):
                    test_action = self._build_test_tool_action(
                        run,
                        context_pack,
                        workflow,
                        default_command="python -m pytest",
                    )
                    if test_action is not None:
                        return test_action
                    return await self._finalize(
                        run,
                        context_pack,
                        workflow,
                        extra_trace={"test_status": "test_not_run"},
                    )
                if self._advance_to_next_step(run):
                    return await self._execute_current_step(run, context_pack, workflow)
                test_action = self._build_test_tool_action(run, context_pack, workflow)
                if test_action is not None:
                    return test_action
            if workflow == "agentic-workflow" and self._advance_to_next_step(run):
                return await self._execute_current_step(run, context_pack, workflow)
            return await self._finalize(
                run,
                context_pack,
                workflow,
                extra_trace={
                    "test_status": (
                        "test_not_required" if workflow == "agentic-workflow" else "test_not_run"
                    )
                },
            )

        planner_selection = self.model_selector.select(workflow, "planner")
        planner_messages = build_planner_messages(task, context_pack)
        planner_response = await self._call_with_fallback(planner_selection, planner_messages)
        raw_plan = _response_content(planner_response)

        try:
            run.plan = parse_planner_steps(raw_plan)
        except Exception as exc:
            run.status = "failed"
            failure = AgentFailure(
                stage="planner",
                message=f"planner_json_parse_failed: {exc}",
                metadata={"reason": "planner_json_parse_failed", "raw": raw_plan},
            )
            run.failures.append(failure)
            run.trace = self._trace(
                run,
                stage="planner",
                selection=planner_selection,
                status="failed",
                extra={"raw_planner_output": raw_plan},
            )
            self.store.save_run(run)
            return FusionAction(
                type="final_answer",
                text="planner_json_parse_failed",
                tool_name=None,
                tool_input=None,
                confidence=0.0,
                rationale_summary="Planner output could not be parsed as required JSON.",
                trace=run.trace,
            )

        run.status = "executing"
        run.current_step_index = 0
        run.artifacts.append(
            AgentArtifact(
                artifact_id=new_id("artifact"),
                run_id=run.run_id,
                kind="plan",
                content=raw_plan,
                metadata={"step_count": len(run.plan)},
            )
        )

        current_step = run.current_step()
        if current_step is None:
            run.status = "failed"
            run.trace = self._trace(run, stage="planner", selection=planner_selection, status="failed")
            self.store.save_run(run)
            return FusionAction(
                type="final_answer",
                text="planner_json_parse_failed: no steps",
                tool_name=None,
                tool_input=None,
                confidence=0.0,
                rationale_summary="Planner returned no executable steps.",
                trace=run.trace,
            )

        if workflow in {"agentic-workflow", "coding-workflow"}:
            critic_action = await self._review_plan(run, context_pack, workflow)
            if critic_action is not None:
                return critic_action

        return await self._execute_current_step(run, context_pack, workflow)

    def _executor_repair_selection(self, workflow: str, stage: str) -> RoleSelection:
        """v1.8: route the executor/repair stages through the Qwable-aware
        stage selector (``select_for_stage``) while keeping the v1.7
        ``RoleSelection`` shape the rest of the orchestrator consumes.

        When ENABLE_QWABLE_EXECUTOR is on, the EXECUTE_PATCH / REPAIR_PATCH
        chain puts Qwable first (then qwen3-coder-next, then agentic-mlx);
        when off it resolves to qwen3-coder-next, matching the v1.7 default.

        Falls back to the v1.7 ``select()`` result when the selector does not
        expose ``select_for_stage`` (e.g. test fakes) or stage resolution
        fails, so existing behavior is preserved.
        """
        base = self.model_selector.select(workflow, stage)
        select_for_stage = getattr(self.model_selector, "select_for_stage", None)
        if select_for_stage is None:
            return base
        wf_stage = (
            WorkflowStage.EXECUTE_PATCH if stage == "executor" else WorkflowStage.REPAIR_PATCH
        )
        try:
            selected = select_for_stage(wf_stage)
        except Exception:
            return base
        full_chain = [selected.model_name, *selected.fallback_chain]
        temperature = selected.generation_config.get("temperature", base.temperature)
        return RoleSelection(
            workflow=workflow,
            stage=stage,
            role=selected.role,
            model=selected.model_name,
            fallback_chain=full_chain,
            max_tokens=base.max_tokens,
            temperature=temperature,
            reason=selected.reason,
        )

    async def _call_with_fallback(self, selection: RoleSelection, messages: list[dict]) -> dict:
        errors = []
        for model in selection.fallback_chain:
            try:
                self.core_last_used_model = model
                result = self.model_client.chat_completion(
                    model=model,
                    messages=messages,
                    max_tokens=selection.max_tokens,
                    temperature=selection.temperature,
                )
                if inspect.isawaitable(result):
                    result = await result
                return result
            except Exception as exc:
                errors.append({"model": model, "error": str(exc)})
                continue
        raise RuntimeError(f"all fallback models failed: {errors}")

    def _append_tool_results(self, run: AgentRun, tool_results: list[ToolResult]) -> None:
        step = run.current_step()
        for result in tool_results:
            name = result.name or "unknown"
            run.artifacts.append(
                AgentArtifact(
                    artifact_id=new_id("artifact"),
                    run_id=run.run_id,
                    kind="tool_result",
                    content=result.content,
                    metadata={
                        "name": name,
                        "tool_call_id": result.tool_call_id,
                        "is_error": result.is_error,
                    },
                )
            )
            if step is not None:
                step.evidence.append(f"{name}:{result.content}")

    def _mark_current_step_done(self, run: AgentRun, result: ToolResult) -> None:
        step = run.current_step()
        if step is None:
            return
        if result.is_error:
            step.status = "failed"
            step.error = result.content
            run.status = "blocked"
            return
        step.status = "done"
        step.output = result.content

    async def _review_plan(self, run: AgentRun, context_pack, workflow: str) -> FusionAction | None:
        selection = self.model_selector.select(workflow, "plan_critic")
        run.status = "reviewing_plan"
        critic_messages = build_plan_critic_messages(run, context_pack)
        critic_response = await self._call_with_fallback(selection, critic_messages)
        raw_review = _response_content(critic_response)
        run.artifacts.append(
            AgentArtifact(
                artifact_id=new_id("artifact"),
                run_id=run.run_id,
                kind="plan_review",
                content=raw_review,
                metadata={"stage": "plan_critic"},
            )
        )

        if not _critic_has_fatal_blocker(raw_review):
            run.status = "executing"
            return None

        run.status = "blocked"
        current_step = run.current_step()
        if current_step is not None:
            current_step.status = "blocked"
            current_step.error = "fatal_blocker"
        run.failures.append(
            AgentFailure(
                stage="plan_critic",
                message="plan_critic_blocked:fatal_blocker",
                metadata={"reason": "fatal_blocker", "raw": raw_review},
            )
        )
        run.trace = self._trace(
            run,
            stage="plan_critic",
            selection=selection,
            status="blocked",
            extra={"fatal_blocker": True},
        )
        self.store.save_run(run)
        return FusionAction(
            type="final_answer",
            text="plan_critic_blocked:fatal_blocker",
            tool_name=None,
            tool_input=None,
            confidence=0.0,
            rationale_summary="Plan critic found a fatal blocker.",
            trace=run.trace,
        )

    async def _execute_current_step(self, run: AgentRun, context_pack, workflow: str) -> FusionAction:
        current_step = run.current_step()
        selection = self._executor_repair_selection(workflow, "executor")
        if current_step is None:
            run.status = "failed"
            run.failures.append(
                AgentFailure(
                    stage="executor",
                    message="executor_step_missing",
                    metadata={"reason": "executor_step_missing"},
                )
            )
            run.trace = self._trace(run, stage="executor", selection=selection, status="failed")
            self.store.save_run(run)
            return FusionAction(
                type="final_answer",
                text="executor_step_missing",
                tool_name=None,
                tool_input=None,
                confidence=0.0,
                rationale_summary="No current step was available for execution.",
                trace=run.trace,
            )

        run.status = "executing"
        current_step.status = "running"
        current_step.attempt_count += 1
        executor_messages = build_executor_messages(run, context_pack, current_step)
        executor_response = await self._call_with_fallback(selection, executor_messages)
        raw_executor = _response_content(executor_response)
        try:
            executor_payload = _parse_json_object(raw_executor)
            tool_call = _extract_tool_call(executor_payload)
        except ValueError as exc:
            return self._malformed_executor_tool_call_action(
                run,
                raw_executor,
                str(exc),
                selection,
                stage="executor",
            )

        if tool_call is not None:
            return self._tool_call_action(run, tool_call, selection, stage="executor")

        run.status = "completed"
        current_step.status = "done"
        current_step.output = raw_executor
        run.trace = self._trace(run, stage="executor", selection=selection, status="completed")
        self.store.save_run(run)
        return FusionAction(
            type="final_answer",
            text=raw_executor,
            tool_name=None,
            tool_input=None,
            confidence=0.8,
            rationale_summary="agent_completed",
            trace=run.trace,
        )

    def _advance_to_next_step(self, run: AgentRun) -> bool:
        next_index = run.current_step_index + 1
        if next_index >= len(run.plan):
            return False
        run.current_step_index = next_index
        return True

    def _build_test_tool_action(
        self,
        run: AgentRun,
        context_pack,
        workflow: str,
        *,
        default_command: str | None = None,
    ) -> FusionAction | None:
        commands = self.test_command_planner.infer_test_commands(context_pack, workflow)
        if not commands and default_command:
            commands = [default_command]
        if not commands:
            return None

        selection = self.model_selector.select(workflow, "test")
        tool_call = self.test_command_planner.build_test_tool_call(commands[0])
        tool_name = tool_call["tool_name"]
        tool_input = tool_call["tool_input"]
        ok, reason = validate_tool_call(tool_name, tool_input)
        if not ok:
            run.status = "blocked"
            current_step = run.current_step()
            if current_step is not None:
                current_step.status = "blocked"
                current_step.error = reason
            run.failures.append(
                AgentFailure(
                    stage="test",
                    message=f"test_tool_call_rejected: {reason}",
                    metadata={"tool_name": tool_name, "tool_input": tool_input},
                )
            )
            run.trace = self._trace(
                run,
                stage="test",
                selection=selection,
                status="blocked",
                extra={"tool_validation_error": reason},
            )
            self.store.save_run(run)
            return FusionAction(
                type="final_answer",
                text=f"test_tool_call_rejected: {reason}",
                tool_name=None,
                tool_input=None,
                confidence=0.0,
                rationale_summary="Test planner produced a rejected test tool call.",
                trace=run.trace,
            )

        limit_action = self._tool_call_limit_action(
            run,
            selection,
            stage="test",
            tool_name=tool_name,
            tool_input=tool_input,
        )
        if limit_action is not None:
            return limit_action

        run.status = "testing"
        run.tool_call_count += 1
        run.artifacts.append(
            AgentArtifact(
                artifact_id=new_id("artifact"),
                run_id=run.run_id,
                kind="tool_call",
                content=json.dumps(
                    {"tool_name": tool_name, "tool_input": tool_input},
                    ensure_ascii=False,
                    sort_keys=True,
                ),
                metadata={"stage": "test"},
            )
        )
        run.trace = self._trace(run, stage="test", selection=selection, status="testing")
        self.store.save_run(run)
        return FusionAction(
            type="tool_call",
            text=None,
            tool_name=tool_name,
            tool_input=tool_input,
            confidence=0.8,
            rationale_summary="test_requested_tool",
            trace=run.trace,
        )

    async def _handle_test_failure(
        self,
        run: AgentRun,
        context_pack,
        latest_result: ToolResult,
        workflow: str,
    ) -> FusionAction:
        selection = self._executor_repair_selection(workflow, "repair")
        decision = self.repair_loop.decide(run, latest_result.content)
        if not decision.should_repair:
            run.status = "blocked"
            current_step = run.current_step()
            if current_step is not None:
                current_step.status = "blocked"
                current_step.error = latest_result.content
            run.failures.append(
                AgentFailure(
                    stage="repair",
                    message=f"repair_blocked:{decision.reason}",
                    metadata={"failure_summary": decision.failure_summary},
                )
            )
            run.trace = self._trace(
                run,
                stage="repair",
                selection=selection,
                status="blocked",
                extra={
                    "repair_decision": decision.reason,
                    "failure_summary": decision.failure_summary,
                },
            )
            self.store.save_run(run)
            return FusionAction(
                type="final_answer",
                text=f"repair_blocked:{decision.reason}",
                tool_name=None,
                tool_input=None,
                confidence=0.0,
                rationale_summary=decision.failure_summary,
                trace=run.trace,
            )

        run.repair_count += 1
        run.status = "repairing"
        repair_messages = build_repair_messages(run, context_pack, latest_result.content)
        repair_response = await self._call_with_fallback(selection, repair_messages)
        raw_repair = _response_content(repair_response)
        try:
            repair_payload = _parse_json_object(raw_repair)
            tool_call = _extract_tool_call(repair_payload)
        except ValueError:
            # Malformed repair output → treat as a missing tool_call and
            # route to the graceful "blocked" path below instead of 500-ing.
            tool_call = None
        if tool_call is None:
            run.status = "blocked"
            run.failures.append(
                AgentFailure(
                    stage="repair",
                    message="repair_tool_call_missing",
                    metadata={"raw": raw_repair},
                )
            )
            run.trace = self._trace(
                run,
                stage="repair",
                selection=selection,
                status="blocked",
                extra={"raw_repair_output": raw_repair},
            )
            self.store.save_run(run)
            return FusionAction(
                type="final_answer",
                text="repair_tool_call_missing",
                tool_name=None,
                tool_input=None,
                confidence=0.0,
                rationale_summary="Repair model did not return a tool call.",
                trace=run.trace,
            )

        return self._tool_call_action(run, tool_call, selection, stage="repair")

    def _tool_result_error_action(
        self,
        run: AgentRun,
        workflow: str,
        result: ToolResult,
    ) -> FusionAction:
        selection = self._executor_repair_selection(workflow, "executor")
        run.status = "blocked"
        run.failures.append(
            AgentFailure(
                stage="executor",
                message=f"tool_result_failed:{result.name or 'unknown'}",
                metadata={
                    "reason": "tool_result_failed",
                    "tool_name": result.name,
                    "content": result.content,
                },
            )
        )
        run.trace = self._trace(
            run,
            stage="executor",
            selection=selection,
            status="blocked",
            extra={"tool_result_error": result.content},
        )
        self.store.save_run(run)
        return FusionAction(
            type="final_answer",
            text=f"tool_result_failed:{result.name or 'unknown'}",
            tool_name=None,
            tool_input=None,
            confidence=0.0,
            rationale_summary="Tool result reported an error before tests could run.",
            trace=run.trace,
        )

    async def _run_review_workflow(
        self,
        run: AgentRun,
        context_pack,
        task: ParsedAgentTask,
    ) -> FusionAction:
        run.status = "reviewing"
        reviewer_selection = self.model_selector.select(run.workflow, "reviewer")
        reviewer_response = await self._call_with_fallback(
            reviewer_selection,
            _build_review_messages(task, context_pack),
        )
        raw_review = _response_content(reviewer_response)
        review_text, suppressed_tool_call = _review_text_without_tool_call(raw_review)
        run.artifacts.append(
            AgentArtifact(
                artifact_id=new_id("artifact"),
                run_id=run.run_id,
                kind="review",
                content=review_text,
                metadata={
                    "stage": "reviewer",
                    "suppressed_tool_call": suppressed_tool_call,
                },
            )
        )

        judge_selection = self.model_selector.select(run.workflow, "judge")
        judge_response = await self._call_with_fallback(
            judge_selection,
            _build_review_judge_messages(context_pack, review_text),
        )
        judge_text = _response_content(judge_response)
        run.artifacts.append(
            AgentArtifact(
                artifact_id=new_id("artifact"),
                run_id=run.run_id,
                kind="review",
                content=judge_text,
                metadata={"stage": "judge"},
            )
        )

        finalizer_selection = self.model_selector.select(run.workflow, "finalizer")
        run.status = "finalizing"
        finalizer_response = await self._call_with_fallback(
            finalizer_selection,
            build_finalizer_messages(run, context_pack),
        )
        final_text = _response_content(finalizer_response)
        run.status = "completed"
        run.artifacts.append(
            AgentArtifact(
                artifact_id=new_id("artifact"),
                run_id=run.run_id,
                kind="final_report",
                content=final_text,
                metadata={"stage": "finalizer", "review_only": True},
            )
        )
        extra_trace = {"review_only": True}
        if suppressed_tool_call:
            extra_trace["suppressed_tool_call"] = suppressed_tool_call
        run.trace = self._trace(
            run,
            stage="finalizer",
            selection=finalizer_selection,
            status="completed",
            extra=extra_trace,
        )
        self.store.save_run(run)
        return FusionAction(
            type="final_answer",
            text=final_text,
            tool_name=None,
            tool_input=None,
            confidence=0.8,
            rationale_summary="review_completed",
            trace=run.trace,
        )

    async def _finalize(
        self,
        run: AgentRun,
        context_pack,
        workflow: str,
        *,
        extra_trace: dict[str, Any] | None = None,
    ) -> FusionAction:
        selection = self.model_selector.select(workflow, "finalizer")
        run.status = "finalizing"
        finalizer_messages = build_finalizer_messages(run, context_pack)
        finalizer_response = await self._call_with_fallback(selection, finalizer_messages)
        final_text = _response_content(finalizer_response)
        current_step = run.current_step()
        if current_step is not None and current_step.status not in {"blocked", "failed"}:
            current_step.status = "done"
        run.status = "completed"
        run.artifacts.append(
            AgentArtifact(
                artifact_id=new_id("artifact"),
                run_id=run.run_id,
                kind="final_report",
                content=final_text,
                metadata={"stage": "finalizer"},
            )
        )
        run.trace = self._trace(
            run,
            stage="finalizer",
            selection=selection,
            status="completed",
            extra=extra_trace,
        )
        self.store.save_run(run)
        return FusionAction(
            type="final_answer",
            text=final_text,
            tool_name=None,
            tool_input=None,
            confidence=0.8,
            rationale_summary="agent_completed",
            trace=run.trace,
        )

    def _step_result_action_for_executor_suppression(
        self,
        run: AgentRun,
        text: str,
        selection: RoleSelection,
        *,
        stage: str,
        trace_extra: dict[str, Any] | None = None,
    ) -> FusionAction:
        current_step = run.current_step()
        run.status = "completed"
        if current_step is not None:
            current_step.status = "done"
            current_step.output = text
        run.trace = self._trace(
            run,
            stage=stage,
            selection=selection,
            status="completed",
            extra=trace_extra,
        )
        self.store.save_run(run)
        return FusionAction(
            type="final_answer",
            text=text,
            tool_name=None,
            tool_input=None,
            confidence=0.7,
            rationale_summary="executor_tool_call_suppressed_as_step_result",
            trace=run.trace,
        )

    def _rejected_shell_as_step_result_action(
        self,
        run: AgentRun,
        tool_name: str,
        tool_input: dict[str, Any],
        selection: RoleSelection,
        *,
        stage: str,
        reason: str,
    ) -> FusionAction:
        command = tool_input.get("command") if isinstance(tool_input, dict) else None
        content = (
            "The model requested a shell command, but this runtime only allows "
            "a narrow test-command subset. Returning safe guidance instead of "
            "blocking the workflow."
        )
        if isinstance(command, str) and command.strip():
            content += f"\n\nSuggested command for the user to run manually:\n{command.strip()}"
        text = json.dumps(
            {
                "step_result": {
                    "status": "done",
                    "summary": "Shell tool request suppressed by policy; returned safe guidance.",
                    "content": content,
                }
            },
            ensure_ascii=False,
            indent=2,
        )
        return self._step_result_action_for_executor_suppression(
            run,
            text,
            selection,
            stage=stage,
            trace_extra={
                "suppressed_tool_call": tool_name,
                "tool_validation_error": reason,
            },
        )

    def _malformed_executor_tool_call_action(
        self,
        run: AgentRun,
        raw_executor: str,
        error: str,
        selection: RoleSelection,
        *,
        stage: str,
    ) -> FusionAction:
        text = json.dumps(
            {
                "step_result": {
                    "status": "done",
                    "summary": "Executor emitted a malformed tool_call; retained raw output instead of failing the request.",
                    "content": raw_executor,
                }
            },
            ensure_ascii=False,
            indent=2,
        )
        return self._step_result_action_for_executor_suppression(
            run,
            text,
            selection,
            stage=stage,
            trace_extra={"malformed_tool_call_error": error},
        )

    def _tool_call_action(
        self,
        run: AgentRun,
        tool_call: dict[str, Any],
        selection: RoleSelection,
        *,
        stage: str,
    ) -> FusionAction:
        tool_name = tool_call["name"]
        tool_input = tool_call["input"]
        ok, reason = validate_tool_call(tool_name, tool_input)
        if not ok:
            if reason == "shell_command_not_allowed":
                return self._rejected_shell_as_step_result_action(
                    run,
                    tool_name,
                    tool_input,
                    selection,
                    stage=stage,
                    reason=reason,
                )
            run.status = "blocked"
            current_step = run.current_step()
            if current_step is not None:
                current_step.status = "blocked"
                current_step.error = reason
            run.failures.append(
                AgentFailure(
                    stage=stage,
                    message=f"tool_call_rejected: {reason}",
                    metadata={"tool_name": tool_name, "tool_input": tool_input},
                )
            )
            run.trace = self._trace(
                run,
                stage=stage,
                selection=selection,
                status="blocked",
                extra={"tool_validation_error": reason},
            )
            self.store.save_run(run)
            return FusionAction(
                type="final_answer",
                text=f"tool_call_rejected: {reason}",
                tool_name=None,
                tool_input=None,
                confidence=0.0,
                rationale_summary="Model requested a rejected tool call.",
                trace=run.trace,
            )

        limit_action = self._tool_call_limit_action(
            run,
            selection,
            stage=stage,
            tool_name=tool_name,
            tool_input=tool_input,
        )
        if limit_action is not None:
            return limit_action

        run.status = "waiting_for_tool"
        current_step = run.current_step()
        if current_step is not None:
            current_step.status = "waiting_for_tool"
        run.tool_call_count += 1
        run.artifacts.append(
            AgentArtifact(
                artifact_id=new_id("artifact"),
                run_id=run.run_id,
                kind="tool_call",
                content=json.dumps(
                    {"tool_name": tool_name, "tool_input": tool_input},
                    ensure_ascii=False,
                    sort_keys=True,
                ),
                metadata={"stage": stage},
            )
        )
        run.trace = self._trace(run, stage=stage, selection=selection, status="waiting_for_tool")
        self.store.save_run(run)
        return FusionAction(
            type="tool_call",
            text=None,
            tool_name=tool_name,
            tool_input=tool_input,
            confidence=0.8,
            rationale_summary=f"{stage}_requested_tool",
            trace=run.trace,
        )

    def _tool_call_limit_action(
        self,
        run: AgentRun,
        selection: RoleSelection,
        *,
        stage: str,
        tool_name: str,
        tool_input: dict[str, Any],
    ) -> FusionAction | None:
        if run.tool_call_count < self.config.agent_max_tool_calls:
            return None

        run.status = "failed"
        current_step = run.current_step()
        if current_step is not None:
            current_step.status = "failed"
            current_step.error = "tool_call_limit_exceeded"
        run.failures.append(
            AgentFailure(
                stage=stage,
                message="tool_call_limit_exceeded",
                metadata={
                    "reason": "tool_call_limit_exceeded",
                    "tool_name": tool_name,
                    "tool_input": tool_input,
                    "tool_call_limit": self.config.agent_max_tool_calls,
                },
            )
        )
        run.trace = self._trace(
            run,
            stage=stage,
            selection=selection,
            status="failed",
            extra={"tool_call_limit": self.config.agent_max_tool_calls},
        )
        self.store.save_run(run)
        return FusionAction(
            type="final_answer",
            text="tool_call_limit_exceeded",
            tool_name=None,
            tool_input=None,
            confidence=0.0,
            rationale_summary="tool_call_limit_exceeded",
            trace=run.trace,
        )

    def _load_or_create_run(self, task: ParsedAgentTask, workflow: str) -> AgentRun:
        run_id = extract_agent_run_id(task)
        if run_id:
            existing = self.store.load_run(run_id)
            if existing is not None:
                return existing
        return AgentRun.create(goal=task.text, workflow=workflow)

    def _trace(
        self,
        run: AgentRun,
        *,
        stage: str,
        selection: RoleSelection,
        status: str,
        extra: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        trace = {
            "agent_runtime": True,
            "agent_run_id": run.run_id,
            "workflow": run.workflow,
            "agent_status": status,
            "stage": stage,
            "model_role": selection.role.value,
            "selected_model": self.core_last_used_model or selection.model,
            "model": self.core_last_used_model or selection.model,
            "fallback_chain": selection.fallback_chain,
            "current_step_index": run.current_step_index,
            "repair_count": run.repair_count,
            "tool_call_count": run.tool_call_count,
        }
        if extra:
            trace.update(extra)
        return trace


def parse_planner_steps(raw_output: str) -> list[AgentStep]:
    payload = _parse_json_object(raw_output)
    raw_steps = payload.get("steps")
    if not isinstance(raw_steps, list) or not raw_steps:
        raise ValueError("planner output must contain non-empty steps")

    steps: list[AgentStep] = []
    for item in raw_steps:
        if not isinstance(item, dict):
            raise ValueError("planner step must be an object")
        title = item["title"]
        intent = item["intent"]
        if not isinstance(title, str) or not isinstance(intent, str):
            raise ValueError("planner step title and intent must be strings")
        steps.append(
            AgentStep(
                step_id=new_id("step"),
                title=title,
                intent=intent,
                required_tools=_string_list(item.get("required_tools", [])),
                success_criteria=_string_list(item.get("success_criteria", [])),
                failure_criteria=_string_list(item.get("failure_criteria", [])),
            )
        )
    return steps


def _response_content(response: Any) -> str:
    """Extract the assistant's reply text from a model response.

    Handles four shapes, in priority order:
      1) A raw string (legacy / test mocks)
      2) An OpenAI-compatible dict with `content` (the normal case)
      3) An OpenAI-compatible dict with `reasoning_content` (LM Studio
         reasoning models such as qwen3.6-35b-a3b and deepseek-r1-distill-*
         emit their final structured output in `reasoning_content` when
         chain-of-thought consumes the entire max_tokens budget — at which
         point `content` is empty. Without this fallback the gateway's
         planner / critic / judge stages fail with
         `*_json_parse_failed` on every reasoning-model call.)
      4) Fallback to str(response) so we never silently return ""
    """
    if isinstance(response, str):
        return response
    if isinstance(response, dict):
        content = response.get("content")
        if isinstance(content, str) and content:
            return content
        # Fallback: reasoning models (LM Studio) put final output in reasoning_content
        # when thinking exhausts the token budget. Try it before giving up.
        reasoning = response.get("reasoning_content")
        if isinstance(reasoning, str) and reasoning:
            return reasoning
        message = response.get("message")
        if isinstance(message, dict):
            msg_content = message.get("content")
            if isinstance(msg_content, str) and msg_content:
                return msg_content
            msg_reasoning = message.get("reasoning_content")
            if isinstance(msg_reasoning, str) and msg_reasoning:
                return msg_reasoning
        choices = response.get("choices")
        if isinstance(choices, list) and choices:
            first = choices[0]
            if isinstance(first, dict):
                message = first.get("message")
                if isinstance(message, dict):
                    msg_content = message.get("content")
                    if isinstance(msg_content, str) and msg_content:
                        return msg_content
                    msg_reasoning = message.get("reasoning_content")
                    if isinstance(msg_reasoning, str) and msg_reasoning:
                        return msg_reasoning
                if isinstance(first.get("text"), str):
                    return first["text"]
    return str(response)


def _parse_json_object(text: str) -> dict[str, Any]:
    """Extract the first parseable JSON object from a possibly noisy model output.

    Handles three common shapes produced by reasoning models:
      1. Pure JSON (legacy)
      2. A single ```json ... ``` fence (current code handled this)
      3. Thinking prose + one or more ```json ... ``` fences (e.g. qwen3.6
         planning with reasoning_content = 10K chars of "Here's a thinking
         process: ..." followed by a real plan inside a fence). The real
         plan is the LAST parseable JSON object found, so we try the fenced
         candidates from newest to oldest before falling back to the outer
         brace match.
    """
    cleaned = text.strip()

    # Strip a single wrapping fence if the entire output is fenced.
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        if lines and lines[0].lstrip().startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        single = "\n".join(lines).strip()
        try:
            payload = json.loads(single)
            if isinstance(payload, dict):
                return payload
        except (json.JSONDecodeError, ValueError):
            pass

    # Multi-fence case: try each ```json``` block from last to first.
    fence_matches = list(
        re.finditer(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", cleaned, re.IGNORECASE)
    )
    for m in reversed(fence_matches):
        candidate = m.group(1)
        try:
            payload = json.loads(candidate)
        except (json.JSONDecodeError, ValueError):
            continue
        if isinstance(payload, dict):
            return payload

    # Fall back to scanning for a balanced JSON object. The naive
    # find('{')..rfind('}') slice spans TWO objects when reasoning prose
    # contains an earlier brace (common for reasoning models this targets),
    # making json.loads fail. Instead, raw_decode at each '{' and keep the
    # last object that decodes — preferring the last non-empty one.
    decoder = json.JSONDecoder()
    candidates: list[dict] = []
    idx = cleaned.find("{")
    while idx != -1:
        try:
            obj, end = decoder.raw_decode(cleaned, idx)
        except json.JSONDecodeError:
            idx = cleaned.find("{", idx + 1)
            continue
        if isinstance(obj, dict):
            candidates.append(obj)
        # Advance PAST the decoded object so nested braces inside it aren't
        # re-parsed (which would wrongly return an inner object such as a
        # tool_call value instead of the whole object).
        idx = cleaned.find("{", end)
    if candidates:
        for obj in reversed(candidates):
            if obj:
                return obj
        return candidates[-1]

    raise ValueError("no JSON object found in planner output")


def _extract_tool_call(payload: dict[str, Any]) -> dict[str, Any] | None:
    raw = payload.get("tool_call")

    # Qwable / Fable-style local models often emit a tool request as
    # {"tool": "terminal", "action": "run_command", "arguments": {...}}
    # rather than the v1.7-native {"tool_call": {"name": ..., "input": ...}}.
    # Normalize that shape so validation can make the allow/deny decision
    # instead of silently treating it as a completed step.
    has_tool_intent = any(key in payload for key in ("tool", "action", "arguments"))
    if raw is None and has_tool_intent:
        raw = _tool_action_arguments_to_tool_call(payload)
        if raw is None:
            # The model asked for a tool but in an unrecognized shape — surface
            # it as malformed/blocked instead of silently treating it as a
            # completed step (which would skip validation entirely).
            raise ValueError("unrecognized tool request shape")

    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise ValueError("tool_call must be an object")
    name = raw.get("name") or raw.get("tool_name")
    tool_input = raw.get("input") if "input" in raw else raw.get("tool_input")
    if not isinstance(name, str):
        raise ValueError("tool_call name must be a string")
    if isinstance(tool_input, str):
        try:
            parsed_input = json.loads(tool_input)
        except json.JSONDecodeError:
            parsed_input = None
        if isinstance(parsed_input, dict):
            tool_input = parsed_input
    if not isinstance(tool_input, dict):
        raise ValueError("tool_call input must be an object")
    return {"name": name, "input": tool_input}


def _tool_action_arguments_to_tool_call(payload: dict[str, Any]) -> dict[str, Any] | None:
    tool = payload.get("tool")
    action = payload.get("action")
    arguments = payload.get("arguments")
    if not isinstance(tool, str) or not isinstance(arguments, dict):
        return None

    tool_norm = tool.strip().lower()
    action_norm = action.strip().lower() if isinstance(action, str) else ""

    # Terminal/shell aliases. Keep only the command because patch_protocol
    # validates the rest; descriptions/run flags are model prose, not executor input.
    if tool_norm in {"terminal", "shell", "bash", "command"} or action_norm in {
        "run_command",
        "execute",
        "shell",
    }:
        command = arguments.get("command") or arguments.get("cmd")
        if isinstance(command, str):
            return {"name": "shell", "input": {"command": command}}

    # Common direct patch-protocol aliases.
    if tool_norm in {"read_file", "search_files", "list_files", "edit_file", "apply_patch", "run_tests"}:
        return {"name": tool_norm, "input": arguments}

    return None


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, str)]


def _is_test_result(result: ToolResult) -> bool:
    return result.name == "run_tests"


def _is_mutating_tool_result(result: ToolResult) -> bool:
    return result.name in {"edit_file", "apply_patch"}


def _critic_has_fatal_blocker(raw_review: str) -> bool:
    try:
        payload = _parse_json_object(raw_review)
    except Exception:
        return False
    return payload.get("fatal_blocker") is True


def _build_review_messages(task: ParsedAgentTask, context_pack) -> list[dict]:
    return [
        {
            "role": "system",
            "content": (
                "You are Qwable Review Agent. Review only. "
                "Do not output apply_patch, shell, or any mutating tool_call. "
                "Use this format: 1. Blockers 2. High-risk issues "
                "3. Medium-risk issues 4. Missing tests "
                "5. Compatibility risks 6. Suggested minimal fixes."
            ),
        },
        {
            "role": "user",
            "content": "\n\n".join(
                [
                    f"USER_GOAL:\n{task.text}",
                    context_pack.to_prompt_text(max_chars=64000),
                ]
            ),
        },
    ]


def _build_review_judge_messages(context_pack, review_text: str) -> list[dict]:
    return [
        {
            "role": "system",
            "content": (
                "You are Qwable Review Judge. Validate the review for evidence, "
                "risk severity, missing tests, and compatibility concerns. "
                "Do not propose patch tool calls."
            ),
        },
        {
            "role": "user",
            "content": "\n\n".join(
                [
                    "REVIEW_DRAFT:\n" + review_text,
                    context_pack.to_prompt_text(max_chars=64000),
                ]
            ),
        },
    ]


def _review_text_without_tool_call(raw_review: str) -> tuple[str, str | None]:
    try:
        payload = _parse_json_object(raw_review)
    except Exception:
        return raw_review, None

    tool_call = _extract_tool_call(payload)
    if tool_call is None:
        return raw_review, None
    tool_name = tool_call["name"]
    return (
        json.dumps(
            {
                "review_mode": "report_only",
                "suppressed_tool_call": tool_name,
                "finding": "Reviewer attempted a mutating tool call; retained as review evidence only.",
                "raw_review": payload,
            },
            ensure_ascii=False,
            sort_keys=True,
        ),
        tool_name,
    )

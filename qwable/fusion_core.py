"""Fusion Core — routes ParsedAgentTask to the correct agent profile."""

from qwable.schemas import ParsedAgentTask, FusionAction
from qwable.models import OllamaClient, DS4Client
from qwable.prompts import (
    FAST_AGENT_SYSTEM,
    FULL_AGENT_CODER_SYSTEM,
    FULL_AGENT_TOOLER_SYSTEM,
    FULL_AGENT_CRITIC_SYSTEM,
    FULL_AGENT_JUDGE_SYSTEM,
    HEAVY_AGENT_PRIMARY_SYSTEM,
    HEAVY_AGENT_CHECKER_SYSTEM,
    HEAVY_AGENT_CRITIC_SYSTEM,
    HEAVY_AGENT_JUDGE_SYSTEM,
    CHAT_AGENT_SYSTEM,
)
from qwable.action_parser import parse_action_from_text
from qwable.agent_router import AgentRouter
from qwable.config import FusionConfig
from qwable.agent_orchestrator import AgentOrchestrator
from qwable.agent_store import AgentStore
from qwable.context_compactor import ContextCompactor
from qwable.fusion_deliberation import run_fusion_agent
from qwable.fusion_presets import FusionPresetError, resolve_preset
from qwable.fusion_request import extract_fusion_request
from qwable.model_selector import ModelSelector
from qwable.static_fit import check_static_fit
from qwable.vision import format_vision_evidence
from qwable.vision_processor import VisionProcessor
from qwable.vision_router import select_vision_profile
from dataclasses import replace
from typing import Literal
import json
import logging

logger = logging.getLogger("qwable.fusion_core")


class FusionCore:
    """Routes tasks to the correct agent profile."""

    def __init__(self, config: FusionConfig):
        self.config = config
        self.ollama = OllamaClient(
            config.ollama_base_url,
            timeout=config.qwable_timeout_seconds,
            backend=config.local_model_backend,
            lmstudio_cli_path=config.lmstudio_cli_path,
            ttl_seconds=getattr(config, "lmstudio_ttl_seconds", None),
        )
        self.ds4 = DS4Client(config.ds4_base_url, timeout=config.ds4_timeout_seconds)
        self.vision = VisionProcessor(config, self.ollama)
        self.agent_router = AgentRouter(config)
        self.model_selector = ModelSelector(config)
        self.agent_store = AgentStore(config.agent_store_path)
        self.agent_store.init_schema()
        self.context_compactor = ContextCompactor(config)
        self.agent_orchestrator = AgentOrchestrator(
            config=config,
            model_client=self.ollama,
            model_selector=self.model_selector,
            store=self.agent_store,
            compactor=self.context_compactor,
        )
        # G12 micro D+E: track the most recent model used by any profile
        # (for /health introspection and trace breadcrumbs)
        self.last_used_model: str | None = None

    async def execute(self, task: ParsedAgentTask) -> FusionAction:
        """Execute a ParsedAgentTask and return a FusionAction."""
        if task.images and task.profile not in ("vision-fast", "vision-pro", "vision-heavy"):
            selected_vision_profile = select_vision_profile(task)
            if selected_vision_profile == "vision-heavy":
                return await self._run_vision_heavy_agent(task)
            if selected_vision_profile in ("vision-fast", "vision-pro"):
                task = await self._with_vision_evidence(task, selected_vision_profile)

        original_workflow = task.profile
        profile = self.agent_router.resolve_workflow(task, task.profile)
        routed_from_workflow = original_workflow if profile != original_workflow else None
        if routed_from_workflow is not None:
            task = replace(task, profile=profile)
        limit = self._context_limit_for_profile(profile)
        input_chars = self._task_input_chars(task)
        if limit is not None and input_chars > limit:
            return FusionAction(
                type="final_answer",
                text=f"context limit exceeded: {input_chars} chars exceeds {limit} chars for {profile}; compact the task or use a heavier profile",
                tool_name=None,
                tool_input=None,
                confidence=0.0,
                rationale_summary="context_limit_exceeded",
                trace={
                    "profile": profile,
                    "error": "context_limit_exceeded",
                    "input_chars": input_chars,
                    "limit_chars": limit,
                },
            )

        if profile == "fast-agent":
            # G11: Auto-route short text-only fast-agent requests through
            # formatter-mlx (gemma via MLX) to save the ~30s load time of
            # the standard qwen3.6 model. Bypassed when tools are present
            # or text is over the threshold.
            if self._should_use_mlx_formatter(task):
                return await self._run_formatter_mlx_agent(task)
            return await self._run_fast_agent(task)
        elif profile == "full-agent":
            return await self._run_full_agent(task)
        elif profile == "heavy-agent":
            return await self._run_heavy_agent(task)
        elif profile == "chat-agent":
            return await self._run_chat_agent(task)
        elif profile in ("vision-fast", "vision-pro"):
            return await self._run_vision_agent(task, profile)
        elif profile == "vision-heavy":
            return await self._run_vision_heavy_agent(task)
        elif profile == "agentic-pro":
            return await self._run_agentic_pro_agent(task)
        elif profile == "hermes-pro":
            return await self._run_hermes_pro_agent(task)
        elif profile == "agentic-mlx":
            return await self._run_agentic_mlx_agent(task)
        elif profile == "formatter-mlx":
            return await self._run_formatter_mlx_agent(task)
        elif profile == "fusion-agent":
            return await self._run_fusion_agent(task)
        elif profile == "agentic-workflow":
            return self._annotate_workflow_routing(
                await self._run_agentic_workflow(task),
                routed_from_workflow,
            )
        elif profile == "coding-workflow":
            return self._annotate_workflow_routing(
                await self._run_coding_workflow(task),
                routed_from_workflow,
            )
        elif profile == "review-workflow":
            return self._annotate_workflow_routing(
                await self._run_review_workflow(task),
                routed_from_workflow,
            )
        else:
            return FusionAction(
                type="final_answer",
                text=f"Unknown profile: {profile}",
                tool_name=None,
                tool_input=None,
                confidence=0.0,
                rationale_summary=None,
            )

    def _annotate_trace(self, action: FusionAction, model_id: str) -> FusionAction:
        """G12 micro D: record the model that produced this action.

        Updates both the instance attribute (for /health introspection) and
        the action's trace dict (for downstream observability).

        Returns the action unchanged so callers can chain `return self._annotate_trace(...)`.
        """
        self.last_used_model = model_id
        if action.trace is None:
            action.trace = {}
        action.trace["last_used_model"] = model_id
        return action

    def _should_use_mlx_formatter(self, task: ParsedAgentTask) -> bool:
        """G11: Decide if a fast-agent request should auto-route to formatter-mlx.

        Conditions (ALL must hold):
          - prefer_mlx_formatter is True (config flag)
          - task has no tools
          - task text is below mlx_formatter_max_chars threshold

        Returns True → caller should dispatch to _run_formatter_mlx_agent
                       instead of _run_fast_agent (saves qwen3.6 model load).
        """
        if not self.config.prefer_mlx_formatter:
            return False
        if task.tools:
            return False
        text_len = len((task.text or "").strip())
        return text_len < self.config.mlx_formatter_max_chars

    def _build_messages(self, system_prompt: str, task: ParsedAgentTask) -> list[dict]:
        """Build messages list from a ParsedAgentTask."""
        messages = [{"role": "system", "content": system_prompt}]

        # Add user message
        if task.text.strip():
            messages.append({"role": "user", "content": task.text})

        for index, evidence in enumerate(task.vision_evidence, start=1):
            messages.append({
                "role": "user",
                "content": format_vision_evidence(evidence, index),
            })

        # Tool results are user-provided evidence from the client loop. Keep
        # them prominent so models do not invent a different tool outcome.
        for tr in task.tool_results:
            messages.append({
                "role": "user",
                "content": self._format_tool_result_evidence(tr),
            })

        return messages

    def _format_tool_result_evidence(self, tr) -> str:
        return (
            "AUTHORITATIVE_TOOL_RESULT\n"
            f"source_protocol={tr.source_protocol}\n"
            f"tool_call_id={tr.tool_call_id or ''}\n"
            f"tool_name={tr.name or ''}\n"
            f"is_error={str(tr.is_error).lower()}\n"
            "content:\n"
            f"{tr.content}\n"
            "END_AUTHORITATIVE_TOOL_RESULT\n\n"
            "請只根據上述工具結果回答；不要改寫、補造、猜測或加入工具結果中不存在的項目。"
        )

    def _context_limit_for_profile(self, profile: str) -> int | None:
        if profile in ("fast-agent", "chat-agent"):
            return self.config.fast_max_input_chars
        if profile == "full-agent":
            return self.config.full_max_input_chars
        if profile == "heavy-agent":
            return self.config.heavy_max_input_chars
        if profile == "vision-fast":
            return self.config.vision_fast_max_input_chars
        if profile in ("vision-pro", "agentic-pro", "hermes-pro"):
            return self.config.vision_pro_max_input_chars
        if profile == "agentic-mlx":
            # Dedicated ceiling — qwen3.6:35b-a3b-nvfp4 supports 262144 (256K) ctx
            return self.config.agentic_mlx_max_input_chars
        if profile == "formatter-mlx":
            return self.config.fast_max_input_chars
        if profile == "vision-heavy":
            return self.config.heavy_max_input_chars
        if profile == "agentic-workflow":
            return self.config.agentic_workflow_max_input_chars
        if profile == "coding-workflow":
            return self.config.coding_workflow_max_input_chars
        if profile == "review-workflow":
            return self.config.review_workflow_max_input_chars
        return None

    def _task_input_chars(self, task: ParsedAgentTask) -> int:
        evidence_chars = sum(len(e.raw_text or e.summary or "") for e in task.vision_evidence)
        return len(task.text) + evidence_chars + sum(len(tr.content or "") for tr in task.tool_results)

    def _request_max_tokens(self, task: ParsedAgentTask, default: int) -> int:
        value = task.raw_request.get("max_tokens")
        if isinstance(value, int) and value > 0:
            return min(value, default)
        return default

    def _request_temperature(self, task: ParsedAgentTask, default: float = 0.7) -> float:
        value = task.raw_request.get("temperature")
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return float(value)
        return default

    def _tools_payload(self, task: ParsedAgentTask) -> list[dict] | None:
        """Convert ToolSpec to Ollama tool format, respecting tool_choice=none."""
        if task.raw_request.get("tool_choice") == "none" or not task.tools:
            return None
        tools_payload = []
        for t in task.tools:
            tools_payload.append({
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description or "",
                    "parameters": t.input_schema,
                },
            })
        return tools_payload

    def _action_from_ollama_response(self, response: dict, tools_payload: list[dict] | None = None) -> FusionAction:
        message = response.get("choices", [{}])[0].get("message", {})
        text = message.get("content", "")
        tool_calls_raw = message.get("tool_calls")
        if tool_calls_raw and tools_payload:
            return parse_action_from_text({"tool_calls": tool_calls_raw}, tools=tools_payload)
        return parse_action_from_text(text, tools=tools_payload)

    def _ollama_models_for_unload(self) -> list[str]:
        models = [
            self.config.model_fast,
            self.config.model_coder,
            self.config.model_tooler,
            self.config.model_critic,
            self.config.model_judge,
            self.config.model_formatter,
            self.config.model_vision_fast,
            self.config.model_vision_pro,
            self.config.model_vision_pro_fallback,
            self.config.model_agentic_pro,
            self.config.model_hermes_pro,
            self.config.model_agentic_mlx,
            self.config.model_formatter_mlx,
        ]
        return list(dict.fromkeys(model for model in models if model))

    def _release_ollama_before_ds4(self) -> None:
        unload_models = getattr(self.ollama, "unload_models", None)
        if callable(unload_models):
            unload_models(self._ollama_models_for_unload())

    async def _with_vision_evidence(self, task: ParsedAgentTask, profile: str) -> ParsedAgentTask:
        """Return task with VisionEvidence appended once."""
        if task.vision_evidence:
            return task
        if getattr(self.vision, "ollama", None) is not self.ollama:
            self.vision.ollama = self.ollama
        evidence = await self.vision.extract_evidence(task, profile)
        return replace(task, vision_evidence=[*task.vision_evidence, *evidence])

    def _vision_profile_for_heavy_task(self, task: ParsedAgentTask) -> str:
        routed = select_vision_profile(replace(task, profile="fast-agent"))
        if routed in ("vision-fast", "vision-pro"):
            return routed
        return "vision-pro"

    async def _run_vision_agent(self, task: ParsedAgentTask, profile: str) -> FusionAction:
        """Vision profile: extract evidence, then either answer or hand tools to coder."""
        task_with_evidence = await self._with_vision_evidence(task, profile)
        tools_payload = self._tools_payload(task_with_evidence)
        if tools_payload:
            coder_task = replace(task_with_evidence, profile="fast-agent")
            return await self._run_fast_agent(coder_task)

        evidence_text = "\n\n".join(
            format_vision_evidence(evidence, index)
            for index, evidence in enumerate(task_with_evidence.vision_evidence, start=1)
        )
        return FusionAction(
            type="final_answer",
            text=evidence_text,
            tool_name=None,
            tool_input=None,
            confidence=1.0,
            rationale_summary="vision_evidence_only",
            trace={
                "profile": profile,
                "vision_models": [e.model for e in task_with_evidence.vision_evidence],
                "tool_loop": None,
            },
        )

    async def _run_vision_heavy_agent(self, task: ParsedAgentTask) -> FusionAction:
        """Vision-heavy: extract evidence, unload vision/pro models, then run heavy-agent."""
        vision_profile = self._vision_profile_for_heavy_task(task)
        task_with_evidence = await self._with_vision_evidence(task, vision_profile)
        self._release_ollama_before_ds4()
        heavy_task = replace(task_with_evidence, profile="heavy-agent")
        action = await self._run_heavy_agent(heavy_task)
        if action.trace is None:
            action.trace = {}
        action.trace["vision_profile"] = vision_profile
        action.trace["vision_models"] = [e.model for e in task_with_evidence.vision_evidence]
        return action

    async def _run_fast_agent(self, task: ParsedAgentTask) -> FusionAction:
        """Fast-agent: single call to qwen3-coder with native tool calling."""
        messages = self._build_messages(FAST_AGENT_SYSTEM, task)

        tools_payload = self._tools_payload(task)

        response = self.ollama.chat_completion(
            model=self.config.model_fast,
            messages=messages,
            max_tokens=self._request_max_tokens(task, self.config.fast_max_tokens),
            tools=tools_payload,
            stream=False,
            temperature=self._request_temperature(task),
        )

        return self._action_from_ollama_response(response, tools_payload)

    async def _run_full_agent(self, task: ParsedAgentTask) -> FusionAction:
        """Full-agent: coder → tooler → critic → judge panel."""
        logger.info("Running full-agent panel")

        # Step 1: Coder produces proposal
        coder_messages = self._build_messages(FULL_AGENT_CODER_SYSTEM, task)
        tools_payload = self._tools_payload(task)
        coder_response = self.ollama.chat_completion(
            model=self.config.model_coder,
            messages=coder_messages,
            max_tokens=self._request_max_tokens(task, self.config.full_panel_max_tokens),
            stream=False,
            tools=tools_payload,
            temperature=self._request_temperature(task),
        )
        if tools_payload:
            coder_action = self._action_from_ollama_response(coder_response, tools_payload)
            if coder_action.type == "tool_call":
                return coder_action
            coder_text = coder_action.text or ""
        else:
            coder_text = coder_response.get("choices", [{}])[0].get("message", {}).get("content", "")

        # Step 2: Tooler reviews
        tooler_messages = [
            {"role": "system", "content": FULL_AGENT_TOOLER_SYSTEM},
            {"role": "user", "content": f"審查以下方案：\n\n{coder_text}"},
        ]
        tooler_response = self.ollama.chat_completion(
            model=self.config.model_tooler,
            messages=tooler_messages,
            max_tokens=self._request_max_tokens(task, self.config.full_panel_max_tokens),
            stream=False,
            temperature=self._request_temperature(task),
        )
        tooler_text = tooler_response.get("choices", [{}])[0].get("message", {}).get("content", "")

        # Step 3: Critic reviews
        critic_messages = [
            {"role": "system", "content": FULL_AGENT_CRITIC_SYSTEM},
            {"role": "user", "content": f"審查以下方案與審查意見：\n\n方案：{coder_text}\n\ntooler 意見：{tooler_text}"},
        ]
        critic_response = self.ollama.chat_completion(
            model=self.config.model_critic,
            messages=critic_messages,
            max_tokens=self._request_max_tokens(task, self.config.full_panel_max_tokens),
            stream=False,
            temperature=self._request_temperature(task),
        )
        critic_text = critic_response.get("choices", [{}])[0].get("message", {}).get("content", "")

        # Step 4: Judge synthesizes
        judge_messages = [
            {"role": "system", "content": FULL_AGENT_JUDGE_SYSTEM},
            {
                "role": "user",
                "content": f"方案：{coder_text}\n\ntooler 審查：{tooler_text}\n\ncritic 審查：{critic_text}",
            },
        ]
        judge_response = self.ollama.chat_completion(
            model=self.config.model_judge,
            messages=judge_messages,
            max_tokens=self._request_max_tokens(task, self.config.full_judge_max_tokens),
            stream=False,
            temperature=self._request_temperature(task),
        )
        judge_text = judge_response.get("choices", [{}])[0].get("message", {}).get("content", "")

        # Parse judge output as FusionAction
        return parse_action_from_text(judge_text)

    async def _run_heavy_agent(self, task: ParsedAgentTask) -> FusionAction:
        """Heavy-agent: ds4 primary → checker → critic → judge, with ds4 fallback."""
        logger.info("Running heavy-agent")

        tools_payload = self._tools_payload(task)
        if tools_payload:
            coder_response = self.ollama.chat_completion(
                model=self.config.model_coder,
                messages=self._build_messages(FAST_AGENT_SYSTEM, task),
                max_tokens=self._request_max_tokens(task, self.config.fast_max_tokens),
                tools=tools_payload,
                stream=False,
                temperature=self._request_temperature(task),
            )
            coder_action = self._action_from_ollama_response(coder_response, tools_payload)
            if coder_action.type == "tool_call":
                coder_action.trace = {
                    "profile": "heavy-agent",
                    "tool_loop": "coder",
                    "heavy_backend": None,
                    "fallback": None,
                }
                return coder_action

        # Check ds4 health
        ds4_online = self.ds4.health()

        if ds4_online:
            self._release_ollama_before_ds4()

            # Step 1: Primary (ds4)
            primary_messages = self._build_messages(HEAVY_AGENT_PRIMARY_SYSTEM, task)
            try:
                primary_response = self.ds4.chat_completion(
                    model=self.config.ds4_model,
                    messages=primary_messages,
                    max_tokens=self._request_max_tokens(task, self.config.heavy_max_tokens),
                    stream=False,
                    temperature=self._request_temperature(task),
                )
                primary_text = primary_response.get("choices", [{}])[0].get("message", {}).get("content", "")
            except Exception as e:
                logger.warning(f"ds4 primary failed: {e}")
                return await self._full_agent_fallback(task, "ds4_primary_failed")

            if not isinstance(primary_text, str) or not primary_text.strip():
                logger.warning("ds4 primary returned bad response")
                return await self._full_agent_fallback(task, "ds4_bad_response")

            fit = check_static_fit(
                model_estimates_gb=[
                    self.config.est_model_heavy_gb,
                    self.config.est_model_coder_gb,
                    self.config.est_model_critic_gb,
                    self.config.est_model_judge_gb,
                ],
                parallel_count=1,
                unified_memory_gb=self.config.m5_unified_memory_gb,
                reserved_memory_gb=self.config.m5_reserved_memory_gb,
                kv_cache_reserve_gb_per_parallel=self.config.m5_kv_cache_reserve_gb_per_parallel_model,
            )
            if not fit.ok:
                logger.warning("heavy-agent resource guard: %s", fit.reason)
                return self._primary_action(
                    primary_text,
                    f"heavy_resource_guard: {fit.reason}",
                    {
                        "profile": "heavy-agent",
                        "heavy_backend": "ds4",
                        "fallback": None,
                        "resource_guard": True,
                        "reason": fit.reason,
                    },
                )

            # Step 2: Checker (Ollama qwen3-coder)
            checker_messages = [
                {"role": "system", "content": HEAVY_AGENT_CHECKER_SYSTEM},
                {"role": "user", "content": f"審查以下方案：\n\n{primary_text}"},
            ]
            checker_response = self.ollama.chat_completion(
                model=self.config.model_coder,
                messages=checker_messages,
                max_tokens=self._request_max_tokens(task, self.config.full_panel_max_tokens),
                stream=False,
                temperature=self._request_temperature(task),
            )
            checker_text = checker_response.get("choices", [{}])[0].get("message", {}).get("content", "")

            # Step 3: Critic (deepseek-r1)
            critic_messages = [
                {"role": "system", "content": HEAVY_AGENT_CRITIC_SYSTEM},
                {
                    "role": "user",
                    "content": f"方案：{primary_text}\n\nchecker 審查：{checker_text}",
                },
            ]
            critic_response = self.ollama.chat_completion(
                model=self.config.model_critic,
                messages=critic_messages,
                max_tokens=self._request_max_tokens(task, self.config.full_panel_max_tokens),
                stream=False,
                temperature=self._request_temperature(task),
            )
            critic_text = critic_response.get("choices", [{}])[0].get("message", {}).get("content", "")

            # Step 4: Judge (deepseek-r1)
            judge_messages = [
                {"role": "system", "content": HEAVY_AGENT_JUDGE_SYSTEM},
                {
                    "role": "user",
                    "content": f"方案：{primary_text}\n\nchecker 審查：{checker_text}\n\ncritic 審查：{critic_text}",
                },
            ]
            judge_response = self.ollama.chat_completion(
                model=self.config.model_judge,
                messages=judge_messages,
                max_tokens=self._request_max_tokens(task, self.config.full_judge_max_tokens),
                stream=False,
                temperature=self._request_temperature(task),
            )
            judge_text = judge_response.get("choices", [{}])[0].get("message", {}).get("content", "")

            judge_action = parse_action_from_text(judge_text)
            if judge_action.type == "final_answer" and not (judge_action.text or "").strip():
                return self._primary_action(
                    primary_text,
                    "judge_empty: using ds4 primary answer",
                    {
                        "profile": "heavy-agent",
                        "heavy_backend": "ds4",
                        "fallback": None,
                        "judge_empty": True,
                    },
                )
            judge_action.trace = {
                "profile": "heavy-agent",
                "heavy_backend": "ds4",
                "fallback": None,
            }
            return judge_action
        else:
            logger.info("ds4 offline, falling back to full-agent")
            return await self._full_agent_fallback(task, "ds4_offline")

    async def _full_agent_fallback(self, task: ParsedAgentTask, fallback_reason: str) -> FusionAction:
        """Run full-agent as heavy-agent fallback and annotate the path used."""
        try:
            action = await self._run_full_agent(task)
        except Exception as e:
            logger.warning("heavy-agent fallback failed: %s", e)
            return FusionAction(
                type="final_answer",
                text=f"heavy-agent fallback failed: {e}",
                tool_name=None,
                tool_input=None,
                confidence=0.0,
                rationale_summary="fallback_failed",
                trace={
                    "profile": "heavy-agent",
                    "heavy_backend": None,
                    "fallback": "full-agent",
                    "fallback_reason": fallback_reason,
                    "error": "fallback_failed",
                },
            )
        if action.type == "final_answer" and not (action.text or "").strip():
            return FusionAction(
                type="final_answer",
                text=(
                    "heavy-agent fallback produced empty output after "
                    f"{fallback_reason}; ds4 primary failed and full-agent judge returned no usable content"
                ),
                tool_name=None,
                tool_input=None,
                confidence=0.0,
                rationale_summary="fallback_empty",
                trace={
                    "profile": "heavy-agent",
                    "heavy_backend": None,
                    "fallback": "full-agent",
                    "fallback_reason": fallback_reason,
                    "error": "fallback_empty",
                },
            )
        action.trace = {
            "profile": "heavy-agent",
            "heavy_backend": None,
            "fallback": "full-agent",
            "fallback_reason": fallback_reason,
        }
        return action

    def _primary_action(self, primary_text: str, rationale: str, trace: dict | None = None) -> FusionAction:
        """Convert ds4 primary output into an action and annotate the path used."""
        action = parse_action_from_text(primary_text)
        if action.type == "final_answer":
            action.rationale_summary = rationale
        action.trace = trace
        return action

    async def _run_chat_agent(self, task: ParsedAgentTask) -> FusionAction:
        """Chat-agent: simple Ollama chat for Hermes Desktop."""
        messages = self._build_messages(CHAT_AGENT_SYSTEM, task)

        response = self.ollama.chat_completion(
            model=self.config.model_coder,
            messages=messages,
            max_tokens=self._request_max_tokens(task, self.config.fast_max_tokens),
            stream=False,
            temperature=self._request_temperature(task),
        )

        text = response.get("choices", [{}])[0].get("message", {}).get("content", "")
        return self._annotate_trace(FusionAction(
            type="final_answer",
            text=text,
            tool_name=None,
            tool_input=None,
            confidence=1.0,
            rationale_summary=None,
        ), self.config.model_coder)

    async def _run_agentic_workflow(self, task: ParsedAgentTask) -> FusionAction:
        return self._annotate_agent_runtime_trace(
            await self.agent_orchestrator.run(task, "agentic-workflow")
        )

    async def _run_coding_workflow(self, task: ParsedAgentTask) -> FusionAction:
        return self._annotate_agent_runtime_trace(
            await self.agent_orchestrator.run(task, "coding-workflow")
        )

    async def _run_review_workflow(self, task: ParsedAgentTask) -> FusionAction:
        return self._annotate_agent_runtime_trace(
            await self.agent_orchestrator.run(task, "review-workflow")
        )

    def _annotate_agent_runtime_trace(self, action: FusionAction) -> FusionAction:
        if action.trace is None:
            action.trace = {}
        model_id = (
            action.trace.get("selected_model")
            or action.trace.get("model")
            or getattr(self.agent_orchestrator, "core_last_used_model", None)
        )
        if isinstance(model_id, str) and model_id:
            self.last_used_model = model_id
            action.trace["last_used_model"] = model_id
        return action

    def _annotate_workflow_routing(
        self,
        action: FusionAction,
        routed_from_workflow: str | None,
    ) -> FusionAction:
        if routed_from_workflow is None:
            return action
        if action.trace is None:
            action.trace = {}
        action.trace["routed_from_workflow"] = routed_from_workflow
        return action

    async def _run_agentic_pro_agent(self, task: ParsedAgentTask) -> FusionAction:
        """Agentic-pro: qwen3.6 coding/thinking/tool profile, not the default fast path."""
        task = replace(task, text=f"/no_think\n{task.text}" if task.text else "/no_think")
        messages = self._build_messages(FAST_AGENT_SYSTEM, task)
        tools_payload = self._tools_payload(task)
        response = self.ollama.chat_completion(
            model=self.config.model_agentic_pro,
            # Cap at 800 — qwen3.6 thinking blows larger budgets before
            # writing any content. formatter-mlx uses the same model + 800
            # reliably.
            messages=messages,
            max_tokens=self._request_max_tokens(task, 800),
            tools=tools_payload,
            stream=False,
            # Low temperature suppresses excessive thinking-block generation
            # on qwen3.6; formatter-mlx uses 0.2 reliably.
            temperature=self._request_temperature(task, 0.2),
        )
        action = self._action_from_ollama_response(response, tools_payload)
        if action.trace is None:
            action.trace = {}
        action.trace["profile"] = "agentic-pro"
        action.trace["model"] = self.config.model_agentic_pro
        return self._annotate_trace(action, self.config.model_agentic_pro)

    async def _run_hermes_pro_agent(self, task: ParsedAgentTask) -> FusionAction:
        """Hermes-pro: qwen3.6 general multimodal desktop profile."""
        task = replace(task, text=f"/no_think\n{task.text}" if task.text else "/no_think")
        messages = self._build_messages(CHAT_AGENT_SYSTEM, task)
        tools_payload = self._tools_payload(task)
        response = self.ollama.chat_completion(
            model=self.config.model_hermes_pro,
            # Cap at 800 (matches formatter-mlx's reliable budget).
            messages=messages,
            max_tokens=self._request_max_tokens(task, 800),
            tools=tools_payload,
            stream=False,
            # formatter-mlx uses 0.2; same model, same reliable temp.
            temperature=self._request_temperature(task, 0.2),
        )
        action = self._action_from_ollama_response(response, tools_payload)
        if action.trace is None:
            action.trace = {}
        action.trace["profile"] = "hermes-pro"
        action.trace["model"] = self.config.model_hermes_pro
        return self._annotate_trace(action, self.config.model_hermes_pro)

    async def _run_agentic_mlx_agent(self, task: ParsedAgentTask) -> FusionAction:
        """Optional MLX agentic lane for qwen3.6 NVFP4 without changing defaults.

        Uses a dedicated 256K input ceiling (qwen3.6:35b-a3b-nvfp4 native ctx)
        rather than the vision-pro 96K shared limit.
        """
        task = replace(task, text=f"/no_think\n{task.text}" if task.text else "/no_think")
        messages = self._build_messages(FAST_AGENT_SYSTEM, task)
        tools_payload = self._tools_payload(task)
        response = self.ollama.chat_completion(
            model=self.config.model_agentic_mlx,
            # Cap at 800 — qwen3.6 thinking model, formatter-mlx confirmed this is reliable.
            messages=messages,
            max_tokens=self._request_max_tokens(task, 800),
            tools=tools_payload,
            stream=False,
            # formatter-mlx uses 0.2; same model, same reliable temp.
            temperature=self._request_temperature(task, 0.2),
        )
        action = self._action_from_ollama_response(response, tools_payload)
        if action.trace is None:
            action.trace = {}
        action.trace["profile"] = "agentic-mlx"
        action.trace["model"] = self.config.model_agentic_mlx
        return self._annotate_trace(action, self.config.model_agentic_mlx)

    async def _run_formatter_mlx_agent(self, task: ParsedAgentTask) -> FusionAction:
        """Optional MLX formatter lane for text-only formatting and summarization."""
        task = replace(task, text=f"/no_think\n{task.text}" if task.text else "/no_think")
        messages = self._build_messages(CHAT_AGENT_SYSTEM, task)
        response = self.ollama.chat_completion(
            model=self.config.model_formatter_mlx,
            messages=messages,
            max_tokens=self._request_max_tokens(task, self.config.fast_max_tokens),
            stream=False,
            temperature=self._request_temperature(task, 0.2),
        )
        text = response.get("choices", [{}])[0].get("message", {}).get("content", "")
        action = parse_action_from_text(text)
        if action.trace is None:
            action.trace = {}
        action.trace["profile"] = "formatter-mlx"
        action.trace["model"] = self.config.model_formatter_mlx
        return self._annotate_trace(action, self.config.model_formatter_mlx)

    async def _run_fusion_agent(self, task: ParsedAgentTask) -> FusionAction:
        """G10: OpenRouter-style multi-model deliberation router.

        Extract FusionRequest from raw body → resolve preset → run serial
        panel via fusion_deliberation.run_fusion_agent → wrap into
        FusionAction with structured output and full trace.
        """
        raw = task.raw_request or {}
        fusion_req = extract_fusion_request(raw)
        trace: dict = {
            "profile": "fusion-agent",
            "source_protocol": task.source_protocol,
            "stream": task.stream,
            "request_shape": (
                "plugins" if isinstance(raw.get("plugins"), list)
                and any(
                    isinstance(p, dict) and p.get("id") == "fusion"
                    for p in raw.get("plugins", [])
                )
                else "fusion_block" if isinstance(raw.get("fusion"), dict)
                else "default"
            ),
        }
        try:
            preset = resolve_preset(fusion_req, default=self.config.fusion_default_preset)
        except FusionPresetError as exc:
            trace["error"] = "fusion_preset_error"
            trace["error_detail"] = str(exc)
            return FusionAction(
                type="final_answer",
                text=(
                    f"fusion-agent preset error: {exc}. "
                    f"Known presets: quality / budget / coding / heavy. "
                    f"Or pass `fusion.analysis_models` (list) and optionally "
                    f"`fusion.judge_model`."
                ),
                tool_name=None,
                tool_input=None,
                confidence=0.0,
                rationale_summary="fusion_preset_error",
                trace=trace,
            )

        trace["fusion"] = {
            "preset": preset.name,
            "analysis_models": list(preset.analysis_models),
            "judge_model": preset.judge_model,
            "description": preset.description,
        }

        try:
            result = await run_fusion_agent(
                ollama_client=self.ollama,
                ds4_client=self.ds4,
                preset=preset,
                original_prompt=task.text or "",
                panel_max_tokens=self._request_max_tokens(
                    task, self.config.fusion_max_tokens_panel
                ),
                judge_max_tokens=self._request_max_tokens(
                    task, self.config.fusion_max_tokens_judge
                ),
                ds4_model=self.config.ds4_model,
                temperature=self._request_temperature(task, 0.3),
            )
        except Exception as exc:
            trace["error"] = "fusion_runner_error"
            trace["error_detail"] = f"{type(exc).__name__}: {exc}"
            logger.exception("fusion_agent runner crashed")
            return FusionAction(
                type="final_answer",
                text=(
                    f"fusion-agent runner error: {type(exc).__name__}: {exc}. "
                    f"Preset was {preset.name}. Check LM Studio / ds4 health."
                ),
                tool_name=None,
                tool_input=None,
                confidence=0.0,
                rationale_summary="fusion_runner_error",
                trace=trace,
            )

        # Merge runner trace into our outer trace
        trace["fusion"]["judge_backend"] = result["trace"]["judge_backend"]
        trace["fusion"]["panel_responses"] = result["trace"]["panel_responses"]
        trace["fusion"]["judge_text_preview"] = result["trace"]["judge_text_preview"]
        trace["fusion"]["structured_had_fallback"] = result["trace"]["structured_had_fallback"]
        trace["fusion"]["total_latency_ms"] = result["trace"]["total_latency_ms"]
        trace["fusion"]["structured"] = {
            "final_answer": result["structured"].final_answer,
            "consensus": result["structured"].consensus,
            "contradictions": result["structured"].contradictions,
            "blind_spots": result["structured"].blind_spots,
            "per_model_notes": result["structured"].per_model_notes,
            "had_fallback": result["structured"].had_fallback,
        }

        # Confidence: higher if structured, lower if fallback
        confidence = 0.85 if not result["structured"].had_fallback else 0.5
        rationale = (
            "fusion_deliberation_completed"
            if not result["structured"].had_fallback
            else "fusion_deliberation_fallback_used"
        )

        return self._annotate_trace(FusionAction(
            type="final_answer",
            text=result["text"],
            tool_name=None,
            tool_input=None,
            confidence=confidence,
            rationale_summary=rationale,
            trace=trace,
        ), preset.judge_model)

    def close(self):
        self.ollama.close()
        self.ds4.close()

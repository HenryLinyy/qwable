"""Select concrete models for agent workflow stages."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from qwable.config import FusionConfig
from qwable.model_capabilities import (
    LocalModelSpec,
    assert_model_allowed_for_role,
    build_qwable_spec,
    build_qwythos_spec,
)
from qwable.model_roles import ModelRole, RoleSelection, WorkflowStage
from qwable.workflows import (
    STAGE_ROLE_MAP,
    WORKFLOW_DEFAULT_MAX_TOKENS,
    WORKFLOW_STAGE_ROLE_MAP,
)


def _split_chain(value: str) -> list[str]:
    return [item.strip() for item in (value or "").split(",") if item.strip()]


def _dedupe_preserve_order(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        if item and item not in seen:
            seen.add(item)
            out.append(item)
    return out


# v1.8: richer selection result. Carries the generation config + role
# capabilities so the orchestrator can log/trace exactly which model is
# being used and why.
@dataclass(frozen=True)
class SelectedModel:
    role: ModelRole
    stage: WorkflowStage | None
    model_name: str
    fallback_chain: list[str]
    generation_config: dict[str, Any] = field(default_factory=dict)
    spec: LocalModelSpec | None = None
    reason: str = ""


class ModelSelector:
    def __init__(self, config: FusionConfig):
        self.config = config

    # ── v1.7 API (preserved for backward compat) ─────────────────────

    def model_for_role(self, role: ModelRole) -> str:
        c = self.config
        mapping = {
            ModelRole.SIMPLE_FORMATTER: (
                c.model_role_simple_formatter or c.model_formatter_mlx or c.model_fast
            ),
            ModelRole.PLANNER: (
                c.model_role_planner
                or c.model_agentic_mlx
                or c.model_agentic_pro
                or c.model_coder
            ),
            ModelRole.EXECUTOR: c.model_role_executor
            or c.model_coder
            or c.model_tooler,
            ModelRole.REPAIR: c.model_role_repair or c.model_coder or c.model_tooler,
            ModelRole.LONG_CONTEXT_WORKER: (
                c.model_role_planner or c.model_agentic_mlx or c.model_heavy
            ),
            ModelRole.CRITIC: c.model_role_critic or c.model_critic,
            ModelRole.JUDGE: c.model_role_judge or c.model_judge or c.model_critic,
            ModelRole.HEAVY_PRIMARY: c.model_role_heavy_primary or c.model_heavy,
            ModelRole.VISION: c.model_role_vision or c.model_vision_pro,
        }
        model = mapping.get(role)
        if not model:
            raise RuntimeError(f"No model configured for role: {role}")
        return model

    def fallback_chain_for_role(self, role: ModelRole) -> list[str]:
        c = self.config
        mapping = {
            ModelRole.PLANNER: c.model_role_planner_fallback_chain,
            ModelRole.EXECUTOR: c.model_role_executor_fallback_chain,
            ModelRole.REPAIR: c.model_role_repair_fallback_chain,
            ModelRole.CRITIC: c.model_role_critic_fallback_chain,
            ModelRole.JUDGE: c.model_role_judge_fallback_chain,
        }
        chain = _split_chain(mapping.get(role, ""))
        primary = self.model_for_role(role)
        if primary and primary not in chain:
            chain.insert(0, primary)
        return chain or [primary]

    def select(
        self, workflow: str, stage: str, *, temperature: float = 0.2
    ) -> RoleSelection:
        try:
            role = WORKFLOW_STAGE_ROLE_MAP[workflow][stage]
        except KeyError as exc:
            raise RuntimeError(
                f"No model role mapped for workflow={workflow} stage={stage}"
            ) from exc

        max_tokens = WORKFLOW_DEFAULT_MAX_TOKENS.get(workflow, {}).get(stage, 1200)
        model = self.model_for_role(role)
        return RoleSelection(
            workflow=workflow,
            stage=stage,
            role=role,
            model=model,
            fallback_chain=self.fallback_chain_for_role(role),
            max_tokens=max_tokens,
            temperature=temperature,
            reason=f"workflow={workflow}; stage={stage}; role={role.value}",
        )

    # ── v1.8 API (per plan §7) ───────────────────────────────────────

    def resolve_executor_chain(self) -> list[str]:
        """Per plan §7.3: [Qwable, model_coder, model_agentic_mlx] when enabled."""
        c = self.config
        chain: list[str] = []
        if c.enable_qwable_executor and c.model_qwable:
            chain.append(c.model_qwable)
        if c.model_coder:
            chain.append(c.model_coder)
        if c.model_agentic_mlx:
            chain.append(c.model_agentic_mlx)
        return _dedupe_preserve_order(chain)

    def resolve_repair_chain(self) -> list[str]:
        """Per plan §7.4: same chain shape as executor."""
        return self.resolve_executor_chain()

    def resolve_long_context_chain(self) -> list[str]:
        """Per plan §7.5: Qwythos opt-in, then qwen3.6/agentic_mlx, then heavy."""
        c = self.config
        chain: list[str] = []
        if c.enable_qwythos_long_context and c.model_qwythos:
            chain.append(c.model_qwythos)
        if c.model_agentic_mlx:
            chain.append(c.model_agentic_mlx)
        if c.model_heavy:
            chain.append(c.model_heavy)
        return _dedupe_preserve_order(chain)

    def select_for_stage(
        self,
        stage: WorkflowStage,
        *,
        temperature: float | None = None,
    ) -> SelectedModel:
        """Per plan §7.2: select a model for an explicit stage.

        The role is looked up from STAGE_ROLE_MAP; the model is selected
        from the role-specific chain (Qwable / Qwythos aware).
        """
        try:
            role = STAGE_ROLE_MAP[stage]
        except KeyError as exc:
            raise RuntimeError(f"No role mapped for stage: {stage}") from exc

        # Pick the chain based on role (per plan §7.3/§7.4/§7.5).
        if role == ModelRole.EXECUTOR:
            chain = self.resolve_executor_chain()
        elif role == ModelRole.REPAIR:
            chain = self.resolve_repair_chain()
        elif role == ModelRole.LONG_CONTEXT_WORKER:
            chain = self.resolve_long_context_chain()
        else:
            # Planner / Critic / Judge / Vision use the v1.7 chain
            chain = self.fallback_chain_for_role(role)

        if not chain:
            raise RuntimeError(f"No model in chain for role: {role}")

        primary = chain[0]
        spec = self._spec_for_model(primary, role)
        if spec is not None:
            assert_model_allowed_for_role(spec, role.value)

        gen_cfg = spec.generation_config() if spec else {}
        if temperature is not None:
            gen_cfg = {**gen_cfg, "temperature": temperature}
        elif "temperature" not in gen_cfg:
            # Key-presence check, not truthiness: an explicit temperature=0.0
            # (greedy decoding) must not be clobbered to 0.2.
            gen_cfg = {**gen_cfg, "temperature": 0.2}

        return SelectedModel(
            role=role,
            stage=stage,
            model_name=primary,
            fallback_chain=list(chain[1:]),
            generation_config=gen_cfg,
            spec=spec,
            reason=f"stage={stage.value}; role={role.value}",
        )

    def _spec_for_model(
        self, model_name: str, role: ModelRole
    ) -> LocalModelSpec | None:
        """Return the LocalModelSpec for a model name, or None if not registered."""
        c = self.config
        if model_name == c.model_qwable:
            return build_qwable_spec(c)
        if model_name == c.model_qwythos:
            return build_qwythos_spec(c)
        # Other models (qwen3-coder-next, qwen3.6, etc.) have no v1.8 spec yet.
        return None

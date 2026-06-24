"""v1.8: Model capability metadata and local model specs.

This module is the single source of truth for *what each model can do* and
*how it should be invoked*. It feeds:

- `ModelSelector` (which uses capability sets to forbid role mismatches)
- `/health/models` endpoint (which reports role → primary / fallback / capability)
- Repair loop / context compactor (which read `generation_config` from a spec)
- Test suite (which asserts that Qwable cannot be selected as JUDGE, etc.)

Per the v1.8 plan §6:
- Qwable is allowed for CODING, PATCHING, REPAIR, TOOL_REASONING only.
- Qwythos is allowed for LONG_CONTEXT, TOOL_REASONING, JSON only.
- Neither is allowed for PLANNING, CRITIC, JUDGE, or VISION.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from qwable.config import FusionConfig


class ModelCapability(str, Enum):
    CODING = "coding"
    PATCHING = "patching"
    REPAIR = "repair"
    LONG_CONTEXT = "long_context"
    TOOL_REASONING = "tool_reasoning"
    VISION = "vision"
    CRITIC = "critic"
    JUDGE = "judge"
    PLANNING = "planning"
    JSON = "json"


class ModelRuntime(str, Enum):
    LMSTUDIO = "lmstudio"
    OLLAMA = "ollama"
    LLAMA_CPP = "llama_cpp"
    MLX = "mlx"
    OPENAI_COMPAT = "openai_compat"


@dataclass(frozen=True)
class LocalModelSpec:
    """Static description of a local model's role and invocation shape."""

    name: str
    runtime: ModelRuntime
    capabilities: frozenset[ModelCapability]
    context_limit: int
    temperature: float
    top_p: float | None = None
    top_k: int | None = None
    repeat_penalty: float | None = None
    max_tokens: int = 4096
    may_emit_think_blocks: bool = False
    notes: str = ""

    def has_capability(self, capability: ModelCapability) -> bool:
        return capability in self.capabilities

    def generation_config(self) -> dict:
        cfg: dict = {
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        if self.top_p is not None:
            cfg["top_p"] = self.top_p
        if self.top_k is not None:
            cfg["top_k"] = self.top_k
        if self.repeat_penalty is not None:
            cfg["repeat_penalty"] = self.repeat_penalty
        return cfg


QWABLE_CAPABILITIES: frozenset[ModelCapability] = frozenset(
    {
        ModelCapability.CODING,
        ModelCapability.PATCHING,
        ModelCapability.REPAIR,
        ModelCapability.TOOL_REASONING,
    }
)

QWYTHOS_CAPABILITIES: frozenset[ModelCapability] = frozenset(
    {
        ModelCapability.LONG_CONTEXT,
        ModelCapability.TOOL_REASONING,
        ModelCapability.JSON,
    }
)


def build_qwable_spec(settings: FusionConfig) -> LocalModelSpec:
    """Construct the Qwable spec from current settings.

    The Qwable spec is created lazily from settings because:
    - Settings may be overridden via env at startup
    - The spec is read-only after construction, so it's safe to cache
    """
    return LocalModelSpec(
        name=settings.model_qwable,
        runtime=ModelRuntime(settings.model_qwable_runtime),
        capabilities=QWABLE_CAPABILITIES,
        context_limit=settings.model_qwable_context_limit,
        temperature=settings.model_qwable_temperature,
        top_p=settings.model_qwable_top_p,
        repeat_penalty=settings.model_qwable_repeat_penalty,
        max_tokens=4096,
        may_emit_think_blocks=False,
        notes="Fable-style 9B coding executor / repair worker",
    )


def build_qwythos_spec(settings: FusionConfig) -> LocalModelSpec:
    return LocalModelSpec(
        name=settings.model_qwythos,
        runtime=ModelRuntime(settings.model_qwythos_runtime),
        capabilities=QWYTHOS_CAPABILITIES,
        context_limit=settings.model_qwythos_context_limit,
        temperature=settings.model_qwythos_temperature,
        top_p=settings.model_qwythos_top_p,
        top_k=settings.model_qwythos_top_k,
        repeat_penalty=settings.model_qwythos_repeat_penalty,
        max_tokens=4096,
        may_emit_think_blocks=True,
        notes="Mythos-style 9B long-context reasoning worker",
    )


# ── Capability gate ─────────────────────────────────────────────────────


class RoleCapabilityError(ValueError):
    """Raised when a model lacks a required capability for a role.

    Per the v1.8 plan §1, certain role assignments are forbidden:
    - Qwable as planner / critic / judge / vision
    - Qwythos as critic / judge / planner
    """


_FORBIDDEN_ROLE_CAPABILITIES: dict[str, frozenset[ModelCapability]] = {
    "planner": frozenset({ModelCapability.PLANNING}),
    "critic": frozenset({ModelCapability.CRITIC}),
    "judge": frozenset({ModelCapability.JUDGE}),
    "vision": frozenset({ModelCapability.VISION}),
}


def assert_model_allowed_for_role(
    spec: LocalModelSpec,
    role: str,
) -> None:
    """Reject a spec whose capabilities don't cover the role's required set.

    Empty requirement set (e.g. for "executor", "repair") always passes —
    capability restrictions are only used to block specific unsafe swaps
    (Qwable→JUDGE, Qwythos→PLANNER, etc.).
    """
    required = _FORBIDDEN_ROLE_CAPABILITIES.get(role, frozenset())
    if not required:
        return
    missing = required - spec.capabilities
    if missing:
        raise RoleCapabilityError(
            f"Model {spec.name!r} is not allowed for role {role!r}: "
            f"missing capabilities {sorted(m.value for m in missing)}"
        )

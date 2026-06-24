"""G10: Fusion deliberation presets.

Defines the four built-in presets (quality / budget / coding / heavy) and the
resolver that turns a caller `FusionRequest` into a concrete `FusionPreset`.

Preset contents reference model ids from `~/.lmstudio/hub/models` and the
`MODEL_HEAVY=deepseek-v4-flash` (ds4) env var. See HANDOFF.md for sizes.
"""

from typing import NamedTuple, Optional

from qwable.fusion_schemas import FusionRequest


class FusionPreset(NamedTuple):
    """Concrete deliberation panel definition."""

    name: str
    analysis_models: tuple[str, ...]
    judge_model: str
    description: str


# ─── Built-in presets ───────────────────────────────────────────────────

PRESETS: dict[str, FusionPreset] = {
    "quality": FusionPreset(
        name="quality",
        analysis_models=(
            "qwen/qwen3-coder-next",
            "qwen/qwen3.6-35b-a3b",
            "deepseek-r1-distill-qwen-32b",
        ),
        judge_model="qwen/qwen3.6-35b-a3b",
        description="Deep reasoning — 3 large models, qwen3.6 judge",
    ),
    "budget": FusionPreset(
        name="budget",
        analysis_models=(
            "google/gemma-4-26b-a4b-qat",
            "qwen/qwen3.6-35b-a3b",
        ),
        judge_model="qwen/qwen3.6-35b-a3b",
        description="Light deliberation — 2 models, qwen3.6 judge (reliable structured output)",
    ),
    "coding": FusionPreset(
        name="coding",
        analysis_models=(
            "qwen/qwen3-coder-next",
            "qwen/qwen3.6-35b-a3b",
            "deepseek-r1-distill-qwen-32b",
        ),
        judge_model="qwen/qwen3-coder-next",
        description="Coding focus — coder is judge",
    ),
    "heavy": FusionPreset(
        name="heavy",
        analysis_models=(
            "qwen/qwen3-coder-next",
            "deepseek-r1-distill-qwen-32b",
        ),
        judge_model="deepseek-v4-flash",
        description="Long-context — ds4 deepseek-v4-flash as judge",
    ),
}

DEFAULT_PRESET = "quality"


class FusionPresetError(ValueError):
    """Raised when preset name is unknown or custom panel is malformed."""


def resolve_preset(
    request: FusionRequest,
    default: Optional[str] = None,
) -> FusionPreset:
    """Resolve a FusionRequest to a concrete FusionPreset.

    Resolution rules:
    1. Custom panel: `analysis_models` set → use it; `judge_model` if given,
       else inherit judge from named preset (or `default`) if available.
    2. Named preset: `preset` set → look up in PRESETS.
    3. Default preset from config (`default` arg, falls back to DEFAULT_PRESET).

    Raises FusionPresetError on unknown preset or empty custom panel.
    """
    fallback = default if default is not None else DEFAULT_PRESET

    # ─── Custom panel path ──────────────────────────────────────────────
    if request.analysis_models is not None:
        if not request.analysis_models:
            raise FusionPresetError(
                "custom fusion panel: analysis_models cannot be empty"
            )
        if request.judge_model is not None:
            return FusionPreset(
                name=request.preset or "custom",
                analysis_models=tuple(request.analysis_models),
                judge_model=request.judge_model,
                description=f"custom panel with {len(request.analysis_models)} models",
            )
        # Inherit judge from preset (if any) else default preset
        base_name = request.preset or fallback
        base = PRESETS.get(base_name)
        if base is None:
            raise FusionPresetError(
                f"custom fusion panel requires preset or judge_model; "
                f"preset {base_name!r} unknown"
            )
        return FusionPreset(
            name=request.preset or "custom",
            analysis_models=tuple(request.analysis_models),
            judge_model=base.judge_model,
            description=f"custom panel with {len(request.analysis_models)} models",
        )

    # ─── Named preset path ──────────────────────────────────────────────
    name = request.preset or fallback
    preset = PRESETS.get(name)
    if preset is None:
        raise FusionPresetError(
            f"unknown fusion preset {name!r}; known: {sorted(PRESETS)}"
        )

    # Override judge only
    if request.judge_model is not None:
        return FusionPreset(
            name=preset.name,
            analysis_models=preset.analysis_models,
            judge_model=request.judge_model,
            description=preset.description,
        )
    return preset

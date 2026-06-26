"""G10: Extract FusionRequest from a raw request body.

Two input shapes are supported:
  A) OpenRouter-style plugins:
       { "plugins": [{"id": "fusion", "preset": "quality"}] }
  B) Top-level fusion block (simplified):
       { "fusion": {"preset": "...", "analysis_models": [...], "judge_model": "..."} }

If both shapes are present, plugins wins (matches OpenRouter convention).
"""

from qwable.fusion_schemas import FusionRequest

# Preset can be picked purely by model name (for GUI clients that only choose a
# model), e.g. "qwable-fusion-quality" -> preset "quality". An explicit
# fusion block / plugins preset in the body still wins over the model name.
_PRESET_SUFFIXES = ("budget", "quality", "coding", "heavy")


def _preset_from_model_name(raw_body: dict) -> str | None:
    model = raw_body.get("model")
    if not isinstance(model, str):
        return None
    for preset in _PRESET_SUFFIXES:
        if model.endswith(f"-fusion-{preset}"):
            return preset
    return None


def extract_fusion_request(raw_body: dict) -> FusionRequest:
    """Extract a FusionRequest from a request body dict.

    Never mutates the input. Returns an empty FusionRequest when no fusion
    override is present.
    """
    if not isinstance(raw_body, dict):
        return FusionRequest()

    name_preset = _preset_from_model_name(raw_body)

    # ─── Shape A: OpenRouter plugins style ────────────────────────────────
    plugins = raw_body.get("plugins")
    if isinstance(plugins, list):
        for entry in plugins:
            if isinstance(entry, dict) and entry.get("id") == "fusion":
                return FusionRequest(preset=entry.get("preset") or name_preset)

    # ─── Shape B: top-level fusion block ─────────────────────────────────
    fusion_block = raw_body.get("fusion")
    if isinstance(fusion_block, dict):
        # Validate types at this boundary so a stray string can't fan out into
        # per-character "models", and a non-string preset/judge can't slip through.
        raw_models = fusion_block.get("analysis_models")
        analysis_models = (
            list(raw_models)
            if isinstance(raw_models, list)
            and all(isinstance(m, str) for m in raw_models)
            else None
        )
        raw_preset = fusion_block.get("preset")
        raw_judge = fusion_block.get("judge_model")
        return FusionRequest(
            preset=raw_preset if isinstance(raw_preset, str) else name_preset,
            analysis_models=analysis_models,
            judge_model=raw_judge if isinstance(raw_judge, str) else None,
        )

    # ─── Preset selected purely by model name ─────────────────────────────
    if name_preset:
        return FusionRequest(preset=name_preset)

    # ─── No override ──────────────────────────────────────────────────────
    return FusionRequest()

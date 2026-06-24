"""Tests for fusion preset resolution."""

import pytest

from qwable.fusion_presets import (
    DEFAULT_PRESET,
    PRESETS,
    FusionPresetError,
    resolve_preset,
)
from qwable.fusion_schemas import FusionRequest


def test_all_presets_defined():
    """All four presets must be present."""
    assert set(PRESETS) == {"quality", "budget", "coding", "heavy"}


def test_quality_preset_contents():
    """Quality preset uses 3 large models and qwen3.6 judge."""
    p = PRESETS["quality"]
    assert "qwen/qwen3-coder-next" in p.analysis_models
    assert "qwen/qwen3.6-35b-a3b" in p.analysis_models
    assert "deepseek-r1-distill-qwen-32b" in p.analysis_models
    assert p.judge_model == "qwen/qwen3.6-35b-a3b"
    assert len(p.analysis_models) == 3


def test_budget_preset_is_light():
    """Budget preset uses 2 models with qwen3.6 judge (reliable structured output)."""
    p = PRESETS["budget"]
    assert "google/gemma-4-26b-a4b-qat" in p.analysis_models
    assert len(p.analysis_models) == 2
    assert p.judge_model == "qwen/qwen3.6-35b-a3b"


def test_coding_preset_judge_is_coder():
    """Coding preset uses qwen-coder-next as judge."""
    p = PRESETS["coding"]
    assert p.judge_model == "qwen/qwen3-coder-next"


def test_heavy_preset_uses_ds4_judge():
    """Heavy preset uses ds4 deepseek-v4-flash as judge."""
    p = PRESETS["heavy"]
    assert p.judge_model == "deepseek-v4-flash"


def test_default_preset_is_quality():
    assert DEFAULT_PRESET == "quality"


def test_resolve_preset_by_name():
    req = FusionRequest(preset="coding")
    p = resolve_preset(req)
    assert p.name == "coding"
    assert p.judge_model == "qwen/qwen3-coder-next"


def test_resolve_preset_uses_default_when_no_name():
    """Empty FusionRequest resolves to DEFAULT_PRESET."""
    req = FusionRequest()
    p = resolve_preset(req)
    assert p.name == "quality"


def test_resolve_preset_unknown_raises():
    req = FusionRequest(preset="does-not-exist")
    with pytest.raises(FusionPresetError):
        resolve_preset(req)


def test_resolve_custom_panel_with_judge():
    """Custom panel with both analysis_models and judge_model returns custom preset."""
    req = FusionRequest(
        analysis_models=["m1", "m2", "m3"],
        judge_model="m-judge",
    )
    p = resolve_preset(req)
    assert p.analysis_models == ("m1", "m2", "m3")
    assert p.judge_model == "m-judge"
    assert p.name == "custom"


def test_resolve_custom_panel_without_judge_falls_back():
    """Custom panel without judge_model uses preset's judge."""
    req = FusionRequest(preset="budget", analysis_models=["m1", "m2"])
    p = resolve_preset(req)
    assert p.judge_model == PRESETS["budget"].judge_model


def test_resolve_custom_panel_empty_raises():
    """Empty analysis_models list should raise FusionPresetError."""
    req = FusionRequest(analysis_models=[])
    with pytest.raises(FusionPresetError):
        resolve_preset(req)


def test_resolve_preset_judge_override():
    """Override judge_model on a named preset should replace judge only."""
    req = FusionRequest(preset="quality", judge_model="custom-judge")
    p = resolve_preset(req)
    assert p.analysis_models == PRESETS["quality"].analysis_models
    assert p.judge_model == "custom-judge"

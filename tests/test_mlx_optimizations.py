"""G15: tests for MLX optimizations enablement."""

import json
from pathlib import Path

import pytest

from qwable.mlx_optimizations import (
    RECOMMENDED_CONTEXT_LENGTH,
    SETTINGS_PATH,
    apply_recommended_settings,
    get_current_optimizations,
)


@pytest.fixture
def fake_settings(tmp_path: Path, monkeypatch):
    """Provide a fake settings.json for tests."""
    fake_path = tmp_path / "settings.json"
    fake_path.write_text(json.dumps({
        "language": "zh_Hant",
        "defaultContextLength": {"type": "custom", "value": 8192},
        "configPresetInclusiveness": {"speculativeDecoding": False},
    }))
    monkeypatch.setattr("qwable.mlx_optimizations.SETTINGS_PATH", fake_path)
    return fake_path


def test_apply_recommended_settings_bumps_context_length(fake_settings):
    changes = apply_recommended_settings(context_length=32768, dry_run=False)
    # Verify changes were applied
    with fake_settings.open() as f:
        settings = json.load(f)
    assert settings["defaultContextLength"]["value"] == 32768
    # Changes dict reflects old → new
    assert changes["context_length"]["old"] == 8192
    assert changes["context_length"]["new"] == 32768


def test_apply_recommended_settings_enables_speculative(fake_settings):
    apply_recommended_settings(enable_speculative=True, dry_run=False)
    with fake_settings.open() as f:
        settings = json.load(f)
    assert settings["configPresetInclusiveness"]["speculativeDecoding"] is True


def test_apply_recommended_settings_dry_run_does_not_modify(fake_settings):
    apply_recommended_settings(context_length=65536, dry_run=True)
    # File unchanged
    with fake_settings.open() as f:
        settings = json.load(f)
    assert settings["defaultContextLength"]["value"] == 8192  # original


def test_apply_recommended_settings_idempotent(fake_settings):
    """Running twice produces same final state."""
    apply_recommended_settings(context_length=16384, dry_run=False)
    apply_recommended_settings(context_length=16384, dry_run=False)
    with fake_settings.open() as f:
        settings = json.load(f)
    assert settings["defaultContextLength"]["value"] == 16384


def test_apply_recommended_settings_handles_missing_sections(fake_settings):
    """If configPresetInclusiveness doesn't exist, create it."""
    with fake_settings.open() as f:
        settings = json.load(f)
    del settings["configPresetInclusiveness"]
    with fake_settings.open("w") as f:
        json.dump(settings, f)
    apply_recommended_settings(enable_speculative=True, dry_run=False)
    with fake_settings.open() as f:
        settings = json.load(f)
    assert "configPresetInclusiveness" in settings
    assert settings["configPresetInclusiveness"]["speculativeDecoding"] is True


def test_apply_recommended_settings_handles_old_format(fake_settings):
    """If defaultContextLength is a scalar, replace with new dict format."""
    with fake_settings.open() as f:
        settings = json.load(f)
    settings["defaultContextLength"] = 4096  # old scalar format
    with fake_settings.open("w") as f:
        json.dump(settings, f)
    apply_recommended_settings(context_length=32768, dry_run=False)
    with fake_settings.open() as f:
        settings = json.load(f)
    assert settings["defaultContextLength"]["value"] == 32768
    assert settings["defaultContextLength"]["type"] == "custom"


def test_recommended_constants():
    """Recommended values are sensible for M5 Max 128GB."""
    assert RECOMMENDED_CONTEXT_LENGTH == 32768


def test_get_current_optimizations_returns_dict():
    """Returns a dict with expected keys (smoke test)."""
    info = get_current_optimizations()
    assert isinstance(info, dict)
    assert "context_length" in info
    assert "speculative_decoding" in info

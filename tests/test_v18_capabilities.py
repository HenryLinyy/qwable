"""v1.8 capability-metadata tests — per plan §6 / §16.3."""

import pytest

from qwable.config import FusionConfig
from qwable.model_capabilities import (
    ModelCapability,
    ModelRuntime,
    QWABLE_CAPABILITIES,
    QWYTHOS_CAPABILITIES,
    RoleCapabilityError,
    build_qwable_spec,
    build_qwythos_spec,
    assert_model_allowed_for_role,
)


def test_qwable_has_executor_capabilities():
    assert ModelCapability.CODING in QWABLE_CAPABILITIES
    assert ModelCapability.REPAIR in QWABLE_CAPABILITIES
    assert ModelCapability.PATCHING in QWABLE_CAPABILITIES


def test_qwythos_has_long_context_capability():
    assert ModelCapability.LONG_CONTEXT in QWYTHOS_CAPABILITIES


def test_qwable_not_allowed_for_judge_role():
    cfg = FusionConfig()
    spec = build_qwable_spec(cfg)
    with pytest.raises(RoleCapabilityError):
        assert_model_allowed_for_role(spec, "judge")


def test_qwable_not_allowed_for_critic_role():
    cfg = FusionConfig()
    spec = build_qwable_spec(cfg)
    with pytest.raises(RoleCapabilityError):
        assert_model_allowed_for_role(spec, "critic")


def test_qwable_not_allowed_for_planner_role():
    cfg = FusionConfig()
    spec = build_qwable_spec(cfg)
    with pytest.raises(RoleCapabilityError):
        assert_model_allowed_for_role(spec, "planner")


def test_qwable_allowed_for_executor_role():
    """Empty requirement set for "executor" — must always pass."""
    cfg = FusionConfig()
    spec = build_qwable_spec(cfg)
    assert_model_allowed_for_role(spec, "executor")  # no exception


def test_qwable_allowed_for_repair_role():
    cfg = FusionConfig()
    spec = build_qwable_spec(cfg)
    assert_model_allowed_for_role(spec, "repair")  # no exception


def test_qwythos_not_allowed_for_critic_role():
    cfg = FusionConfig()
    spec = build_qwythos_spec(cfg)
    with pytest.raises(RoleCapabilityError):
        assert_model_allowed_for_role(spec, "critic")


def test_qwythos_not_allowed_for_judge_role():
    cfg = FusionConfig()
    spec = build_qwythos_spec(cfg)
    with pytest.raises(RoleCapabilityError):
        assert_model_allowed_for_role(spec, "judge")


def test_qwythos_not_allowed_for_planner_role():
    cfg = FusionConfig()
    spec = build_qwythos_spec(cfg)
    with pytest.raises(RoleCapabilityError):
        assert_model_allowed_for_role(spec, "planner")


def test_qwythos_allowed_for_long_context_role():
    cfg = FusionConfig()
    spec = build_qwythos_spec(cfg)
    assert_model_allowed_for_role(spec, "long_context_worker")


def test_qwable_spec_uses_settings_values():
    """Spec must reflect the live settings (env overrides)."""
    import os

    os.environ["MODEL_QWABLE"] = "custom-qwable-7b"
    os.environ["MODEL_QWABLE_CONTEXT_LIMIT"] = "16384"
    try:
        cfg = FusionConfig()
        spec = build_qwable_spec(cfg)
        assert spec.name == "custom-qwable-7b"
        assert spec.context_limit == 16384
    finally:
        del os.environ["MODEL_QWABLE"]
        del os.environ["MODEL_QWABLE_CONTEXT_LIMIT"]


def test_qwable_generation_config_matches_plan():
    """Per plan §13.1: temperature=0.25, top_p=0.9, repeat_penalty=1.05."""
    cfg = FusionConfig()
    spec = build_qwable_spec(cfg)
    gen = spec.generation_config()
    assert gen["temperature"] == 0.25
    assert gen["top_p"] == 0.9
    assert gen["repeat_penalty"] == 1.05
    assert "top_k" not in gen  # not set on Qwable


def test_qwythos_generation_config_matches_plan():
    """Per plan §13.2: temperature=0.6, top_p=0.95, top_k=20, repeat_penalty=1.05."""
    cfg = FusionConfig()
    spec = build_qwythos_spec(cfg)
    gen = spec.generation_config()
    assert gen["temperature"] == 0.6
    assert gen["top_p"] == 0.95
    assert gen["top_k"] == 20
    assert gen["repeat_penalty"] == 1.05


def test_qwythos_spec_marks_think_blocks():
    """Qwythos is a reasoning model and may emit <think> blocks."""
    cfg = FusionConfig()
    spec = build_qwythos_spec(cfg)
    assert spec.may_emit_think_blocks is True
    # Qwable is a non-reasoning coding model — no think blocks expected.
    qwable = build_qwable_spec(cfg)
    assert qwable.may_emit_think_blocks is False


def test_qwable_runtime_default_lmstudio():
    cfg = FusionConfig()
    spec = build_qwable_spec(cfg)
    assert spec.runtime == ModelRuntime.LMSTUDIO

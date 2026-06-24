"""v1.8 fallback behavior tests — per plan §10 / §16.4.

These tests cover the *plumbing* of the v1.8 fallback chain. Full
end-to-end tests with mocked unavailable runtimes are in the wider
integration test harness (test_v18_orchestrator_integration.py).
"""

import pytest

from qwable.config import FusionConfig
from qwable.model_capabilities import RoleCapabilityError
from qwable.model_roles import ModelRole, WorkflowStage
from qwable.model_selector import ModelSelector


# ── §10.2 Executor fallback chain shape ──────────────────────────────


def test_executor_chain_qwable_first_when_enabled():
    cfg = FusionConfig()
    cfg.enable_qwable_executor = True
    chain = ModelSelector(cfg).resolve_executor_chain()
    assert chain[0] == cfg.model_qwable


def test_executor_chain_qwen_coder_is_first_when_qwable_disabled():
    cfg = FusionConfig()
    cfg.enable_qwable_executor = False
    chain = ModelSelector(cfg).resolve_executor_chain()
    assert chain[0] == cfg.model_coder


def test_repair_chain_uses_same_shape_as_executor():
    """Per plan §7.4: repair chain is identical to executor chain."""
    cfg = FusionConfig()
    sel = ModelSelector(cfg)
    assert sel.resolve_repair_chain() == sel.resolve_executor_chain()


def test_qwable_disabled_keeps_coder_fallback_intact():
    """Per plan §18 rollback: ENABLE_QWABLE_EXECUTOR=false → coder primary."""
    cfg = FusionConfig()
    cfg.enable_qwable_executor = False
    sel = ModelSelector(cfg)
    selected = sel.select_for_stage(WorkflowStage.EXECUTE_PATCH)
    assert selected.model_name == cfg.model_coder
    # Fallback chain still contains agentic_mlx (qwen3.6).
    assert cfg.model_agentic_mlx in selected.fallback_chain


def test_qwythos_disabled_keeps_qwen3_6_in_long_context():
    """Per plan §18 rollback: ENABLE_QWYTHOS_LONG_CONTEXT=false → qwen3.6 primary."""
    cfg = FusionConfig()
    cfg.enable_qwythos_long_context = False
    sel = ModelSelector(cfg)
    selected = sel.select_for_stage(WorkflowStage.CONTEXT_COMPACTION)
    # Primary should be qwen3.6 (model_agentic_mlx), NOT Qwythos.
    assert selected.model_name == cfg.model_agentic_mlx
    assert cfg.model_qwythos not in selected.fallback_chain


# ── §10.4 Long context fallback chain ────────────────────────────────


def test_long_context_chain_omits_qwythos_when_disabled():
    cfg = FusionConfig()
    cfg.enable_qwythos_long_context = False
    chain = ModelSelector(cfg).resolve_long_context_chain()
    assert cfg.model_qwythos not in chain


def test_long_context_chain_qwythos_first_when_enabled():
    cfg = FusionConfig()
    cfg.enable_qwythos_long_context = True
    chain = ModelSelector(cfg).resolve_long_context_chain()
    assert chain[0] == cfg.model_qwythos


# ── §16.4 Misconfig robustness ──────────────────────────────────────


def test_chain_dedupes_when_qwable_equals_coder():
    """If someone sets MODEL_QWABLE=qwen3-coder-next, chain dedupes."""
    cfg = FusionConfig()
    cfg.model_qwable = cfg.model_coder
    chain = ModelSelector(cfg).resolve_executor_chain()
    assert chain.count(cfg.model_coder) == 1


def test_chain_dedupes_when_qwythos_equals_heavy():
    cfg = FusionConfig()
    cfg.model_qwythos = cfg.model_heavy
    cfg.enable_qwythos_long_context = True
    chain = ModelSelector(cfg).resolve_long_context_chain()
    assert chain.count(cfg.model_heavy) == 1


def test_capability_gate_still_blocks_misconfig():
    """Even with a misconfigured MODEL_QWABLE, the gate blocks Qwable for JUDGE."""
    from qwable.model_capabilities import (
        build_qwable_spec,
        assert_model_allowed_for_role,
    )
    cfg = FusionConfig()
    cfg.model_role_judge = cfg.model_qwable  # misconfig: Qwable as judge
    spec = build_qwable_spec(cfg)
    with pytest.raises(RoleCapabilityError):
        assert_model_allowed_for_role(spec, "judge")


def test_executor_repair_use_distinct_stage_traces():
    """Per plan §7.4: stage=execute_patch and stage=repair_patch must be distinguishable
    in trace output (different stage, same model)."""
    cfg = FusionConfig()
    sel = ModelSelector(cfg)
    exec_sel = sel.select_for_stage(WorkflowStage.EXECUTE_PATCH)
    rep_sel = sel.select_for_stage(WorkflowStage.REPAIR_PATCH)
    # Same model (Qwable), but different stages.
    assert exec_sel.model_name == rep_sel.model_name == cfg.model_qwable
    assert exec_sel.stage == WorkflowStage.EXECUTE_PATCH
    assert rep_sel.stage == WorkflowStage.REPAIR_PATCH
    # And the role differs.
    assert exec_sel.role == ModelRole.EXECUTOR
    assert rep_sel.role == ModelRole.REPAIR

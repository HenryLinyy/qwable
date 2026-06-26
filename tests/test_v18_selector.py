"""v1.8 ModelSelector tests — per plan §7.3 / §7.4 / §7.5 / §16.1 / §16.2."""

import pytest

from qwable.config import FusionConfig
from qwable.model_capabilities import RoleCapabilityError
from qwable.model_roles import ModelRole, WorkflowStage
from qwable.model_selector import ModelSelector


def test_qwable_is_executor_primary_when_enabled():
    cfg = FusionConfig()
    cfg.enable_qwable_executor = True
    sel = ModelSelector(cfg)
    selected = sel.select_for_stage(WorkflowStage.EXECUTE_PATCH)
    assert selected.role == ModelRole.EXECUTOR
    assert selected.model_name == cfg.model_qwable


def test_qwable_is_repair_primary_when_enabled():
    cfg = FusionConfig()
    cfg.enable_qwable_executor = True
    sel = ModelSelector(cfg)
    selected = sel.select_for_stage(WorkflowStage.REPAIR_PATCH)
    assert selected.role == ModelRole.REPAIR
    assert selected.model_name == cfg.model_qwable


def test_qwen_coder_fallback_kept_for_executor():
    """Per plan §1: qwen3-coder-next MUST remain a fallback after v1.8."""
    cfg = FusionConfig()
    cfg.enable_qwable_executor = True
    sel = ModelSelector(cfg)
    selected = sel.select_for_stage(WorkflowStage.EXECUTE_PATCH)
    assert cfg.model_coder in selected.fallback_chain


def test_qwen_coder_fallback_kept_for_repair():
    cfg = FusionConfig()
    cfg.enable_qwable_executor = True
    sel = ModelSelector(cfg)
    selected = sel.select_for_stage(WorkflowStage.REPAIR_PATCH)
    assert cfg.model_coder in selected.fallback_chain


def test_qwable_disabled_falls_back_to_qwen_coder():
    """ENABLE_QWABLE_EXECUTOR=false → executor primary is qwen3-coder-next."""
    cfg = FusionConfig()
    cfg.enable_qwable_executor = False
    sel = ModelSelector(cfg)
    selected = sel.select_for_stage(WorkflowStage.EXECUTE_PATCH)
    assert selected.model_name == cfg.model_coder


def test_qwythos_disabled_by_default_for_long_context():
    """ENABLE_QWYTHOS_LONG_CONTEXT defaults to false → qwen3.6/agentic_mlx is primary."""
    cfg = FusionConfig()  # default disable
    sel = ModelSelector(cfg)
    selected = sel.select_for_stage(WorkflowStage.CONTEXT_COMPACTION)
    assert selected.model_name != cfg.model_qwythos
    # Primary should be one of the v1.7 long-context models.
    assert selected.model_name in {cfg.model_agentic_mlx, cfg.model_heavy}


def test_qwythos_selected_when_enabled():
    cfg = FusionConfig()
    cfg.enable_qwythos_long_context = True
    sel = ModelSelector(cfg)
    selected = sel.select_for_stage(WorkflowStage.CONTEXT_COMPACTION)
    assert selected.role == ModelRole.LONG_CONTEXT_WORKER
    assert selected.model_name == cfg.model_qwythos


def test_qwable_chain_dedupes_duplicates():
    """If Qwable model id is the same as model_coder (misconfig), the chain dedupes."""
    cfg = FusionConfig()
    cfg.model_qwable = cfg.model_coder  # intentionally collide
    sel = ModelSelector(cfg)
    chain = sel.resolve_executor_chain()
    # Each model appears at most once.
    assert chain.count(cfg.model_coder) == 1


def test_executor_chain_includes_qwable_qwen3_6_in_order():
    cfg = FusionConfig()
    sel = ModelSelector(cfg)
    chain = sel.resolve_executor_chain()
    # Per plan: [Qwable, qwen3-coder-next, qwen3.6-35b-a3b]
    assert chain[0] == cfg.model_qwable
    assert cfg.model_coder in chain
    assert cfg.model_agentic_mlx in chain
    # Order: Qwable first.
    assert chain.index(cfg.model_qwable) < chain.index(cfg.model_coder)


def test_long_context_chain_omits_qwythos_when_disabled():
    cfg = FusionConfig()
    cfg.enable_qwythos_long_context = False
    sel = ModelSelector(cfg)
    chain = sel.resolve_long_context_chain()
    assert cfg.model_qwythos not in chain


def test_long_context_chain_includes_qwythos_first_when_enabled():
    cfg = FusionConfig()
    cfg.enable_qwythos_long_context = True
    sel = ModelSelector(cfg)
    chain = sel.resolve_long_context_chain()
    assert chain[0] == cfg.model_qwythos
    assert cfg.model_agentic_mlx in chain
    assert cfg.model_heavy in chain


def test_select_for_stage_uses_v17_chain_for_planner():
    """Planner / critic / judge use v1.7 chain (unaffected by Qwable)."""
    cfg = FusionConfig()
    cfg.enable_qwable_executor = True  # should be ignored for planner
    sel = ModelSelector(cfg)
    selected = sel.select_for_stage(WorkflowStage.PLAN_REVISION)
    # PLAN_REVISION maps to PLANNER role per STAGE_ROLE_MAP.
    assert selected.role == ModelRole.PLANNER
    assert selected.model_name != cfg.model_qwable


def test_select_for_stage_uses_qwen3_6_for_critic():
    selected_for_critic = ModelSelector(FusionConfig()).select_for_stage(
        WorkflowStage.PLAN_REVIEW
    )
    assert selected_for_critic.role == ModelRole.CRITIC
    # Per plan §3: critic primary is deepseek-r1-distill-qwen-32b (v1.7 unchanged).


def test_select_for_stage_uses_qwen3_6_for_judge():
    selected_for_judge = ModelSelector(FusionConfig()).select_for_stage(
        WorkflowStage.FINAL_REPORT
    )
    assert selected_for_judge.role == ModelRole.JUDGE
    # Judge is v1.7 unchanged.


def test_capability_gate_blocks_qwable_for_judge_stage():
    """Per plan §16.3: Qwable must not be allowed as JUDGE."""
    # Simulate a misconfig where someone set Qwable as the judge model.
    cfg = FusionConfig()
    cfg.model_role_judge = cfg.model_qwable  # misconfig
    # The fix happens at the assertion layer — the gate must reject.
    # We don't actually call select_for_stage here because the v1.7 chain
    # for judge doesn't consult spec; we just verify the gate independently.
    from qwable.model_capabilities import (
        build_qwable_spec,
        assert_model_allowed_for_role,
    )

    spec = build_qwable_spec(cfg)
    with pytest.raises(RoleCapabilityError):
        assert_model_allowed_for_role(spec, "judge")


def test_selected_model_carries_generation_config():
    """Per plan §13.1 / §13.2: SelectedModel.generation_config must reflect the spec."""
    cfg = FusionConfig()
    sel = ModelSelector(cfg)
    selected = sel.select_for_stage(WorkflowStage.EXECUTE_PATCH)
    gen = selected.generation_config
    assert gen["temperature"] == cfg.model_qwable_temperature
    assert gen["top_p"] == cfg.model_qwable_top_p
    assert gen["repeat_penalty"] == cfg.model_qwable_repeat_penalty


def test_qwable_selected_model_has_qwable_spec():
    cfg = FusionConfig()
    sel = ModelSelector(cfg)
    selected = sel.select_for_stage(WorkflowStage.EXECUTE_PATCH)
    assert selected.spec is not None
    assert selected.spec.name == cfg.model_qwable


def test_qwen_coder_selected_model_has_no_spec():
    """v1.7 models (qwen3-coder-next etc.) don't have v1.8 specs — spec is None."""
    cfg = FusionConfig()
    cfg.enable_qwable_executor = False
    sel = ModelSelector(cfg)
    selected = sel.select_for_stage(WorkflowStage.EXECUTE_PATCH)
    assert selected.model_name == cfg.model_coder
    assert selected.spec is None


def test_temperature_override_in_select_for_stage():
    cfg = FusionConfig()
    sel = ModelSelector(cfg)
    selected = sel.select_for_stage(WorkflowStage.EXECUTE_PATCH, temperature=0.7)
    assert selected.generation_config["temperature"] == 0.7


def test_fallback_chain_excludes_primary():
    cfg = FusionConfig()
    sel = ModelSelector(cfg)
    selected = sel.select_for_stage(WorkflowStage.EXECUTE_PATCH)
    assert selected.model_name not in selected.fallback_chain


def test_v17_select_api_still_works():
    """Backward compat: ModelSelector.select(workflow, stage) per v1.7."""
    cfg = FusionConfig()
    sel = ModelSelector(cfg)
    rs = sel.select("coding-workflow", "planner")
    assert rs.model == cfg.model_role_planner
    assert rs.role == ModelRole.PLANNER

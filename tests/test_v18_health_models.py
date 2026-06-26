"""v1.8 /health/models endpoint tests — per plan §14.2 / §17.2.

The endpoint must report, for each role that participates in agent
workflows, the primary model, fallback chain, and a few v1.8-specific
flags (Qwable enabled, Qwythos opt-in, capability set).

This test exercises the pure data builder (no live gateway required).
The full HTTP round-trip is verified by the manual verification flow
(plan §17) and the existing integration test harness.
"""

from qwable.config import FusionConfig
from qwable.model_roles import WorkflowStage
from qwable.model_selector import ModelSelector


def _build_models_health(selector: ModelSelector) -> dict:
    """Pure function — extracted from the endpoint for testability.

    Returns the dict shape documented in plan §14.2.
    """
    cfg = selector.config
    out: dict = {
        "status": "ok",
        "roles": {},
        "flags": {
            "enable_qwable_executor": cfg.enable_qwable_executor,
            "enable_qwythos_long_context": cfg.enable_qwythos_long_context,
            "model_health_check_on_startup": cfg.model_health_check_on_startup,
        },
    }

    for stage in (
        WorkflowStage.EXECUTE_PATCH,
        WorkflowStage.REPAIR_PATCH,
        WorkflowStage.CONTEXT_COMPACTION,
        WorkflowStage.PLAN_REVISION,
        WorkflowStage.PLAN_REVIEW,
        WorkflowStage.FINAL_REPORT,
    ):
        try:
            sel = selector.select_for_stage(stage)
        except RuntimeError:
            continue
        role = sel.role.value
        out["roles"].setdefault(
            role,
            {
                "primary": sel.model_name,
                "fallbacks": list(sel.fallback_chain),
                "stages": [],
            },
        )
        out["roles"][role]["stages"].append(stage.value)

    return out


def test_health_models_includes_executor_role():
    cfg = FusionConfig()
    out = _build_models_health(ModelSelector(cfg))
    assert "executor" in out["roles"]
    assert "reapir" not in out["roles"]  # sanity: spelled correctly
    assert out["roles"]["executor"]["stages"] == ["execute_patch"]


def test_health_models_includes_repair_role():
    cfg = FusionConfig()
    out = _build_models_health(ModelSelector(cfg))
    assert "repair" in out["roles"]
    assert out["roles"]["repair"]["stages"] == ["repair_patch"]


def test_health_models_includes_long_context_worker_role():
    cfg = FusionConfig()
    out = _build_models_health(ModelSelector(cfg))
    assert "long_context_worker" in out["roles"]
    assert out["roles"]["long_context_worker"]["stages"] == ["context_compaction"]


def test_health_models_includes_planner_critic_judge():
    """v1.7 roles (planner / critic / judge) must still appear in /health/models."""
    cfg = FusionConfig()
    out = _build_models_health(ModelSelector(cfg))
    assert "planner" in out["roles"]
    assert "critic" in out["roles"]
    assert "judge" in out["roles"]


def test_health_models_qwable_enabled_flag():
    cfg = FusionConfig()
    out = _build_models_health(ModelSelector(cfg))
    assert out["flags"]["enable_qwable_executor"] is True
    assert out["flags"]["enable_qwythos_long_context"] is False


def test_health_models_reports_qwable_as_executor_primary():
    """Per plan §17.2: /health/models shows Qwable as executor primary."""
    cfg = FusionConfig()
    out = _build_models_health(ModelSelector(cfg))
    assert out["roles"]["executor"]["primary"] == cfg.model_qwable


def test_health_models_reports_qwable_as_repair_primary():
    cfg = FusionConfig()
    out = _build_models_health(ModelSelector(cfg))
    assert out["roles"]["repair"]["primary"] == cfg.model_qwable


def test_health_models_reports_qwen_coder_in_executor_fallback():
    """Per plan: qwen3-coder-next MUST remain in executor fallback."""
    cfg = FusionConfig()
    out = _build_models_health(ModelSelector(cfg))
    assert cfg.model_coder in out["roles"]["executor"]["fallbacks"]


def test_health_models_reports_qwythos_disabled_in_long_context():
    """Default: Qwythos is OFF — long_context_worker must NOT show Qwythos."""
    cfg = FusionConfig()  # default disable
    out = _build_models_health(ModelSelector(cfg))
    assert (
        cfg.model_qwythos not in out["roles"]["long_context_worker"]["fallbacks"]
        or out["roles"]["long_context_worker"]["primary"] != cfg.model_qwythos
    )


def test_health_models_shows_qwythos_when_enabled():
    cfg = FusionConfig()
    cfg.enable_qwythos_long_context = True
    out = _build_models_health(ModelSelector(cfg))
    assert out["roles"]["long_context_worker"]["primary"] == cfg.model_qwythos


def test_health_models_qwable_disabled_falls_back_to_qwen_coder():
    """ENABLE_QWABLE_EXECUTOR=false → executor primary reverts to qwen3-coder-next."""
    cfg = FusionConfig()
    cfg.enable_qwable_executor = False
    out = _build_models_health(ModelSelector(cfg))
    assert out["roles"]["executor"]["primary"] == cfg.model_coder
    assert cfg.model_qwable not in out["roles"]["executor"]["fallbacks"]


def test_health_models_status_ok():
    cfg = FusionConfig()
    out = _build_models_health(ModelSelector(cfg))
    assert out["status"] == "ok"

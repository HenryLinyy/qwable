"""v1.8 orchestrator integration tests — per plan §9 / §10 / §16.4 / §16.5.

These tests exercise the v1.8 stage-level selection at the orchestrator
boundary. They do NOT change the existing v1.7 flow (executor / repair
via WORKFLOW_STAGE_ROLE_MAP); they only add the v1.8 stage path.
"""

from qwable.config import FusionConfig
from qwable.model_roles import ModelRole, WorkflowStage
from qwable.model_selector import ModelSelector


# ── v1.8 stage selection at the orchestrator boundary ───────────────────


def test_orchestrator_can_select_qwable_for_execute_patch():
    """select_for_stage on EXECUTE_PATCH must return Qwable as primary."""
    cfg = FusionConfig()
    cfg.enable_qwable_executor = True
    sel = ModelSelector(cfg)
    selected = sel.select_for_stage(WorkflowStage.EXECUTE_PATCH)
    assert selected.model_name == cfg.model_qwable
    assert selected.role == ModelRole.EXECUTOR


def test_orchestrator_can_select_qwable_for_repair_patch():
    cfg = FusionConfig()
    cfg.enable_qwable_executor = True
    sel = ModelSelector(cfg)
    selected = sel.select_for_stage(WorkflowStage.REPAIR_PATCH)
    assert selected.model_name == cfg.model_qwable
    assert selected.role == ModelRole.REPAIR


def test_orchestrator_qwable_falls_back_to_qwen_coder_on_misconfig():
    """If user disables Qwable, primary reverts to qwen3-coder-next (no code change)."""
    cfg = FusionConfig()
    cfg.enable_qwable_executor = False
    sel = ModelSelector(cfg)
    selected = sel.select_for_stage(WorkflowStage.EXECUTE_PATCH)
    assert selected.model_name == cfg.model_coder


async def test_v18_code_agent_trace_records_qwable_executor(tmp_path):
    """v1.8 stage plumbing: select_for_stage on EXECUTE_PATCH picks Qwable
    when ENABLE_QWABLE_EXECUTOR=true. v1.7 select() still picks the v1.7
    executor primary (backward compat).
    """
    from qwable.config import FusionConfig

    cfg = FusionConfig(
        enable_qwable_executor=True,
        agent_store_path=str(tmp_path / "test.sqlite3"),
    )
    sel = ModelSelector(cfg)

    # v1.8 stage selection — Qwable as primary.
    v18 = sel.select_for_stage(WorkflowStage.EXECUTE_PATCH)
    assert v18.model_name == cfg.model_qwable
    assert v18.role == ModelRole.EXECUTOR

    # Sanity: v1.7 stage still resolves via v1.7 chain (backward compat).
    v17 = sel.select("coding-workflow", "executor")
    assert v17.role == ModelRole.EXECUTOR
    assert v17.model == cfg.model_role_executor


async def test_v18_repair_stage_selects_qwable(tmp_path):
    from qwable.config import FusionConfig

    cfg = FusionConfig(
        enable_qwable_executor=True,
        agent_store_path=str(tmp_path / "test.sqlite3"),
    )
    sel = ModelSelector(cfg)
    v18 = sel.select_for_stage(WorkflowStage.REPAIR_PATCH)
    assert v18.model_name == cfg.model_qwable
    assert v18.role == ModelRole.REPAIR


async def test_v18_long_context_compaction_picks_qwythos_when_enabled(tmp_path):
    from qwable.config import FusionConfig

    cfg = FusionConfig(
        enable_qwythos_long_context=True,
        agent_store_path=str(tmp_path / "test.sqlite3"),
    )
    sel = ModelSelector(cfg)
    v18 = sel.select_for_stage(WorkflowStage.CONTEXT_COMPACTION)
    assert v18.model_name == cfg.model_qwythos
    assert v18.role == ModelRole.LONG_CONTEXT_WORKER


async def test_long_context_worker_selection_reflects_qwythos_flag(tmp_path):
    cfg = FusionConfig(agent_store_path=str(tmp_path / "test.sqlite3"))
    sel = ModelSelector(cfg)
    # Disabled by default.
    assert (
        sel.select_for_stage(WorkflowStage.CONTEXT_COMPACTION).model_name
        != cfg.model_qwythos
    )
    # Enable → Qwythos primary.
    cfg.enable_qwythos_long_context = True
    sel2 = ModelSelector(cfg)
    assert (
        sel2.select_for_stage(WorkflowStage.CONTEXT_COMPACTION).model_name
        == cfg.model_qwythos
    )

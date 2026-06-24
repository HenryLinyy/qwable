"""Tests for selecting models by workflow role."""

import pytest

from qwable.config import FusionConfig


def test_split_chain_trims_and_drops_empty_entries():
    from qwable.model_selector import _split_chain

    assert _split_chain(" a, ,b ,, c ") == ["a", "b", "c"]


def test_selector_resolves_coding_executor():
    from qwable.model_roles import ModelRole
    from qwable.model_selector import ModelSelector

    selector = ModelSelector(FusionConfig())

    selection = selector.select("coding-workflow", "executor")

    assert selection.workflow == "coding-workflow"
    assert selection.stage == "executor"
    assert selection.role == ModelRole.EXECUTOR
    assert selection.model == "qwen/qwen3-coder-next"
    assert selection.fallback_chain == [
        "qwen/qwen3-coder-next",
        "qwen/qwen3.6-35b-a3b",
    ]
    assert selection.max_tokens == 1800
    assert selection.temperature == 0.2
    assert selection.reason == "workflow=coding-workflow; stage=executor; role=executor"


def test_selector_inserts_primary_model_when_missing_from_fallback_chain():
    from qwable.model_roles import ModelRole
    from qwable.model_selector import ModelSelector

    cfg = FusionConfig(
        model_role_executor="primary/executor",
        model_role_executor_fallback_chain="fallback/a,fallback/b",
    )
    selector = ModelSelector(cfg)

    assert selector.fallback_chain_for_role(ModelRole.EXECUTOR) == [
        "primary/executor",
        "fallback/a",
        "fallback/b",
    ]


def test_selector_uses_legacy_defaults_when_role_config_is_empty():
    from qwable.model_roles import ModelRole
    from qwable.model_selector import ModelSelector

    cfg = FusionConfig(
        model_role_planner="",
        model_agentic_mlx="",
        model_agentic_pro="legacy/planner",
    )
    selector = ModelSelector(cfg)

    assert selector.model_for_role(ModelRole.PLANNER) == "legacy/planner"


def test_selector_rejects_unmapped_workflow_stage():
    from qwable.model_selector import ModelSelector

    selector = ModelSelector(FusionConfig())

    with pytest.raises(RuntimeError, match="No model role mapped"):
        selector.select("review-workflow", "executor")

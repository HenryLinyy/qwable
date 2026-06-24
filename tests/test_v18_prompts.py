"""v1.8 prompt helpers + strip_think tests — per plan §12 / §13.3."""

import pytest

from qwable.config import FusionConfig
from qwable.model_capabilities import build_qwable_spec, build_qwythos_spec
from qwable.v18_prompts import (
    QWABLE_EXECUTOR_SYSTEM,
    QWABLE_REPAIR_SYSTEM,
    QWYTHOS_CONTEXT_WORKER_SYSTEM,
    maybe_strip_think,
    should_strip_think,
    spec_for_stage,
    strip_think_blocks,
    system_prompt_for_stage,
)


# ── Prompt content (per plan §12) ──────────────────────────────────────


def test_qwable_executor_system_prompt_includes_patch_protocol_rule():
    assert "patch protocol" in QWABLE_EXECUTOR_SYSTEM
    assert "do not modify unrelated files" in QWABLE_EXECUTOR_SYSTEM
    # Explicit list of allowed output shapes.
    assert "unified_diff" in QWABLE_EXECUTOR_SYSTEM
    assert "structured_patch_json" in QWABLE_EXECUTOR_SYSTEM
    assert "tool_request_json" in QWABLE_EXECUTOR_SYSTEM
    assert "blocked_report_json" in QWABLE_EXECUTOR_SYSTEM


def test_qwable_repair_system_prompt_includes_minimal_surface_rule():
    assert "smallest possible surface" in QWABLE_REPAIR_SYSTEM
    assert "do not refactor unrelated code" in QWABLE_REPAIR_SYSTEM


def test_qwythos_context_worker_system_prompt_returns_json():
    assert '"facts"' in QWYTHOS_CONTEXT_WORKER_SYSTEM
    assert '"relevant_files"' in QWYTHOS_CONTEXT_WORKER_SYSTEM
    assert '"compressed_context"' in QWYTHOS_CONTEXT_WORKER_SYSTEM
    # Forbidden actions.
    assert "do not produce patches" in QWYTHOS_CONTEXT_WORKER_SYSTEM.lower()
    assert "do not judge" in QWYTHOS_CONTEXT_WORKER_SYSTEM.lower()


def test_system_prompt_for_stage_routing():
    cfg = FusionConfig()
    assert system_prompt_for_stage("execute_patch", cfg) == QWABLE_EXECUTOR_SYSTEM
    assert system_prompt_for_stage("repair_patch", cfg) == QWABLE_REPAIR_SYSTEM
    assert system_prompt_for_stage("context_compaction", cfg) == QWYTHOS_CONTEXT_WORKER_SYSTEM


def test_system_prompt_for_stage_returns_none_for_v17_stages():
    """v1.7 stages (planner / critic / judge) have no v1.8 prompt."""
    cfg = FusionConfig()
    assert system_prompt_for_stage("planner", cfg) is None
    assert system_prompt_for_stage("plan_critic", cfg) is None
    assert system_prompt_for_stage("judge", cfg) is None
    assert system_prompt_for_stage("finalizer", cfg) is None


def test_spec_for_stage_returns_qwable_for_executor_and_repair():
    cfg = FusionConfig()
    spec_exec = spec_for_stage("execute_patch", cfg)
    spec_repair = spec_for_stage("repair_patch", cfg)
    assert spec_exec is not None and spec_exec.name == cfg.model_qwable
    assert spec_repair is not None and spec_repair.name == cfg.model_qwable


def test_spec_for_stage_returns_qwythos_for_long_context_stages():
    cfg = FusionConfig()
    for stage in ("context_acquisition", "repo_index", "context_compaction", "failure_analysis"):
        spec = spec_for_stage(stage, cfg)
        assert spec is not None and spec.name == cfg.model_qwythos


def test_spec_for_stage_returns_none_for_v17_stages():
    cfg = FusionConfig()
    assert spec_for_stage("planner", cfg) is None
    assert spec_for_stage("judge", cfg) is None


# ── strip_think_blocks (per plan §13.3) ───────────────────────────────


def test_strip_think_blocks_removes_simple_block():
    text = "<think>hidden reasoning</think>actual answer"
    assert strip_think_blocks(text) == "actual answer"


def test_strip_think_blocks_removes_multiline_block():
    text = "<think>line1\nline2\nline3</think>visible"
    assert strip_think_blocks(text) == "visible"


def test_strip_think_blocks_handles_multiple_blocks():
    text = "<think>a</think>middle<think>b</think>end"
    assert strip_think_blocks(text) == "middleend"


def test_strip_think_blocks_passes_through_no_block():
    text = "no think block here"
    assert strip_think_blocks(text) == "no think block here"


def test_strip_think_blocks_strips_whitespace():
    text = " <think>hidden</think>  actual  "
    assert strip_think_blocks(text) == "actual"


def test_should_strip_think_true_for_qwythos():
    cfg = FusionConfig()
    spec = build_qwythos_spec(cfg)
    assert should_strip_think(spec) is True


def test_should_strip_think_false_for_qwable():
    cfg = FusionConfig()
    spec = build_qwable_spec(cfg)
    assert should_strip_think(spec) is False


def test_maybe_strip_think_strips_only_for_qwythos():
    cfg = FusionConfig()
    qwable = build_qwable_spec(cfg)
    qwythos = build_qwythos_spec(cfg)
    text = "<think>reasoning</think>answer"
    # Qwable: keep raw text (no stripping)
    assert maybe_strip_think(text, qwable) == text
    # Qwythos: strip the think block
    assert maybe_strip_think(text, qwythos) == "answer"

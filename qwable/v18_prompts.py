"""v1.8: shared prompt text + helpers for Qwable / Qwythos roles.

This module bundles the *content* of the three new system prompts from
the v1.8 plan §12, plus a `strip_think_blocks` helper from §13.3 that
applies only to models whose spec has `may_emit_think_blocks=True`
(Qwythos).

Keeping the prompt bodies as Python constants (rather than separate .md
files) makes them:
- discoverable in IDEs / grep
- importable in tests
- impossible to drift between the spec and the implementation
"""

from __future__ import annotations

import re

from qwable.config import FusionConfig
from qwable.model_capabilities import (
    LocalModelSpec,
    build_qwable_spec,
    build_qwythos_spec,
)


# ── System prompts (per plan §12) ──────────────────────────────────────

QWABLE_EXECUTOR_SYSTEM = """\
You are the qwable coding executor.
You operate inside a bounded Fable-like agent workflow.

Your job:
- execute exactly one implementation step
- produce a minimal patch or a precise tool request
- follow the patch protocol
- do not redesign the whole system
- do not modify unrelated files
- do not invent files that were not shown unless the plan explicitly asks for new files

Input will include:
- current step
- relevant context pack
- file snippets
- constraints
- patch protocol

Output must be one of:
1. unified_diff
2. structured_patch_json
3. tool_request_json
4. blocked_report_json

Never output final judgment.
Never claim tests passed unless a tool_result says they passed.
"""

QWABLE_REPAIR_SYSTEM = """\
You are the qwable repair agent.
You repair only the current failing test or patch error.

Rules:
- repair the smallest possible surface
- do not refactor unrelated code
- do not rewrite entire modules
- use the test failure tail as primary evidence
- if the failure is ambiguous, request more evidence
- if the same error repeated, produce blocked_report_json

Output must follow the patch protocol.
"""

QWYTHOS_CONTEXT_WORKER_SYSTEM = """\
You are the qwable long-context worker.
Your job is evidence extraction and compression.

You do not make final decisions.
You do not produce patches.
You do not judge the whole project.

Output JSON:
{
  "facts": [],
  "relevant_files": [],
  "symbols": [],
  "constraints": [],
  "risks": [],
  "missing_context": [],
  "compressed_context": ""
}

Remove speculation.
Preserve exact file paths, function names, test names, error messages, and constraints.
"""


# ── Strip helper (per plan §13.3) ─────────────────────────────────────

_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


def strip_think_blocks(text: str) -> str:
    """Remove <think>...</think> blocks from a model response.

    Per plan §13.3, this should only be applied to models whose
    LocalModelSpec has `may_emit_think_blocks=True` (currently just
    Qwythos). The caller is responsible for the check.
    """
    return _THINK_BLOCK_RE.sub("", text).strip()


def should_strip_think(spec: LocalModelSpec) -> bool:
    """Decide whether to strip think blocks for a given model spec.

    Returns True iff the spec's `may_emit_think_blocks` is set. This
    is a thin wrapper so callers don't need to remember the attribute
    name.
    """
    return bool(getattr(spec, "may_emit_think_blocks", False))


def maybe_strip_think(text: str, spec: LocalModelSpec) -> str:
    """Apply strip_think_blocks only when the spec opts in."""
    if should_strip_think(spec):
        return strip_think_blocks(text)
    return text


# ── Spec → system prompt binding (per plan §12) ───────────────────────


def system_prompt_for_stage(stage_name: str, cfg: FusionConfig) -> str | None:
    """Return the v1.8 system prompt for a given v1.8 stage, or None.

    Used by orchestrator when a Qwable / Qwythos stage is invoked. For
    v1.7 stages, returns None and the caller should fall back to the
    v1.7 inline prompts in agent_prompts.py.
    """
    if stage_name in ("execute_patch",):
        return QWABLE_EXECUTOR_SYSTEM
    if stage_name in ("repair_patch",):
        return QWABLE_REPAIR_SYSTEM
    if stage_name in (
        "context_acquisition",
        "repo_index",
        "context_compaction",
        "failure_analysis",
    ):
        return QWYTHOS_CONTEXT_WORKER_SYSTEM
    return None


def spec_for_stage(stage_name: str, cfg: FusionConfig) -> LocalModelSpec | None:
    """Return the LocalModelSpec for the model that handles a given v1.8 stage.

    Returns None for v1.7 stages (planner / critic / judge don't have
    v1.8 specs yet — they use the existing v1.7 inline prompts and
    default model chain).
    """
    if stage_name in ("execute_patch", "repair_patch"):
        return build_qwable_spec(cfg)
    if stage_name in (
        "context_acquisition",
        "repo_index",
        "context_compaction",
        "failure_analysis",
    ):
        return build_qwythos_spec(cfg)
    return None

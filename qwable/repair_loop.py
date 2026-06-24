"""Repair-loop decision policy for agent runs."""

from __future__ import annotations

from dataclasses import dataclass
import re

from qwable.agent_state import AgentRun
from qwable.config import FusionConfig


@dataclass
class RepairDecision:
    should_repair: bool
    reason: str
    failure_summary: str


class RepairLoop:
    def __init__(self, config: FusionConfig):
        self.config = config

    def decide(self, run: AgentRun, latest_failure: str) -> RepairDecision:
        summary = _summarize_failure(latest_failure)

        if run.repair_count >= self.config.agent_max_repair_attempts:
            return RepairDecision(False, "max_repair_attempts_exceeded", summary)

        current_step = run.current_step()
        if current_step is not None and current_step.attempt_count >= 3:
            return RepairDecision(False, "step_attempt_limit_exceeded", summary)

        if _is_blocking_external_or_security_failure(latest_failure):
            return RepairDecision(False, "blocked_external_or_security", summary)

        if not _is_actionable_failure(latest_failure):
            return RepairDecision(False, "not_actionable", summary)

        return RepairDecision(True, "repairable_failure", summary)


_BLOCKED_PATTERNS = (
    r"\bpermission denied\b",
    r"\bunauthorized\b",
    r"\bforbidden\b",
    r"\b401\b",
    r"\b403\b",
    r"\bsudo\b",
    r"\bsecurity\b",
    r"\bpolicy blocked\b",
    r"\bexternal service\b",
    r"\bconnection refused\b",
    r"\bservice unavailable\b",
)

_ACTIONABLE_PATTERNS = (
    r"\bassert(?:ion)?error\b",
    r"\btraceback\b",
    r"\bfailed\b",
    r"\berror\b",
    r"\bexception\b",
    r"\btests?/",
    r"\.py\b",
    r"\bdiff\b",
    r"\bpatch\b",
    r"\bjson\b",
)


def _summarize_failure(latest_failure: str) -> str:
    compact = re.sub(r"\s+", " ", latest_failure.strip())
    return compact[:500]


def _is_blocking_external_or_security_failure(latest_failure: str) -> bool:
    lowered = latest_failure.lower()
    return any(re.search(pattern, lowered) for pattern in _BLOCKED_PATTERNS)


def _is_actionable_failure(latest_failure: str) -> bool:
    lowered = latest_failure.lower()
    return any(re.search(pattern, lowered) for pattern in _ACTIONABLE_PATTERNS)

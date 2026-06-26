"""Compact deterministic context passed into agent workflow prompts."""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ContextFileSummary:
    path: str
    reason: str
    summary: str
    symbols: list[str] = field(default_factory=list)
    score: float = 0.0


@dataclass
class ContextPack:
    goal: str
    workflow: str
    files: list[ContextFileSummary] = field(default_factory=list)
    raw_evidence: list[str] = field(default_factory=list)
    constraints: list[str] = field(default_factory=list)
    risks: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_prompt_text(self, max_chars: int) -> str:
        sections = [f"CONTEXT_PACK\nworkflow={self.workflow}\ngoal={self.goal}\n"]
        if self.constraints:
            sections.append(
                "CONSTRAINTS:\n" + "\n".join(f"- {item}" for item in self.constraints)
            )
        if self.files:
            sections.append(
                "FILES:\n"
                + "\n".join(
                    f"- {file.path}: {file.reason}\n"
                    f"  summary: {file.summary}\n"
                    f"  symbols: {', '.join(file.symbols)}"
                    for file in self.files
                )
            )
        if self.raw_evidence:
            sections.append("RAW_EVIDENCE:\n" + "\n---\n".join(self.raw_evidence))
        if self.risks:
            sections.append("RISKS:\n" + "\n".join(f"- {item}" for item in self.risks))
        return "\n\n".join(sections)[:max_chars]

"""G10: Fusion synthesis prompt building and structured output parsing.

Public surface:
  - build_synthesis_prompt(synthesis) -> (system, user)
  - parse_structured_output(judge_text) -> FusionStructuredOutput
  - FUSION_SECTION_NAMES: list of expected section keys
"""

from dataclasses import dataclass, field

from qwable.fusion_schemas import SynthesisInput
from qwable.prompts import FUSION_AGENT_JUDGE_SYSTEM


FUSION_SECTION_NAMES: list[str] = [
    "final_answer",
    "consensus",
    "contradictions",
    "blind_spots",
    "per_model_notes",
]


# Header strings as they appear in judge markdown output
_SECTION_HEADERS: list[tuple[str, str]] = [
    ("## Final Answer", "final_answer"),
    ("## Consensus", "consensus"),
    ("## Contradictions", "contradictions"),
    ("## Blind Spots", "blind_spots"),
    ("## Per-model Notes", "per_model_notes"),
]


@dataclass
class FusionStructuredOutput:
    """Parsed 5-section output from the judge model.

    `had_fallback` is True if any expected section was missing in the raw text;
    in that case the corresponding field will be empty (or raw_text becomes the
    final_answer when no sections at all are present).
    """

    final_answer: str
    consensus: list[str] = field(default_factory=list)
    contradictions: list[str] = field(default_factory=list)
    blind_spots: list[str] = field(default_factory=list)
    per_model_notes: dict[str, str] = field(default_factory=dict)
    raw_text: str = ""
    had_fallback: bool = False


# ─── build_synthesis_prompt ───────────────────────────────────────────────


def build_synthesis_prompt(synthesis: SynthesisInput) -> tuple[str, str]:
    """Build (system_prompt, user_prompt) for the judge model.

    User prompt structure:
      ## Original Prompt
      <text>

      ## Panel Responses
      ### model_id_1
      <response text or [ERROR: ...]>
      ### model_id_2
      ...

      _Preset: <preset_name>_
    """
    system = FUSION_AGENT_JUDGE_SYSTEM

    parts: list[str] = [
        "## Original Prompt\n",
        synthesis.original_prompt,
        "",
        "## Panel Responses\n",
    ]
    for resp in synthesis.panel_responses:
        parts.append(f"### {resp.model_id}")
        if resp.error:
            parts.append(f"[ERROR: {resp.error}]")
        else:
            parts.append(resp.text)
        parts.append("")

    parts.append(f"_Preset: {synthesis.preset_name}_")

    return system, "\n".join(parts)


# ─── parse_structured_output ──────────────────────────────────────────────


def _split_sections(text: str) -> dict[str, str]:
    """Split judge markdown into {section_name: content} dict."""
    sections = {name: "" for _, name in _SECTION_HEADERS}
    current: str | None = None
    current_lines: list[str] = []
    for line in text.splitlines():
        matched = False
        for header, name in _SECTION_HEADERS:
            if line.strip().startswith(header):
                if current is not None:
                    sections[current] = "\n".join(current_lines).strip()
                current = name
                current_lines = []
                matched = True
                break
        if not matched and current is not None:
            current_lines.append(line)
    if current is not None:
        sections[current] = "\n".join(current_lines).strip()
    return sections


def _parse_bullet_list(text: str) -> list[str]:
    """Parse markdown bullet list (`- item` or `* item`) into list[str]."""
    items: list[str] = []
    for line in text.splitlines():
        s = line.strip()
        if s.startswith("- "):
            items.append(s[2:].strip())
        elif s.startswith("* "):
            items.append(s[2:].strip())
    return items


def _parse_per_model_notes(text: str) -> dict[str, str]:
    """Parse Per-model Notes section: `### model_id\\n<note>` into dict."""
    notes: dict[str, str] = {}
    current_id: str | None = None
    current_lines: list[str] = []
    for line in text.splitlines():
        s = line.strip()
        if s.startswith("### "):
            if current_id is not None:
                notes[current_id] = "\n".join(current_lines).strip()
            current_id = s[4:].strip()
            current_lines = []
        elif current_id is not None:
            current_lines.append(line)
    if current_id is not None:
        notes[current_id] = "\n".join(current_lines).strip()
    return notes


def parse_structured_output(judge_text: str) -> FusionStructuredOutput:
    """Parse judge markdown into FusionStructuredOutput.

    Fallback policy:
    - If ALL 5 sections missing → final_answer = raw_text, others empty, had_fallback=True
    - If some sections missing → those fields are empty, had_fallback=True
    - If final_answer section present but others missing → those empty, had_fallback=True
    """
    sections = _split_sections(judge_text)
    expected_keys = [name for _, name in _SECTION_HEADERS]
    had_fallback = any(not sections[k] for k in expected_keys)

    if had_fallback and not sections["final_answer"]:
        # Completely unstructured
        return FusionStructuredOutput(
            final_answer=judge_text.strip(),
            consensus=[],
            contradictions=[],
            blind_spots=[],
            per_model_notes={},
            raw_text=judge_text,
            had_fallback=True,
        )

    return FusionStructuredOutput(
        final_answer=sections["final_answer"] or judge_text.strip(),
        consensus=_parse_bullet_list(sections["consensus"]),
        contradictions=_parse_bullet_list(sections["contradictions"]),
        blind_spots=_parse_bullet_list(sections["blind_spots"]),
        per_model_notes=_parse_per_model_notes(sections["per_model_notes"]),
        raw_text=judge_text,
        had_fallback=had_fallback,
    )

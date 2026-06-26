"""Tests for fusion synthesis prompt builder and output parser."""

from qwable.fusion_schemas import PanelResponse, SynthesisInput
from qwable.fusion_synthesis import (
    FUSION_SECTION_NAMES,
    build_synthesis_prompt,
    parse_structured_output,
)


# ─── build_synthesis_prompt ───────────────────────────────────────────────


def test_build_synthesis_includes_system_prompt_with_all_5_sections():
    """System prompt must list all 5 required section headers."""
    si = SynthesisInput(
        original_prompt="q",
        panel_responses=[PanelResponse(model_id="m1", text="a")],
        preset_name="quality",
    )
    system, user = build_synthesis_prompt(si)
    for header in [
        "## Final Answer",
        "## Consensus",
        "## Contradictions",
        "## Blind Spots",
        "## Per-model Notes",
    ]:
        assert header in system, f"system prompt missing header {header}"


def test_build_synthesis_includes_original_prompt():
    si = SynthesisInput(
        original_prompt="What is the best sort algorithm for 50k records?",
        panel_responses=[],
        preset_name="quality",
    )
    _, user = build_synthesis_prompt(si)
    assert "What is the best sort algorithm for 50k records?" in user


def test_build_synthesis_includes_all_panel_responses():
    responses = [
        PanelResponse(model_id="qwen-coder", text="quicksort is best"),
        PanelResponse(model_id="deepseek-r1", text="mergesort for stability"),
        PanelResponse(model_id="qwen3.6", text="depends on data distribution"),
    ]
    si = SynthesisInput(
        original_prompt="q",
        panel_responses=responses,
        preset_name="quality",
    )
    _, user = build_synthesis_prompt(si)
    assert "### qwen-coder" in user
    assert "quicksort is best" in user
    assert "### deepseek-r1" in user
    assert "mergesort for stability" in user
    assert "### qwen3.6" in user
    assert "depends on data distribution" in user


def test_build_synthesis_marks_error_responses():
    """Failed panel responses should be tagged [ERROR: ...]."""
    responses = [
        PanelResponse(model_id="good-model", text="real answer"),
        PanelResponse(model_id="bad-model", text="", error="timeout after 30s"),
    ]
    si = SynthesisInput(
        original_prompt="q",
        panel_responses=responses,
        preset_name="quality",
    )
    _, user = build_synthesis_prompt(si)
    assert "### bad-model" in user
    assert "[ERROR: timeout after 30s]" in user


def test_build_synthesis_includes_preset_name():
    si = SynthesisInput(
        original_prompt="q",
        panel_responses=[],
        preset_name="coding",
    )
    _, user = build_synthesis_prompt(si)
    assert "coding" in user


# ─── parse_structured_output ──────────────────────────────────────────────


COMPLETE_JUDGE_OUTPUT = """\
## Final Answer
Use mergesort for stable ordering with 50k records.

## Consensus
- Both models agree stability matters
- Quick sort has worst case O(n^2)

## Contradictions
- qwen-coder prefers quicksort
- deepseek-r1 prefers mergesort; resolution: use mergesort for stability

## Blind Spots
- Memory usage not analyzed by panel

## Per-model Notes
### qwen-coder
Suggested quicksort with random pivot.

### deepseek-r1
Suggested mergesort for guaranteed O(n log n).

### qwen3.6
Pointed out data distribution matters.
"""


def test_parse_output_with_all_sections_present():
    out = parse_structured_output(COMPLETE_JUDGE_OUTPUT)
    assert out.had_fallback is False
    assert "Use mergesort" in out.final_answer
    assert "Both models agree stability matters" in out.consensus
    assert any("Quick sort has worst case" in item for item in out.consensus)
    assert len(out.contradictions) >= 1
    assert "Memory usage" in out.blind_spots[0]
    assert "qwen-coder" in out.per_model_notes
    assert "quicksort with random pivot" in out.per_model_notes["qwen-coder"]
    assert "deepseek-r1" in out.per_model_notes
    assert "qwen3.6" in out.per_model_notes
    assert out.raw_text == COMPLETE_JUDGE_OUTPUT


def test_parse_output_missing_consensus_uses_fallback():
    text = """\
## Final Answer
Just do X.

## Contradictions
- none

## Blind Spots
- none

## Per-model Notes
### m1
notes
"""
    out = parse_structured_output(text)
    assert out.had_fallback is True
    assert out.consensus == []
    assert "Just do X" in out.final_answer


def test_parse_output_missing_multiple_sections_uses_fallback():
    text = """\
## Final Answer
Just do X.
"""
    out = parse_structured_output(text)
    assert out.had_fallback is True
    assert out.consensus == []
    assert out.contradictions == []
    assert out.blind_spots == []
    assert out.per_model_notes == {}
    assert "Just do X" in out.final_answer


def test_parse_output_completely_unstructured_uses_raw_as_final_answer():
    text = "I think the answer is 42 because reasons."
    out = parse_structured_output(text)
    assert out.had_fallback is True
    assert out.final_answer == text
    assert out.consensus == []
    assert out.per_model_notes == {}


def test_parse_output_empty_string():
    out = parse_structured_output("")
    assert out.had_fallback is True
    assert out.final_answer == ""
    assert out.consensus == []


def test_parse_output_section_names_constant():
    """FUSION_SECTION_NAMES should list all 5 expected sections."""
    assert FUSION_SECTION_NAMES == [
        "final_answer",
        "consensus",
        "contradictions",
        "blind_spots",
        "per_model_notes",
    ]


def test_parse_output_handles_asterisk_bullets():
    text = """\
## Final Answer
X.

## Consensus
* point one
* point two
"""
    out = parse_structured_output(text)
    assert "point one" in out.consensus
    assert "point two" in out.consensus


def test_parse_output_handles_per_model_notes_with_extra_whitespace():
    text = """\
## Final Answer
X.

## Per-model Notes
### model-a

  multi-line note
  with indent

### model-b
single line
"""
    out = parse_structured_output(text)
    assert "model-a" in out.per_model_notes
    assert "multi-line note" in out.per_model_notes["model-a"]
    assert "model-b" in out.per_model_notes
    assert "single line" in out.per_model_notes["model-b"]

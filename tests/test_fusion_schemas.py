"""Tests for fusion deliberation dataclasses."""

from qwable.fusion_schemas import FusionRequest, PanelResponse, SynthesisInput


def test_fusion_request_defaults():
    """Empty FusionRequest should have all None defaults."""
    req = FusionRequest()
    assert req.preset is None
    assert req.analysis_models is None
    assert req.judge_model is None


def test_fusion_request_with_all_fields():
    """FusionRequest should accept preset, analysis_models, judge_model."""
    req = FusionRequest(
        preset="quality",
        analysis_models=["m1", "m2"],
        judge_model="m3",
    )
    assert req.preset == "quality"
    assert req.analysis_models == ["m1", "m2"]
    assert req.judge_model == "m3"


def test_panel_response_defaults():
    """PanelResponse should default finish_reason='stop', latency_ms=0, error=None."""
    pr = PanelResponse(model_id="m1", text="hello")
    assert pr.model_id == "m1"
    assert pr.text == "hello"
    assert pr.finish_reason == "stop"
    assert pr.latency_ms == 0
    assert pr.error is None


def test_panel_response_with_error():
    """PanelResponse should accept error message for failed model."""
    pr = PanelResponse(model_id="m1", text="", error="timeout")
    assert pr.error == "timeout"
    assert pr.text == ""


def test_synthesis_input_holds_responses():
    """SynthesisInput should hold prompt + panel responses + preset name."""
    responses = [
        PanelResponse(model_id="m1", text="a"),
        PanelResponse(model_id="m2", text="b"),
    ]
    si = SynthesisInput(
        original_prompt="q",
        panel_responses=responses,
        preset_name="quality",
    )
    assert si.original_prompt == "q"
    assert len(si.panel_responses) == 2
    assert si.preset_name == "quality"
    assert si.panel_responses[0].model_id == "m1"

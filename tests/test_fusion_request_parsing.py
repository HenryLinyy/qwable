"""Tests for fusion request body extraction."""

from qwable.fusion_request import extract_fusion_request


def test_extract_from_empty_body():
    """Empty body should yield empty FusionRequest (caller resolves default)."""
    req = extract_fusion_request({})
    assert req.preset is None
    assert req.analysis_models is None
    assert req.judge_model is None


def test_extract_from_plugins_shape():
    """OpenRouter-style plugins shape: plugins[].id == 'fusion'."""
    body = {"plugins": [{"id": "fusion", "preset": "quality"}]}
    req = extract_fusion_request(body)
    assert req.preset == "quality"


def test_extract_from_plugins_shape_with_extra_fields():
    body = {
        "plugins": [
            {"id": "web", "enabled": True},
            {"id": "fusion", "preset": "coding", "max_models": 5},
        ]
    }
    req = extract_fusion_request(body)
    assert req.preset == "coding"


def test_extract_plugins_ignores_non_fusion_plugins():
    body = {"plugins": [{"id": "web", "max_results": 3}]}
    req = extract_fusion_request(body)
    assert req.preset is None
    assert req.analysis_models is None


def test_extract_from_top_level_fusion_block():
    """Simplified top-level fusion shape."""
    body = {"fusion": {"preset": "coding"}}
    req = extract_fusion_request(body)
    assert req.preset == "coding"


def test_extract_from_top_level_full_override():
    body = {
        "fusion": {
            "preset": "quality",
            "analysis_models": ["m1", "m2", "m3"],
            "judge_model": "m-judge",
        }
    }
    req = extract_fusion_request(body)
    assert req.preset == "quality"
    assert req.analysis_models == ["m1", "m2", "m3"]
    assert req.judge_model == "m-judge"


def test_extract_top_level_custom_panel_without_preset():
    body = {"fusion": {"analysis_models": ["m1", "m2"]}}
    req = extract_fusion_request(body)
    assert req.preset is None
    assert req.analysis_models == ["m1", "m2"]


def test_extract_plugins_takes_priority_over_top_level():
    """If both shapes present, plugins wins (matches OpenRouter convention)."""
    body = {
        "plugins": [{"id": "fusion", "preset": "quality"}],
        "fusion": {"preset": "budget"},
    }
    req = extract_fusion_request(body)
    assert req.preset == "quality"


def test_extract_invalid_fusion_block_falls_back():
    """If 'fusion' key exists but is not a dict, treat as no override."""
    body = {"fusion": "not-a-dict"}
    req = extract_fusion_request(body)
    assert req.preset is None


def test_extract_plugins_not_a_list_falls_back():
    body = {"plugins": "not-a-list"}
    req = extract_fusion_request(body)
    assert req.preset is None


def test_extract_preserves_other_request_fields():
    """extract should NOT mutate the input body."""
    body = {"model": "qwable-fusion", "fusion": {"preset": "quality"}}
    req = extract_fusion_request(body)
    assert req.preset == "quality"
    assert body == {"model": "qwable-fusion", "fusion": {"preset": "quality"}}

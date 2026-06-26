"""Preset-by-model-name aliases (GUI clients can pick preset via the model name)."""

from qwable.fusion_request import extract_fusion_request
from qwable.profiles import resolve_profile


def test_alias_model_names_map_to_fusion_agent():
    for p in ("budget", "quality", "coding", "heavy"):
        assert (
            resolve_profile(f"qwable-fusion-{p}", "openai_responses") == "fusion-agent"
        )
        assert resolve_profile(f"qwable-fusion-{p}", "openai_chat") == "fusion-agent"
        assert (
            resolve_profile(f"claude-qwable-fusion-{p}", "anthropic_messages")
            == "fusion-agent"
        )


def test_preset_derived_from_model_name():
    r = extract_fusion_request({"model": "qwable-fusion-quality"})
    assert r.preset == "quality"
    r2 = extract_fusion_request({"model": "claude-qwable-fusion-heavy"})
    assert r2.preset == "heavy"


def test_plain_fusion_name_has_no_preset_override():
    # plain name -> no preset (gateway applies its default)
    assert extract_fusion_request({"model": "qwable-fusion"}).preset is None


def test_explicit_fusion_block_wins_over_model_name():
    r = extract_fusion_request(
        {"model": "qwable-fusion-budget", "fusion": {"preset": "quality"}}
    )
    assert r.preset == "quality"


def test_default_preset_is_budget():
    from qwable.config import FusionConfig

    assert FusionConfig().fusion_default_preset == "budget"
    assert FusionConfig().lmstudio_ttl_seconds == 600

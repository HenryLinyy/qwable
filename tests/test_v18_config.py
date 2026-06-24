"""v1.8 config tests — Qwable / Qwythos settings.

Regression suite for the v1.8 plan §5 (Config 變更):
- ENABLE_QWABLE_EXECUTOR defaults to True
- ENABLE_QWYTHOS_LONG_CONTEXT defaults to False (opt-in)
- All Qwable / Qwythos env-var overrides work
- The original v1.7 fields still work (no removal)
"""

import os

from qwable.config import FusionConfig


def test_v18_qwable_enabled_by_default():
    """Qwable executor must be ON by default per the v1.8 plan."""
    cfg = FusionConfig()
    assert cfg.enable_qwable_executor is True


def test_v18_qwythos_disabled_by_default():
    """Qwythos must be OFF by default (opt-in per the v1.8 plan)."""
    cfg = FusionConfig()
    assert cfg.enable_qwythos_long_context is False


def test_v18_qwable_model_defaults():
    """Qwable default model id matches the id LM Studio actually serves."""
    cfg = FusionConfig()
    assert cfg.model_qwable == "qwable-9b-claude-fable-5"
    assert cfg.model_qwable_runtime == "lmstudio"
    assert cfg.model_qwable_context_limit == 32768
    assert cfg.model_qwable_temperature == 0.25
    assert cfg.model_qwable_top_p == 0.9
    assert cfg.model_qwable_repeat_penalty == 1.05


def test_v18_qwythos_model_defaults():
    cfg = FusionConfig()
    assert cfg.model_qwythos == "qwythos-9b-claude-mythos-5-1m"
    assert cfg.model_qwythos_context_limit == 65536
    assert cfg.model_qwythos_max_context_limit == 131072
    assert cfg.model_qwythos_top_k == 20


def test_v18_fallback_flags_default_true():
    """All three fallback safety flags default to True (per v1.8 plan)."""
    cfg = FusionConfig()
    assert cfg.enable_executor_fallback is True
    assert cfg.enable_repair_fallback is True
    assert cfg.enable_long_context_fallback is True


def test_v18_health_check_defaults():
    """Health check is opt-in (off by default) and timeout=30s."""
    cfg = FusionConfig()
    assert cfg.model_health_check_on_startup is False
    assert cfg.model_health_check_timeout_seconds == 30


def test_v18_env_override_qwable_executor():
    """Setting ENABLE_QWABLE_EXECUTOR=false must flip the flag."""
    os.environ["ENABLE_QWABLE_EXECUTOR"] = "false"
    try:
        cfg = FusionConfig()
        assert cfg.enable_qwable_executor is False
    finally:
        del os.environ["ENABLE_QWABLE_EXECUTOR"]


def test_v18_env_override_qwable_model():
    os.environ["MODEL_QWABLE"] = "custom-qwable-7b"
    try:
        cfg = FusionConfig()
        assert cfg.model_qwable == "custom-qwable-7b"
    finally:
        del os.environ["MODEL_QWABLE"]


def test_v18_env_override_qwythos_enable():
    os.environ["ENABLE_QWYTHOS_LONG_CONTEXT"] = "true"
    try:
        cfg = FusionConfig()
        assert cfg.enable_qwythos_long_context is True
    finally:
        del os.environ["ENABLE_QWYTHOS_LONG_CONTEXT"]


def test_v18_does_not_break_v17_model_coder():
    """v1.7 fields (model_coder etc.) must remain intact after v1.8 changes."""
    cfg = FusionConfig()
    assert cfg.model_coder == "qwen/qwen3-coder-next"
    assert cfg.model_critic == "deepseek-r1-distill-qwen-32b"
    assert cfg.model_judge == "deepseek-r1-distill-qwen-32b"
    assert cfg.model_heavy == "deepseek-v4-flash"

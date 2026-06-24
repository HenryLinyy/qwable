"""Tests for config module."""

import os
from qwable.config import FusionConfig


def test_default_config():
    """Config should have sensible defaults."""
    cfg = FusionConfig()
    assert cfg.qwable_host == "127.0.0.1"
    assert cfg.qwable_port == 8088
    assert cfg.local_model_backend == "lmstudio"
    assert cfg.ollama_base_url == "http://127.0.0.1:1234/v1"
    assert cfg.lmstudio_cli_path.endswith(".lmstudio/bin/lms")
    assert os.path.isabs(cfg.lmstudio_cli_path)  # ~ expanded to an absolute path
    assert cfg.ds4_base_url == "http://127.0.0.1:8000/v1"
    assert cfg.qwable_default_profile == "fast-agent"


def test_env_override():
    """Config should respect environment variables."""
    os.environ["QWABLE_PORT"] = "9090"
    cfg = FusionConfig()
    assert cfg.qwable_port == 9090
    del os.environ["QWABLE_PORT"]


def test_model_defaults():
    """Config model defaults should be sensible."""
    cfg = FusionConfig()
    # Gate G09 switches local non-ds4 routes to LM Studio model identifiers.
    assert cfg.model_fast == "google/gemma-4-26b-a4b-qat"
    assert cfg.model_coder == "qwen/qwen3-coder-next"
    assert cfg.model_tooler == "qwen/qwen3-coder-next"
    assert cfg.model_formatter == "google/gemma-4-26b-a4b-qat"
    assert cfg.model_vision_fast == "google/gemma-4-26b-a4b-qat"
    assert cfg.model_vision_pro == "qwen/qwen3-vl-30b"
    assert cfg.model_agentic_pro == "qwen/qwen3.6-35b-a3b"
    assert cfg.model_hermes_pro == "qwen/qwen3.6-35b-a3b"
    assert cfg.model_agentic_mlx == "qwen/qwen3.6-35b-a3b"
    assert cfg.model_formatter_mlx == "google/gemma-4-26b-a4b-qat"
    assert cfg.model_heavy == "deepseek-v4-flash"
    assert cfg.fast_max_input_chars == 24000
    # Bumped 1200 -> 1500: thinking MLX models (gemma4:12b-mlx,
    # qwen3.6:35b-a3b-nvfp4) eat ~500 tokens for chain-of-thought; 1500
    # leaves ~1000 tokens of content margin for fast-agent / chat-agent
    # / formatter-mlx clients that forget to set max_tokens explicitly.
    assert cfg.fast_max_tokens == 1500


def test_default_path_uses_lmstudio_ids_not_ollama_tags():
    """Qwable should route local models through LM Studio identifiers.

    Keep ds4 as the heavy remote backend, but all non-ds4 local profile defaults
    should use LM Studio model ids instead of Ollama tags such as qwen3-coder:30b.
    """
    cfg = FusionConfig()
    local_models = [
        cfg.model_fast,
        cfg.model_coder,
        cfg.model_tooler,
        cfg.model_critic,
        cfg.model_judge,
        cfg.model_formatter,
        cfg.model_vision_fast,
        cfg.model_vision_pro,
        cfg.model_agentic_pro,
        cfg.model_hermes_pro,
        cfg.model_agentic_mlx,
        cfg.model_formatter_mlx,
    ]
    assert "qwen3-coder:30b" not in local_models
    assert all(("/" in model or model == "deepseek-r1-distill-qwen-32b") for model in local_models)


def test_v15_memory_estimates_are_serial_not_parallel_defaults():
    """Large pro/vision/heavy models should be configured for serial execution by default."""
    cfg = FusionConfig()
    assert cfg.m5_allow_full_parallel is False
    assert cfg.est_model_vision_fast_gb == 16
    assert cfg.est_model_vision_pro_gb == 34
    assert cfg.est_model_agentic_pro_gb == 38
    assert cfg.est_model_agentic_mlx_gb == 38
    assert cfg.est_model_formatter_mlx_gb == 16
    assert cfg.est_model_heavy_gb == 90
    assert cfg.est_model_fast_gb == 16
    assert cfg.est_model_coder_gb == 65
    assert cfg.est_model_tooler_gb == 65
    assert cfg.est_model_critic_gb == 66
    assert cfg.est_model_judge_gb == 66

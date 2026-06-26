"""Configuration for Qwable Agent Gateway."""

import os

from pydantic import Field
from pydantic_settings import BaseSettings


class FusionConfig(BaseSettings):
    """Configuration loaded from .env or environment variables."""

    # Server
    qwable_host: str = "127.0.0.1"
    qwable_port: int = 8088
    qwable_default_profile: str = "fast-agent"
    qwable_timeout_seconds: int = 900
    qwable_queue_timeout_seconds: int = 5
    qwable_max_concurrent_requests: int = 1

    # Ollama
    # Historical name kept for compatibility: this is the OpenAI-compatible
    # local model backend base URL. Gate G09 routes local models through
    # LM Studio instead of Ollama so LM Studio-managed MLX models are used.
    local_model_backend: str = "lmstudio"
    ollama_base_url: str = "http://127.0.0.1:1234/v1"
    lmstudio_cli_path: str = Field(
        default_factory=lambda: os.path.expanduser("~/.lmstudio/bin/lms")
    )
    # JIT model auto-unload TTL (seconds) sent to LM Studio so idle models free
    # memory faster than the 1h default. 0 = don't send (use LM Studio default).
    lmstudio_ttl_seconds: int = 600

    # ds4
    ds4_base_url: str = "http://127.0.0.1:8000/v1"
    ds4_model: str = "deepseek-v4-flash"
    ds4_timeout_seconds: int = 1200
    ds4_max_input_chars: int = 160000
    ds4_max_concurrent_requests: int = 1

    # Model names — LM Studio model ids from ~/.lmstudio/hub/models.
    model_fast: str = "google/gemma-4-26b-a4b-qat"
    model_coder: str = "qwen/qwen3-coder-next"
    model_tooler: str = "qwen/qwen3-coder-next"
    model_critic: str = "deepseek-r1-distill-qwen-32b"
    model_judge: str = "deepseek-r1-distill-qwen-32b"
    model_formatter: str = "google/gemma-4-26b-a4b-qat"
    model_vision_fast: str = "google/gemma-4-26b-a4b-qat"
    model_vision_pro: str = "qwen/qwen3-vl-30b"
    model_vision_pro_fallback: str = ""
    model_agentic_pro: str = "qwen/qwen3.6-35b-a3b"
    model_hermes_pro: str = "qwen/qwen3.6-35b-a3b"
    model_agentic_mlx: str = "qwen/qwen3.6-35b-a3b"
    model_formatter_mlx: str = "google/gemma-4-26b-a4b-qat"
    model_heavy: str = "deepseek-v4-flash"

    # G10 Fusion deliberation router
    fusion_default_preset: str = "budget"
    fusion_max_tokens_panel: int = 1500
    fusion_max_tokens_judge: int = 3600

    # G11 MLX formatter preference
    prefer_mlx_formatter: bool = Field(
        default=True, validation_alias="QWABLE_PREFER_MLX_FORMATTER"
    )
    mlx_formatter_max_chars: int = Field(
        default=1000, validation_alias="QWABLE_MLX_FORMATTER_MAX_CHARS"
    )

    # G12-3: keep last panel + judge model resident for fast follow-up requests
    # (LM Studio TTL=1h handles eventual eviction; default true)
    keep_last_panel_resident: bool = True

    # G12-5: judge fallback chain — tried in order if primary judge fails.
    # Default: 3 ollama models of varying sizes, each cheaper than the previous.
    fusion_judge_fallback_chain: str = (
        "qwen/qwen3.6-35b-a3b,google/gemma-4-26b-a4b-qat,deepseek-r1-distill-qwen-32b"
    )

    # G13-3: retry on transient failures (timeouts, connection errors).
    # Disabled by default — enable if you see flakiness from LM Studio / ds4.
    fusion_max_retries: int = 0  # 0 = no retry, 2 = up to 2 retries (3 total)
    fusion_retry_base_delay: float = 1.0  # seconds; doubles each retry (1s, 2s, 4s)

    # Agent runtime
    agent_runtime_enabled: bool = True
    agent_store_path: str = ".qwable_agent_runs.sqlite3"
    agent_max_steps: int = 12
    agent_max_tool_calls: int = 40
    agent_max_repair_attempts: int = 3
    agent_max_runtime_seconds: int = 1800
    agent_enable_context_compaction: bool = True
    agent_context_pack_max_chars: int = 64000
    agent_repo_index_max_files: int = 300
    agent_trace_enabled: bool = True
    agent_replay_enabled: bool = True

    # Workflow context limits
    agentic_workflow_max_input_chars: int = 256000
    coding_workflow_max_input_chars: int = 256000
    review_workflow_max_input_chars: int = 160000

    # Role fallback chain as comma-separated model ids
    model_role_simple_formatter: str = "google/gemma-4-26b-a4b-qat"
    model_role_planner: str = "qwen/qwen3.6-35b-a3b"
    model_role_executor: str = "qwen/qwen3-coder-next"
    model_role_repair: str = "qwen/qwen3-coder-next"
    model_role_critic: str = "deepseek-r1-distill-qwen-32b"
    model_role_judge: str = "qwen/qwen3.6-35b-a3b"
    model_role_heavy_primary: str = "deepseek-v4-flash"
    model_role_vision: str = "qwen/qwen3-vl-30b"
    model_role_planner_fallback_chain: str = (
        "qwen/qwen3.6-35b-a3b,qwen/qwen3-coder-next,google/gemma-4-26b-a4b-qat"
    )
    model_role_executor_fallback_chain: str = (
        "qwen/qwen3-coder-next,qwen/qwen3.6-35b-a3b"
    )
    model_role_repair_fallback_chain: str = "qwen/qwen3-coder-next,qwen/qwen3.6-35b-a3b"
    model_role_critic_fallback_chain: str = (
        "deepseek-r1-distill-qwen-32b,qwen/qwen3.6-35b-a3b,google/gemma-4-26b-a4b-qat"
    )
    model_role_judge_fallback_chain: str = (
        "qwen/qwen3.6-35b-a3b,deepseek-r1-distill-qwen-32b,google/gemma-4-26b-a4b-qat"
    )

    # M5 resource estimates
    m5_unified_memory_gb: int = 128
    m5_reserved_memory_gb: int = 28
    m5_allow_full_parallel: bool = False
    m5_kv_cache_reserve_gb_per_parallel_model: int = 5

    est_model_fast_gb: int = 16
    est_model_coder_gb: int = 65
    est_model_tooler_gb: int = 65
    est_model_critic_gb: int = 66
    est_model_judge_gb: int = 66
    est_model_formatter_gb: int = 16
    est_model_vision_fast_gb: int = 16
    est_model_vision_pro_gb: int = 34
    est_model_agentic_pro_gb: int = 38
    est_model_hermes_pro_gb: int = 38
    est_model_agentic_mlx_gb: int = 38
    est_model_formatter_mlx_gb: int = 16
    est_model_heavy_gb: int = 90

    # Context / output limits
    fast_max_input_chars: int = 24000
    full_max_input_chars: int = 96000
    heavy_max_input_chars: int = 160000
    agentic_mlx_max_input_chars: int = 256000
    # Keep a generous default for clients that omit max_tokens. This also
    # leaves enough room if MODEL_FAST/MODEL_CODER are temporarily overridden
    # to reasoning models for targeted probes.
    fast_max_tokens: int = 1500
    full_panel_max_tokens: int = 1800
    full_judge_max_tokens: int = 3600
    heavy_max_tokens: int = 3600
    # qwen3.6:35b-a3b-nvfp4 is a thinking/reasoning model; chain-of-thought
    # eats ~500 tokens, so the default output budget must be >= 600 to leave
    # room for actual content. Override via env if you need longer replies.
    agentic_mlx_max_tokens: int = 600
    vision_max_tokens: int = 1600
    vision_max_images: int = 8
    vision_max_image_mb: int = 16
    vision_fast_max_input_chars: int = 24000
    vision_pro_max_input_chars: int = 96000

    # Streaming
    stream_keepalive_seconds: int = 10
    stream_chunk_chars: int = 1000

    # v1.8: optional Fable/Mythos-style local models
    enable_qwable_executor: bool = True
    model_qwable: str = "qwable-9b-claude-fable-5"
    model_qwable_runtime: str = "lmstudio"
    model_qwable_context_limit: int = 32768
    model_qwable_temperature: float = 0.25
    model_qwable_top_p: float = 0.9
    model_qwable_repeat_penalty: float = 1.05

    enable_qwythos_long_context: bool = False
    model_qwythos: str = "qwythos-9b-claude-mythos-5-1m"
    model_qwythos_runtime: str = "lmstudio"
    model_qwythos_context_limit: int = 65536
    model_qwythos_max_context_limit: int = 131072
    model_qwythos_temperature: float = 0.6
    model_qwythos_top_p: float = 0.95
    model_qwythos_top_k: int = 20
    model_qwythos_repeat_penalty: float = 1.05

    enable_executor_fallback: bool = True
    enable_repair_fallback: bool = True
    enable_long_context_fallback: bool = True

    model_health_check_on_startup: bool = False
    model_health_check_timeout_seconds: int = 30

    model_config = {"env_prefix": "", "populate_by_name": True}

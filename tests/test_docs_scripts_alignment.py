"""Contract checks for documented setup files and helper scripts."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_env_example_documents_all_runtime_knobs():
    env_example = ROOT / ".env.example"
    assert env_example.exists()
    text = env_example.read_text()
    required_keys = [
        "QWABLE_HOST=127.0.0.1",
        "QWABLE_PORT=8088",
        "LOCAL_MODEL_BACKEND=lmstudio",
        "OLLAMA_BASE_URL=http://127.0.0.1:1234/v1",
        "LMSTUDIO_CLI_PATH=$HOME/.lmstudio/bin/lms",
        "DS4_BASE_URL=http://127.0.0.1:8000/v1",
        "DS4_MODEL=deepseek-v4-flash",
        "MODEL_FAST=google/gemma-4-26b-a4b-qat",
        "MODEL_CODER=qwen/qwen3-coder-next",
        "MODEL_TOOLER=qwen/qwen3-coder-next",
        "MODEL_CRITIC=deepseek-r1-distill-qwen-32b",
        "MODEL_JUDGE=deepseek-r1-distill-qwen-32b",
        "MODEL_FORMATTER=google/gemma-4-26b-a4b-qat",
        "MODEL_VISION_FAST=google/gemma-4-26b-a4b-qat",
        "MODEL_VISION_PRO=qwen/qwen3-vl-30b",
        "MODEL_AGENTIC_PRO=qwen/qwen3.6-35b-a3b",
        "MODEL_HERMES_PRO=qwen/qwen3.6-35b-a3b",
        "MODEL_AGENTIC_MLX=qwen/qwen3.6-35b-a3b",
        "MODEL_FORMATTER_MLX=google/gemma-4-26b-a4b-qat",
        "MODEL_HEAVY=deepseek-v4-flash",
        "M5_KV_CACHE_RESERVE_GB_PER_PARALLEL_MODEL=5",
        "FAST_MAX_INPUT_CHARS=24000",
        "FULL_MAX_INPUT_CHARS=96000",
        "HEAVY_MAX_INPUT_CHARS=160000",
        "STREAM_KEEPALIVE_SECONDS=10",
    ]
    for key in required_keys:
        assert key in text


def test_readme_contains_required_alignment_sections():
    text = (ROOT / "docs" / "INTERNAL.md").read_text()
    required_phrases = [
        "Tool-aware Fusion Agent Gateway",
        "Codex 走 `/v1/responses`",
        "Claude Code 走 `/v1/messages`",
        "Hermes Desktop 走 `/v1/chat/completions`",
        "ds4 只作 heavy brain",
        "tool_result / function_call_output 不得丟棄",
        "streaming 必須 keepalive",
        "未閉合 `<think>` 視為模型輸出失敗",
        "M5 Max static fit check 包含 KV cache reserve",
        "長 context 需要 Ollama Modelfile 設 num_ctx",
        "2TB SSD 不要下載收藏型模型",
        "Vision Pro 契約是 vision/tools，不要求 thinking",
        "LM Studio model ids from ~/.lmstudio/hub/models",
        "qwable-agentic-mlx",
        "qwable-formatter-mlx",
        "LM Studio profiles",
        "vision-heavy 保持 two-stage",
        "streaming v1 是 keepalive + final/event streaming",
        "Claude Code / Codex gateway 細節可能變動",
        "Manual End-to-End 驗收",
    ]
    for phrase in required_phrases:
        assert phrase in text


def test_helper_scripts_are_present_and_configurable():
    scripts = {
        "pull_models.sh",
        "setup_ds4.sh",
        "start_ds4.sh",
        "start_server.sh",
        "test_ds4.sh",
        "test_ollama.sh",
        "test_ollama_tools.sh",
        "warmup_models.sh",
        "pull_vision_models.sh",
        "test_vision_fast.sh",
        "test_vision_pro.sh",
        "test_agentic_pro.sh",
        "pull_mlx_models.sh",
        "test_agentic_mlx.sh",
        "test_formatter_mlx.sh",
        "benchmark_m5.sh",
    }
    for script in scripts:
        path = ROOT / "scripts" / script
        assert path.exists(), script
        text = path.read_text()
        assert "set -euo pipefail" in text

    start_server = (ROOT / "scripts" / "start_server.sh").read_text()
    assert 'SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"' in start_server
    assert "--host ${QWABLE_HOST:-127.0.0.1}" in start_server
    assert "--port ${QWABLE_PORT:-8088}" in start_server

    start_ds4 = (ROOT / "scripts" / "start_ds4.sh").read_text()
    assert 'DS4_DIR="${DS4_DIR:-$HOME/Documents/ds4}"' in start_ds4
    assert "--ctx ${DS4_CTX:-100000}" in start_ds4
    assert "--kv-disk-dir" in start_ds4

    pull_models = (ROOT / "scripts" / "pull_models.sh").read_text()
    warmup_models = (ROOT / "scripts" / "warmup_models.sh").read_text()
    legacy_formatter = "gpt" + "-oss:20b"
    assert legacy_formatter not in pull_models
    assert legacy_formatter not in warmup_models
    assert "qwen/qwen3-vl-30b" in pull_models
    assert "qwen/qwen3.6-35b-a3b" in pull_models

    pull_mlx_models = (ROOT / "scripts" / "pull_mlx_models.sh").read_text()
    test_agentic_mlx = (ROOT / "scripts" / "test_agentic_mlx.sh").read_text()
    test_formatter_mlx = (ROOT / "scripts" / "test_formatter_mlx.sh").read_text()
    assert "google/gemma-4-26b-a4b-qat" in pull_mlx_models
    assert "qwen/qwen3.6-35b-a3b" in pull_mlx_models
    assert "MODEL_AGENTIC_MLX" in test_agentic_mlx
    assert "MODEL_FORMATTER_MLX" in test_formatter_mlx

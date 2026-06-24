# Qwable v1.5

Qwable 是本機 **Tool-aware Fusion Agent Gateway**，用一個 8088 gateway 統一 Codex、Claude Code、Hermes Desktop 三種 client，後端接 Ollama 與 ds4。它不是模型訓練、權重融合、MergeKit，也不是 LangChain/LlamaIndex 類 agent framework。

## 核心定位

- Codex 走 `/v1/responses`
- Claude Code 走 `/v1/messages`
- Hermes Desktop 走 `/v1/chat/completions`
- ds4 只作 heavy brain，不是 gateway 主入口
- Fusion 支援 tool_call / tool_use
- tool_result / function_call_output 不得丟棄
- streaming 必須 keepalive
- DeepSeek-R1 的 `<think>` 只供內部使用
- 未閉合 `<think>` 視為模型輸出失敗
- M5 Max static fit check 包含 KV cache reserve
- 長 context 需要 Ollama Modelfile 設 num_ctx
- 2TB SSD 不要下載收藏型模型

## 支援協議

| 協議 | Client | Endpoint |
| --- | --- | --- |
| OpenAI Responses API | Codex / OpenAI SDK | `POST /v1/responses` |
| Anthropic Messages API | Claude Code / Anthropic SDK | `POST /v1/messages` |
| OpenAI Chat Completions | Hermes Desktop / generic OpenAI clients | `POST /v1/chat/completions` |

Health 與 discovery:

```bash
curl http://127.0.0.1:8088/health
curl http://127.0.0.1:8088/v1/models
curl http://127.0.0.1:8088/v1/health
```

## Agent Profiles

| Profile | Model path | Intended use |
| --- | --- | --- |
| `chat-agent` | `MODEL_CODER` | Hermes Desktop 普通聊天 |
| `fast-agent` | `MODEL_FAST` | 日常 coding / tool loop |
| `full-agent` | coder -> tooler -> critic -> judge | 高價值任務、收尾審查 |
| `heavy-agent` | ds4 primary + local checker/critic/judge | 大型 repo、長上下文、重型分析 |
| `fusion-agent` | preset panel -> judge (NEW in G10) | OpenRouter-style multi-model deliberation |

Model aliases:

```text
Codex:
  qwable, qwable-fast, qwable-full, qwable-heavy
  qwable-fusion  # G10: deliberation router

Claude Code:
  claude-qwable, claude-qwable-fast, claude-qwable-full, claude-qwable-heavy
  claude-qwable-fusion  # G10: deliberation router

Hermes Desktop:
  qwable-chat, qwable-fast, qwable-full, qwable-heavy
  qwable-fusion  # G10: deliberation router
```

## Fusion Deliberation Mode (Gate G10)

`qwable-fusion` / `claude-qwable-fusion` 觸發 **OpenRouter-style multi-model deliberation router**：

```
使用者 prompt
  ↓  (從 request body 抽 fusion.preset 或 plugins[].preset)
preset 解析 → 取得 analysis_models[] + judge_model
  ↓
for each analysis_model:    # serial，不是 parallel
  load → chat → unload
  ↓
judge model 收所有 panelist 回應 + 原始 prompt
  ↓
合成 5 段 markdown 結構 (Final Answer / Consensus / Contradictions / Blind Spots / Per-model Notes)
  ↓
回傳 assistant content
```

### 4 個 preset

| Preset | analysis_models | judge | Peak RAM |
|---|---|---|---|
| `quality` | qwen3-coder-next, qwen3.6-35b-a3b, deepseek-r1-distill-qwen-32b | qwen3.6-35b-a3b | ~66GB |
| `budget` | gemma-4-26b-a4b-qat, qwen3.6-35b-a3b | qwen3.6-35b-a3b | ~38GB |
| `coding` | qwen3-coder-next, qwen3.6-35b-a3b, deepseek-r1-distill-qwen-32b | qwen3-coder-next | ~66GB |
| `heavy` | qwen3-coder-next, deepseek-r1-distill-qwen-32b | ds4 deepseek-v4-flash | ~66GB + ds4 |

預設用 `FUSION_DEFAULT_PRESET`（預設 `quality`）。

### Request body — 兩種 shape 都支援

**OpenRouter-style plugins**：
```jsonc
{
  "model": "qwable-fusion",
  "messages": [{"role":"user","content":"..."}],
  "plugins": [{"id":"fusion","preset":"quality"}]
}
```

**簡化 top-level fusion block**（含 custom panel override）：
```jsonc
{
  "model": "qwable-fusion",
  "messages": [{"role":"user","content":"..."}],
  "fusion": {
    "preset": "budget",                                  // optional
    "analysis_models": ["m1", "m2", "m3"],               // optional override
    "judge_model": "qwen/qwen3.6-35b-a3b"                // optional override
  }
}
```

plugins 優先於 top-level fusion。

### 3 個協議的 curl 範例

```bash
# OpenAI Chat + plugins shape
curl -fsS http://127.0.0.1:8088/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwable-fusion",
    "messages": [{"role":"user","content":"Compare mergesort vs quicksort"}],
    "plugins": [{"id":"fusion","preset":"budget"}]
  }'

# OpenAI Responses + custom panel
curl -fsS http://127.0.0.1:8088/v1/responses \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwable-fusion",
    "input": "Review this function",
    "fusion": {
      "analysis_models": ["qwen/qwen3-coder-next", "qwen/qwen3.6-35b-a3b"],
      "judge_model": "qwen/qwen3.6-35b-a3b"
    }
  }'

# Anthropic Messages
curl -fsS http://127.0.0.1:8088/v1/messages \
  -H "Content-Type: application/json" \
  -d '{
    "model": "claude-qwable-fusion",
    "messages": [{"role":"user","content":"..."}],
    "max_tokens": 2048,
    "fusion": {"preset":"coding"}
  }'
```

### Judge 輸出格式（5 段 markdown）

```markdown
## Final Answer
<1-3 句的綜合答案>

## Consensus
- ≥2 位 panelist 同意的重點

## Contradictions
- panelist 矛盾點 (or "- None")

## Blind Spots
- 應質疑的盲點 (or "- None")

## Per-model Notes
### <panelist model id>
<該 model 的 1-2 句摘要>
```

**注意**：reasoning model（如 qwen3.6）不一定會嚴格遵守 markdown 格式 — 此時 runner 走 fallback 路徑，回傳 raw judge text 並標記 `confidence=0.5`、`rationale_summary="fusion_deliberation_fallback_used"`。這是 by design 而非 bug。

### E2E scripts

```bash
bash scripts/warmup_fusion_quality.sh     # preload panel models
bash scripts/test_fusion_quality.sh       # 3 大模型 panel + qwen3.6 judge
bash scripts/test_fusion_budget.sh        # gemma + qwen3.6, qwen3.6 judge
bash scripts/test_fusion_coding.sh        # 3 大模型, coder judge
bash scripts/test_fusion_heavy.sh         # coder + r1 panel, ds4 judge
bash scripts/test_fusion_custom.sh        # custom panel via top-level fusion
bash scripts/test_fusion_bad_preset.sh    # 4xx-style error path
```

Memory invariant：每次 panel 跑完都 `lms unload --all`，確保同時間只有 1 個 model resident，peak RAM 不超過單一最大模型。

### Python 內部呼叫（不透過 HTTP）

```python
from qwable.fusion_presets import PRESETS, resolve_preset
from qwable.fusion_schemas import FusionRequest

# 列出所有 preset
for name, p in PRESETS.items():
    print(name, p.analysis_models, "->", p.judge_model)

# 解析 preset（含 custom panel）
req = FusionRequest(preset="coding")
preset = resolve_preset(req)
print(preset.analysis_models, preset.judge_model)

req = FusionRequest(analysis_models=["m1", "m2"], judge_model="m-judge")
preset = resolve_preset(req)  # 自動用 preset="custom"
print(preset.name)  # "custom"
```

## MLX Formatter Auto-Routing (Gate G11)

`fast-agent` profile 在符合條件時**自動走 MLX 路徑**（gemma 16GB）而非常規 qwen3.6（38GB），省下 ~30s 的 model load 時間。

**條件（必須全部符合）**：

1. `QWABLE_PREFER_MLX_FORMATTER=true`（預設）
2. `tools == []`（沒有 tool calls）
3. `len(text.strip()) < QWABLE_MLX_FORMATTER_MAX_CHARS`（預設 1000 字）

**典型會走 MLX formatter**：

```text
✓ "幫我把這段總結成一句話"
✓ "Translate to Traditional Chinese"
✓ "Format this JSON"
✓ "What's the capital of France?"
```

**會 bypass MLX formatter（用標準 fast-agent）**：

```text
✗ 有 tool calls 的請求（read_file, run_shell ...）
✗ 長 prompt（>1000 字，例如貼一大段 code 要 review）
✗ QWABLE_PREFER_MLX_FORMATTER=false
```

**E2E 觀察 trace**：

```bash
curl -fsS http://127.0.0.1:8088/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwable-fast",
    "messages": [{"role":"user","content":"summarize this in 1 sentence"}]
  }' | python3 -c "
import sys, json
d = json.load(sys.stdin)
# 看 trace 得知走了哪條路徑
print('content:', d['choices'][0]['message']['content'])
"
```

Gateway log 會顯示：

```text
[INFO] profile=formatter-mlx model=google/gemma-4-26b-a4b-qat   # 走 MLX
[INFO] profile=fast-agent   model=qwen/qwen3.6-35b-a3b            # 標準路徑
```

**為什麼這樣設計**：gemma via MLX 載入 ~5s，qwen3.6 載入 ~30s。對於短問答、明顯省時間，且 MLX 變體對中文/英文都已經 fine-tune 過，品質差異不大。

## Streaming Fusion Deliberation (Gate G11-3)

`qwable-fusion` / `claude-qwable-fusion` 加上 `stream: true` 會啟用 SSE 串流：

```bash
curl -sN http://127.0.0.1:8088/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwable-fusion",
    "messages": [{"role":"user","content":"Compare mergesort and quicksort"}],
    "fusion": {"preset":"budget"},
    "stream": true,
    "max_tokens": 4000
  }'
```

**會收到的事件類型**：

1. **OpenAI 標準 chunk**（caller 看得到）：
   ```text
   data: {"choices":[{"delta":{"role":"assistant"}}]}
   data: {"choices":[{"delta":{"content":"\n\n"}}]}
   data: {"choices":[{"delta":{"content":"##"}}]}
   data: {"choices":[{"delta":{"content":" Final"}}]}
   ...
   data: {"choices":[{"delta":{},"finish_reason":"stop"}]}
   data: [DONE]
   ```
   每個 `delta.content` 是 judge model 的一次 token 輸出。

2. **Fusion 內部事件**（OpenAI client 會忽略 `:` 開頭的 SSE comment，但 debug log 有用）：
   ```text
   data: : fusion preset resolved: budget
   data: : fusion panel_start: {'model_id': 'gemma-4-26b-a4b-qat', 'index': 0}
   data: : fusion panel_done: {'model_id': 'gemma-4-26b-a4b-qat', 'latency_ms': 11237, ...}
   data: : fusion panel_start: {'model_id': 'qwen3.6-35b-a3b', 'index': 1}
   data: : fusion panel_done: ...
   data: : fusion judge_start: {'judge_model': 'qwen3.6-35b-a3b', 'judge_backend': 'ollama'}
   ```

3. **Final chunk**（OpenAI 標準 end）：
   ```text
   data: {"choices":[{"delta":{},"finish_reason":"stop"}]}
   data: [DONE]
   ```

**為什麼這個設計**：之前 streaming caller 看到的就是 keepalive + 整段 final chunk（沒有 token-by-token 進度）。現在 caller 看得到：
- 哪個 panel model 跑完了（comment）
- judge 開始（comment）
- judge token 串流（real OpenAI deltas）
- 結束（standard OpenAI stop + DONE）

實作：`OllamaClient.chat_completion_stream()` 用 httpx SSE iterator + `asyncio.run_in_executor` 保持 event loop 不卡。

## Install

```bash
python3 -m venv .venv
./.venv/bin/pip install -r requirements.txt
cp .env.example .env
```

Start gateway:

```bash
bash scripts/start_server.sh
```

The server binds to `127.0.0.1:8088` by default. Do not bind to `0.0.0.0` unless you intentionally expose the gateway.

## Environment

See [.env.example](../.env.example). Important defaults:

```bash
QWABLE_HOST=127.0.0.1
QWABLE_PORT=8088
LOCAL_MODEL_BACKEND=lmstudio
OLLAMA_BASE_URL=http://127.0.0.1:1234/v1
LMSTUDIO_CLI_PATH=/Users/yourname/.lmstudio/bin/lms
DS4_BASE_URL=http://127.0.0.1:8000/v1
MODEL_FAST=google/gemma-4-26b-a4b-qat
MODEL_CODER=qwen/qwen3-coder-next
MODEL_TOOLER=qwen/qwen3-coder-next
MODEL_CRITIC=deepseek-r1-distill-qwen-32b
MODEL_JUDGE=deepseek-r1-distill-qwen-32b
MODEL_FORMATTER=google/gemma-4-26b-a4b-qat
MODEL_VISION_FAST=google/gemma-4-26b-a4b-qat
MODEL_VISION_PRO=qwen/qwen3-vl-30b
MODEL_AGENTIC_PRO=qwen/qwen3.6-35b-a3b
MODEL_HERMES_PRO=qwen/qwen3.6-35b-a3b
MODEL_AGENTIC_MLX=qwen/qwen3.6-35b-a3b
MODEL_FORMATTER_MLX=google/gemma-4-26b-a4b-qat
MODEL_HEAVY=deepseek-v4-flash
```

v1.5 uses model specialization, not replacement. Gate G09 routes local non-ds4 profiles through **LM Studio model ids from ~/.lmstudio/hub/models** instead of Ollama tags:

```text
google/gemma-4-26b-a4b-qat -> MODEL_FAST / MODEL_FORMATTER / MODEL_VISION_FAST / MODEL_FORMATTER_MLX
qwen/qwen3-coder-next -> MODEL_CODER / MODEL_TOOLER
qwen/qwen3-vl-30b -> MODEL_VISION_PRO for OCR/UI/visual-coding evidence
qwen/qwen3.6-35b-a3b -> MODEL_AGENTIC_PRO / MODEL_HERMES_PRO / MODEL_AGENTIC_MLX
```

LM Studio profiles are still serialized by Qwable's global request lock. The compatibility client uses LM Studio's OpenAI-compatible `/v1/chat/completions`; for vision, it translates the previous Ollama-native image shape into OpenAI multimodal `image_url` content. The existing aliases remain available: `qwable-agentic-mlx` maps to `MODEL_AGENTIC_MLX`, and `qwable-formatter-mlx` maps to `MODEL_FORMATTER_MLX`.

Vision Pro 契約是 vision/tools，不要求 thinking. vision-heavy 保持 two-stage: first extract auditable `VisionEvidence`, unload resident LM Studio/Ollama local models, then run ds4 heavy reasoning. The gateway is configured for serial execution on M5 Max 128GB; do not keep qwen-vl, qwen3.6, qwen-coder-next, deepseek-r1, and ds4 resident in parallel.

## Model Storage Policy

LM Studio stores model manifests under:

```text
/Users/yourname/.lmstudio/hub/models
```

Verify the required local models:

```bash
bash scripts/pull_models.sh
bash scripts/pull_mlx_models.sh
bash scripts/pull_vision_models.sh
```

The helper scripts now check LM Studio model availability via `~/.lmstudio/bin/lms`; they do not call `ollama pull`. Avoid duplicate quantizations and collection downloads such as 100B+ archive models, multiple Q2/Q3/Q4/Q5 copies, and large MoE models not needed by this gateway.

## ds4

This machine currently uses:

```bash
DS4_DIR=$HOME/Documents/ds4
DS4_KV_DIR=$HOME/Documents/ds4-kv
```

Setup or rebuild:

```bash
bash scripts/setup_ds4.sh
```

Start ds4:

```bash
bash scripts/start_ds4.sh
```

Smoke:

```bash
bash scripts/test_ds4.sh
```

The ds4 server should expose:

```text
http://127.0.0.1:8000/v1
```

heavy-agent uses ds4 as primary only when ds4 health is OK. If ds4 is offline, times out, or returns a bad response, heavy-agent falls back to full-agent and records debug trace when requested.

## Context / num_ctx

fast-agent is not intended for very large contexts. For longer Ollama context, create a derived model:

```text
FROM qwen3-coder:30b
PARAMETER num_ctx 65536
```

Create it:

```bash
ollama create qwen3-coder-30b-64k -f Modelfile.qwen3-coder-30b-64k
```

Then set:

```bash
MODEL_FAST=qwen3-coder-30b-64k
MODEL_CODER=qwen3-coder-30b-64k
```

Profile input limits:

```text
FAST_MAX_INPUT_CHARS=24000
FULL_MAX_INPUT_CHARS=96000
HEAVY_MAX_INPUT_CHARS=160000
```

Over-limit requests return an explicit compact/use-heavier-profile message instead of silently truncating.

## Client Settings

### Codex

`~/.codex/config.toml`:

```toml
model = "qwable-fast"
model_provider = "qwable"

[model_providers.qwable]
name = "Qwable"
base_url = "http://127.0.0.1:8088/v1"
wire_api = "responses"
env_key = "QWABLE_API_KEY"
stream_idle_timeout_ms = 1000000
```

Shell:

```bash
export QWABLE_API_KEY=local
```

Use `qwable-full` for high-value review tasks and `qwable-heavy` for ds4-backed long-context work.

### Claude Code

```bash
export ANTHROPIC_BASE_URL=http://127.0.0.1:8088
export ANTHROPIC_AUTH_TOKEN=local
export CLAUDE_CODE_ENABLE_GATEWAY_MODEL_DISCOVERY=1
```

Models:

```text
claude-qwable-fast
claude-qwable-full
claude-qwable-heavy
```

### Hermes Desktop

```text
Provider: Custom / OpenAI-compatible
Base URL: http://127.0.0.1:8088/v1
API Key: local
Model: qwable-chat
Context length: 24000
```

Use `qwable-fast` for coding-agent style tasks, `qwable-full` for review, and `qwable-heavy` for ds4-backed heavy work. Use `qwable-vision-pro` for direct screenshot/OCR evidence, `qwable-agentic-pro` for qwen3.6 coding/thinking agent work, and `qwable-hermes-pro` for Hermes Desktop pro multimodal chats.

Optional MLX aliases are available for explicit testing only: `qwable-agentic-mlx` and `qwable-formatter-mlx`.

## Scripts

```bash
bash scripts/pull_models.sh
bash scripts/pull_vision_models.sh
bash scripts/test_ollama.sh
bash scripts/test_ollama_tools.sh
bash scripts/test_vision_fast.sh
bash scripts/test_vision_pro.sh
bash scripts/test_agentic_pro.sh
bash scripts/pull_mlx_models.sh
bash scripts/test_agentic_mlx.sh
bash scripts/test_formatter_mlx.sh
bash scripts/setup_ds4.sh
bash scripts/start_ds4.sh
bash scripts/test_ds4.sh
bash scripts/warmup_models.sh
bash scripts/benchmark_m5.sh
bash scripts/start_server.sh
```

All scripts are env-driven. Useful overrides:

```bash
QWABLE_HOST=127.0.0.1
QWABLE_PORT=8088
OLLAMA_BASE_URL=http://127.0.0.1:11434/v1
DS4_BASE_URL=http://127.0.0.1:8000/v1
DS4_DIR=$HOME/Documents/ds4
DS4_KV_DIR=$HOME/Documents/ds4-kv
DS4_CTX=100000
DS4_KV_DISK_SPACE_MB=32768
```

## Manual End-to-End 驗收

Start Ollama, ds4, then the gateway. Run these checks before marking the system operational.

Codex final answer:

```bash
curl http://127.0.0.1:8088/v1/responses \
  -H "Content-Type: application/json" \
  -d '{"model":"qwable-fast","input":"用一句話說明 Qwable","stream":false}'
```

Codex function_call:

```bash
curl http://127.0.0.1:8088/v1/responses \
  -H "Content-Type: application/json" \
  -d '{"model":"qwable-fast","input":"請讀取 README.md","tools":[{"type":"function","name":"read_file","description":"Read a local file","parameters":{"type":"object","properties":{"path":{"type":"string"}},"required":["path"]}}],"stream":false}'
```

Claude Code tool_use:

```bash
curl http://127.0.0.1:8088/v1/messages \
  -H "Content-Type: application/json" \
  -H "x-api-key: local" \
  -H "anthropic-version: 2023-06-01" \
  -d '{"model":"claude-qwable-fast","max_tokens":1024,"tools":[{"name":"Bash","description":"Run a shell command","input_schema":{"type":"object","properties":{"command":{"type":"string"}},"required":["command"]}}],"messages":[{"role":"user","content":"請列出目前目錄"}],"stream":false}'
```

Hermes/OpenAI chat:

```bash
curl http://127.0.0.1:8088/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer local" \
  -d '{"model":"qwable-chat","messages":[{"role":"user","content":"用三點說明你是什麼。"}],"stream":false}'
```

heavy-agent debug:

```bash
curl http://127.0.0.1:8088/v1/responses \
  -H "Content-Type: application/json" \
  -d '{"model":"qwable-heavy","input":"設計如何分析整個 codebase 的依賴與風險。","metadata":{"debug":true},"stream":false}'
```

G05 Manual E2E acceptance checklist:

```text
[x] Codex final answer
[x] Codex function_call
[x] Codex function_call_output continuation
[x] Claude Code final answer
[x] Claude Code tool_use
[x] Claude Code tool_result continuation
[x] Hermes Desktop chat completion
[x] Hermes Desktop tool_calls compatibility
[x] heavy-agent ds4 online
[x] heavy-agent ds4 fallback
[x] stream keepalive
[x] global request lock
```

## Testing

Unit and endpoint tests mock Ollama/ds4:

```bash
./.venv/bin/pytest tests/ -q
```

Latest verified local result:

```text
2026-06-17 G06-H:
pytest tests/ -q -> 119 passed, 1 warning
route surface -> routes=7 GET=3 POST=4
optional MLX smoke -> formatter-mlx PASS, agentic-mlx tool_calls PASS
optional MLX endpoint contract -> Responses/Chat/Messages aliases PASS
optional MLX live gateway smoke -> Responses/Chat/Anthropic aliases PASS
handoff package refresh -> README/HANDOFF aligned for v1.5
```

## Known Limits

- streaming v1 是 keepalive + final/event streaming，不是底層 token 真 streaming
- 每輪只允許一個 tool_call
- 不支援 parallel requests
- server stateless，不儲存 `previous_response_id`
- function/object schema 優先；複雜 multimodal tool input 標記 unsupported
- 本地模型不保證永遠選對工具；用 schema validation 降風險
- ds4 是 beta 級 heavy backend，必須保留 fallback
- heavy-agent 不適合高頻工具 loop
- Claude Code / Codex gateway 細節可能變動，README 必須記錄驗收版本

## Current Gate Notes

G06-H refreshed the package and handoff docs for v1.5. G06-G verified optional MLX aliases through the live gateway without changing default routing:

```text
qwable-agentic-mlx -> MODEL_AGENTIC_MLX
qwable-formatter-mlx -> MODEL_FORMATTER_MLX
claude-qwable-agentic-mlx -> MODEL_AGENTIC_MLX
claude-qwable-formatter-mlx -> MODEL_FORMATTER_MLX
```

The optional MLX aliases are explicit test/pro profiles only. `qwable-fast`, `qwable`, `qwable-chat`, and the standard Claude aliases keep the non-MLX default model plan.

Re-run the Manual End-to-End checks after changing model names, ds4 build/runtime flags, streaming behavior, or client protocol adapters. Rebuild the handoff archive after any source, test, script, README, HANDOFF, or `.env.example` change.

<p align="center">
  <img src="assets/qwable.jpg" alt="Qwable" width="200">
</p>

# Qwable

[English](README.md) · [繁體中文](README.zh-TW.md) · [英文安裝指南](INSTALL.en.md) · [中文安裝指南](INSTALL.md)

**讓 Claude Code 和 Codex 同時指向你自己的 Mac——本地分工模型、會自我交叉檢查的多模型 council、以及 Qwable 優先的 coding agent runtime。128GB、全程離線、每月 $0。**

Qwable 是一個本機 gateway（`127.0.0.1:8088`），同時講三種 agent 協議——OpenAI **Responses**（Codex）、Anthropic **Messages**（Claude Code）、OpenAI **Chat Completions**（任何 OpenAI 相容 client）。這個單一 endpoint 背後是一整批分工的本地模型，外加一個多模型 **deliberation** 模式，讓一台 Apple Silicon 機器表現得像一個有專家小組的雲端 router。

> 大多數「本地 LLM」方案丟給你一個擅長文字、其他都普通的模型。Qwable 反過來做：把每個請求路由到對的專家（coder、reasoner、vision、fast-formatter），你需要時還能召集好幾個專家組成 panel，由一個 judge model 統整。你拿到的是雲端等級的編排，但沒有雲端。

---

## 為什麼要做這個

單一本地模型很窄。一個 checkpoint 擅長聊天，另一個只會 code；reasoning 跟 vision 通常各需要專屬權重。實務上你最後得同時養五個模型，還要手動決定該找哪一個。

雲端廠商把這一切藏在一個聰明的 endpoint 後面。Qwable 把同樣的體驗帶到你自己的機器上：

- **一個 endpoint，沿用你已經在用的 agent。** Claude Code、Codex、OpenAI-SDK client 都零改 code 接上——只換 base URL。
- **專業化，而非取代。** 每個工作路由到為它而生的模型，而不是逼一個通才硬做全部。
- **單機就能開 council。** deliberation router 以 **serial** 方式跑 panel（load → 回答 → unload），所以「5 個模型互審」可以塞進一台 128GB 的 Mac，而 RAM 峰值永遠不超過單一最大模型。

如果你曾經想要「Claude Code，但 100% 本地，而且內建第二、第三意見」——這就是它的全部。

---

## 你會得到什麼

| 能力 | 意思 |
| --- | --- |
| 🔌 **三協議 gateway** | `/v1/responses`（Codex）、`/v1/messages`（Claude Code）、`/v1/chat/completions`（OpenAI client）——同一個 port |
| 🧠 **任務感知路由** | 每個請求送到對的專家：coder、tool-runner、critic、judge、vision，或 fast formatter |
| 🤝 **Fusion deliberation** | OpenRouter 式多模型 panel → judge 統整 → `Final Answer / Consensus / Contradictions / Blind Spots / Per-model Notes` |
| 👁️ **Vision pipeline** | 兩階段：先在本地 VLM 抽出可稽核的視覺證據，再交給重型 reasoner |
| 🪶 **MLX 快路徑** | 短、無 tool 的 prompt 自動走輕量 MLX 模型——省掉約 30 秒的 model load |
| 📡 **Streaming** | deliberation 走 SSE 串流，附每個 panel 的進度事件 |
| 🧰 **Tool 感知** | 三種協議都處理並驗證 `tool_call` / `tool_use` |
| 🖥️ **即時 dashboard** | 單檔 web UI，即時串流 panel/judge 事件 |
| 🐍 **Python SDK** | 不走 HTTP，直接 in-process 呼叫 gateway |

---

## 架構

```
                 Claude Code        Codex         任何 OpenAI client
                     │                │                   │
              /v1/messages      /v1/responses     /v1/chat/completions
                     └────────────────┼───────────────────┘
                                      ▼
                       ┌──────────────────────────────┐
                       │   Qwable Gateway :8088    │
                       │   • protocol adapters         │
                       │   • task-aware router         │
                       │   • global serial lock        │
                       │   • fusion deliberation runner│
                       └───────────────┬───────────────┘
                          ┌────────────┴─────────────┐
                          ▼                           ▼
                ┌───────────────────┐        ┌──────────────────┐
                │  LM Studio :1234  │        │     ds4 :8000    │
                │  (OpenAI-compat)  │        │  重型 reasoner   │
                │  coder / vision / │        │ （長上下文、     │
                │  reasoner / fast  │        │   大型 repo）     │
                └───────────────────┘        └──────────────────┘
```

一切都是 **serial by design**。一個全域 request lock 保證同一時間只有一個模型 resident，記憶體峰值因此有界——代價是不支援平行請求，這在單一工作站上是對的取捨。

---

## Quickstart

**需求：** 大容量 unified memory 的 Apple Silicon（在 **M5 Max, 128GB** 上開發與調校）、[LM Studio](https://lmstudio.ai) 作為本地後端、Python 3.11+。選用的 `ds4` 重型後端只有在長上下文／大型 repo 時才需要。

```bash
git clone https://github.com/HenryLinyy/qwable.git
cd qwable

python3 -m venv .venv
./.venv/bin/pip install -r requirements.txt
cp .env.example .env          # 然後依你的機器改路徑/模型

bash scripts/start_server.sh  # 綁定 127.0.0.1:8088
```

確認它活著：

```bash
curl http://127.0.0.1:8088/health
curl http://127.0.0.1:8088/v1/models
```

> 安全提醒：gateway 刻意綁 `127.0.0.1`。除非你有意對外暴露，否則不要綁 `0.0.0.0`。

---

## 接上你的 agent

### Claude Code

```bash
export ANTHROPIC_BASE_URL=http://127.0.0.1:8088
export ANTHROPIC_AUTH_TOKEN=local
export CLAUDE_CODE_ENABLE_GATEWAY_MODEL_DISCOVERY=1
# 模型：claude-qwable-fast | -full | -heavy | -fusion
```

### Codex

```toml
# ~/.codex/config.toml
model = "qwable-fast"
model_provider = "qwable"

[model_providers.qwable]
name = "Qwable"
base_url = "http://127.0.0.1:8088/v1"
wire_api = "responses"
env_key = "QWABLE_API_KEY"
```

### 任何 OpenAI 相容 client

```
Base URL: http://127.0.0.1:8088/v1
API Key:  local
Model:    qwable-chat
```

---

## 重點功能：fusion deliberation（council）

這是值得這個 repo 的功能。呼叫 `*-fusion` 模型，Qwable 會跑一個 **模型 panel**，再由一個 **judge** 把它們的答案統整成結構化判斷：

```bash
curl -fsS http://127.0.0.1:8088/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwable-fusion",
    "messages": [{"role":"user","content":"Compare mergesort vs quicksort for our workload"}],
    "plugins": [{"id":"fusion","preset":"quality"}]
  }'
```

judge 回傳五個區塊：

```markdown
## Final Answer        ← 統整後 1–3 句的判斷
## Consensus           ← ≥2 位 panelist 同意的點
## Contradictions      ← panelist 互相矛盾的地方
## Blind Spots         ← 大家可能都漏掉的盲點
## Per-model Notes     ← 每個 panelist 的一句話觀點
```

四個 preset，每個都讓 **RAM 峰值不超過單一最大模型**（serial load/unload）：

| Preset | Panel | Judge | RAM 峰值 |
| --- | --- | --- | --- |
| `quality` | coder + agentic + reasoner | agentic | ~66GB |
| `budget` | fast + agentic | agentic | ~38GB |
| `coding` | coder + agentic + reasoner | coder | ~66GB |
| `heavy` | coder + reasoner | ds4 heavy | ~66GB + ds4 |

也可以 inline 覆寫整個 panel：

```jsonc
{
  "model": "qwable-fusion",
  "messages": [{"role":"user","content":"Review this function"}],
  "fusion": {
    "analysis_models": ["qwen/qwen3-coder-next", "qwen/qwen3.6-35b-a3b"],
    "judge_model": "qwen/qwen3.6-35b-a3b"
  }
}
```

加上 `"stream": true`，就能看到每個 panelist 完成、judge 的 token 透過 SSE 即時抵達。

---

## Agent runtime（v1.8）

除了單次回答與 council，Qwable 還內建多步驟 **agent runtime**——planner → executor → repair → critic → judge 的迴圈，支援 tool call、context 壓縮、repo index，並把可重播的 run trace 存進 SQLite。三個 profile 驅動它：

| 模型 | 用途 |
| --- | --- |
| `qwable-agent` | 通用長任務 agent——規劃、多步工具流、整理研究 |
| `qwable-code-agent` | Coding／repo patch／測試／修復 workflow |
| `qwable-review-agent` | 審查 plan／patch／架構／風險，不主動大改 |

```bash
curl http://127.0.0.1:8088/v1/responses \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwable-code-agent","input":"幫 fetch helper 加上重試，並用測試驗證。"}'
```

每個階段會解析一個模型 **role**，各自帶 fallback chain（`.env` 的 `MODEL_ROLE_*`），缺模型時優雅降級而不是整個 run 失敗。兩個 Fable/Mythos 風格的本地 worker 可加入迴圈：**Qwable**（coding executor／repair，預設開）與 **Qwythos**（長 context worker，需手動開）。兩者都在 `.env` 切換。

---

## 模型專業化

v1.8 把每個 profile 路由到為該工作而生的模型（下面是 LM Studio model ID；在 `.env` 換成你自己的）：

| 角色 | 模型 | 約略 RAM |
| --- | --- | --- |
| Fast / formatter / fast-vision | `gemma-4-26b-a4b-qat` | ~16GB |
| Coder / tool-runner | `qwen3-coder-next` | ~65GB |
| Vision（OCR / UI / visual-coding） | `qwen3-vl-30b` | ~34GB |
| Agentic reasoning / pro chat | `qwen3.6-35b-a3b` | ~38GB |
| Critic / judge | `deepseek-r1-distill-qwen-32b` | ~66GB |
| Heavy 後端（ds4） | `deepseek-v4-flash` | ~90GB |

**Profiles：** `chat-agent`（純聊天）、`fast-agent`（日常 coding／tool loop）、`full-agent`（coder → tooler → critic → judge）、`heavy-agent`（ds4 為主、本地 fallback）、`fusion-agent`（council）。

---

## 即時 dashboard

單一自包含的 `qwable/web/dashboard.html` 會即時串流整個 deliberation——panel 開始／完成事件、judge token、最終統整——讓你真的看著 council 思考，而不是盯著轉圈圈。

---

## 誠實的限制

不誇大——這些都是真實且刻意的設計：

- **單一工作站、serial 執行。** 不支援平行請求；同一時間只有一個模型 resident。這是讓 RAM 有界的取捨。
- **每輪一個 tool call。** 對大多數 agent loop 夠用，不適合大量 fan-out 的工具使用。
- **Streaming v1** 是 keepalive + chunk/event 串流，不是每條路徑都做底層 token 串流（fusion judge 那條路徑*有*串 token）。
- **Stateless server**——不儲存 `previous_response_id`。
- **本地模型不保證永遠選對工具。** schema validation 降低風險，但不能完全消除。
- **`ds4` 重型後端是 beta**，且永遠有退回本地 `full-agent` 的 fallback。
- **硬體需求是真的。** 它是為 128GB unified memory 而建。縮小模型計畫後能在更小的機器跑，但預設 preset 假設有餘裕。

---

## 狀態

`v1.8.0` · **521 個測試通過** · macOS CI（pytest + ruff），涵蓋 Python 3.11 / 3.12。

```bash
./.venv/bin/pytest tests/ -q
```

---

## License

採用 [MIT License](LICENSE)。

歡迎貢獻，請參考 [CONTRIBUTING.md](CONTRIBUTING.md)；安全問題請依 [SECURITY.md](SECURITY.md) 私下回報。

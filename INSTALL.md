# Qwable Agent Gateway M5 v1.5 — 安裝教學

> **Qwable** 是一個 OpenRouter-style multi-model deliberation router。
> 將問題丟給 3-4 個本地 MLX model 平行分析，再由 judge model 綜合輸出。
> 在 M5 Max 128GB 上跑得動，LM Studio + MLX backend + ds4 (heavy preset) 完整支援。

## 📋 系統需求

### 硬體
- **Apple Silicon Mac**（M1/M2/M3/M4/M5）— 必需（MLX 框架）
- **RAM 最低 32GB** / 建議 64GB+ / 完整 fusion quality preset 需要 **128GB**
- 磁碟空間 ~5GB（不含 LM Studio model weights，models 另算 250GB+）

### 軟體
- **macOS 13+** (Apple Silicon)
- **LM Studio 0.4.16+** — [lmstudio.ai](https://lmstudio.ai/)
- **Python 3.11 或 3.12** — `python3 --version` 確認
- **Homebrew**（推薦，方便裝 Python 跟跑 launchd）

### 必要 LM Studio Models

| Model | 用途 | 容量 |
|---|---|---|
| `qwen/qwen3.6-35b-a3b` (MLX-8bit) | quality/budget judge + panelist | ~38GB |
| `qwen/qwen3-coder-next` (MLX) | quality/coding panelist | ~65GB |
| `deepseek-r1-distill-qwen-32b` | quality panelist (heavy reasoning) | ~66GB |
| `google/gemma-4-26b-a4b-qat` | budget panelist（light） | ~16GB |
| (選用) `deepseek-v4-flash` via ds4 | heavy preset judge（外部 backend） | 0GB 本地 |

> 沒有的 model 跑 fusion 時會 auto-download（~30s 首次載入）。
> 跑 budget preset 最少只要 `qwen3.6-35b-a3b` + `gemma-4-26b-a4b-qat`（~54GB）。

### 選用：ds4 backend（heavy preset 用）

如果你要跑 `fusion preset=heavy`，需要 ds4 server：

```bash
# ds4 是另外的 repo，這裡不包含
# 啟動 ds4 server 跑在 http://127.0.0.1:8000
# 確認 ds4 跑得起來後，DS4_BASE_URL=http://127.0.0.1:8000/v1
```

沒 ds4 也能跑其他 3 個 preset（quality / budget / coding）。

---

## 🚀 安裝步驟

### Step 0: 確認環境

```bash
# 1. 確認是 Apple Silicon Mac
uname -m   # 應該是 arm64

# 2. 確認 Python 3.11+
python3 --version   # Python 3.11.x 或 3.12.x

# 3. 確認 LM Studio 已裝且在跑
~/.lmstudio/bin/lms --version   # 應該看到 CLI commit hash
~/.lmstudio/bin/lms ps           # 確認 server 在跑
```

如果沒 Python：
```bash
brew install python@3.12
```

如果沒 LM Studio：
```bash
brew install --cask lm-studio
# 開 LM Studio → Developer → Start Server
```

---

### Step 1: 解壓縮

```bash
# 假設壓縮檔在 ~/Downloads
cd ~/Downloads
unzip qwable-agent-gateway-m5-v1.5.zip
cd qwable-agent-gateway-m5
```

---

### Step 2: 建立 Python 虛擬環境

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

第一次約 1-2 分鐘（裝 fastapi / uvicorn / pydantic / pytest 等）。

---

### Step 3: 設定環境變數

```bash
cp .env.example .env
# 編輯 .env（選用 — 預設值就能跑）
nano .env
```

最少需要的 env vars（其他用預設）：

```bash
# LM Studio 路徑
OLLAMA_BASE_URL=http://127.0.0.1:1234/v1

# (選用) ds4 路徑 — 沒裝就跳過
DS4_BASE_URL=http://127.0.0.1:8000/v1

# (選用) Fusion 設定
FUSION_DEFAULT_PRESET=quality
FUSION_MAX_TOKENS_PANEL=1500
FUSION_MAX_TOKENS_JUDGE=3600

# (選用) Warmup launchd — 早上自動預載模型
QWABLE_WARMUP_ON_BOOT=true
```

---

### Step 4: 跑測試（驗證安裝成功）

```bash
source .venv/bin/activate
pytest tests/ -q
```

**預期**：`287 passed`（或更新版數量）— 大約 5-8 秒完成。

如果 fail，檢查：
- `python3 -m pip install -r requirements.txt` 是否成功
- 錯誤訊息通常會指出哪個套件缺
- 詳見下方「故障排除」

---

### Step 5: 啟動 Gateway

#### 方法 A：手動啟動（先試跑一次）

```bash
source .venv/bin/activate
./scripts/start_server.sh
# 或
python3 -m qwable.cli
```

Gateway 會跑在 `http://127.0.0.1:8088`。

#### 方法 B：launchd 自動啟動（推薦長期使用）

```bash
# 1. 編輯 Library/LaunchAgents/io.github.henrylinyy.qwable.gateway.plist
#    把所有 __REPLACE_ME__ 換成你的實際路徑
#    例如:
#      /Users/yourname/qwable-agent-gateway-m5
#      /Users/yourname/.lmstudio/bin/lms
#      /Users/yourname/.local/share/...

# 2. 複製 plist 到 ~/Library/LaunchAgents/
cp Library/LaunchAgents/io.github.henrylinyy.qwable.gateway.plist ~/Library/LaunchAgents/

# 3. 載入（id 是 plist filename 去 .plist）
launchctl load ~/Library/LaunchAgents/io.github.henrylinyy.qwable.gateway.plist

# 4. 確認在跑
launchctl list | grep qwable
# 應該看到:
#   45157  -  io.github.henrylinyy.qwable.gateway
#   57174  0  io.github.henrylinyy.qwable.warmup.quality
#   57179  0  io.github.henrylinyy.qwable.warmup.budget
```

> launchd plist 裡用 `__REPLACE_ME__` 是故意的 — 你的同事路徑跟你不同，
> 不能用同一個 plist 直接 load。**每個人都要改**自己的 username/路徑。

#### 方法 C：選用 — 開機自動 warmup models

```bash
# 裝 2 個 warmup launchd jobs（早 8:00 預載常用 preset 的 model）
./scripts/install_warmup_launchd.sh

# 立即手動跑（不等 8:00）
./scripts/warmup_now.sh quality
./scripts/warmup_now.sh budget
```

---

### Step 6: 驗證 Gateway 跑得起來

```bash
# 健康檢查
curl http://127.0.0.1:8088/health
# 預期: {"status": "ok", "version": "1.5.0", ...}

# 列出 fusion presets
curl http://127.0.0.1:8088/v1/fusion/presets | python3 -m json.tool | head -30

# 跑 budget preset E2E（最快，<40GB peak）
./scripts/test_fusion_budget.sh
# 預期: PASS budget + 看得到 streaming events
```

開瀏覽器到 **`http://127.0.0.1:8088/dashboard`** — 即時看 fusion deliberation 進度。

---

## 🎯 接下來怎麼用

### Python SDK（推薦）

```python
from qwable_sdk import LocalFusionClient, FusionPreset

client = LocalFusionClient("http://127.0.0.1:8088")

# 列出 presets
presets = client.list_presets()
print(presets["presets"]["quality"]["panel"])

# Non-streaming fusion
result = client.fusion_chat(
    messages=[{"role": "user", "content": "Compare mergesort vs quicksort"}],
    preset=FusionPreset.QUALITY,
)
print(result.text)

# Streaming（看 token 慢慢出來）
for event in client.fusion_chat_stream(
    messages=[{"role": "user", "content": "Explain MLX"}],
    preset=FusionPreset.BUDGET,
):
    if event.event == "judge_token":
        print(event.judge.delta, end="", flush=True)
```

### HTTP API（任何 client）

```bash
# OpenAI Chat 相容
curl http://127.0.0.1:8088/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwable-fusion",
    "messages": [{"role":"user","content":"hi"}],
    "fusion": {"preset": "budget"},
    "stream": true,
    "max_tokens": 4000
  }'

# Anthropic Messages 相容
curl http://127.0.0.1:8088/v1/messages \
  -H "Content-Type: application/json" \
  -d '{
    "model": "claude-qwable-fusion",
    "max_tokens": 4000,
    "messages": [{"role":"user","content":"hi"}],
    "fusion": {"preset": "budget"},
    "stream": true
  }'
```

### Web UI Dashboard

開 `http://127.0.0.1:8088/dashboard` 在瀏覽器 — 互動式 streaming 介面。

### Hermes Desktop 整合

把 `http://127.0.0.1:8088/v1` 加到 Hermes 的 provider 清單，model 填 `qwable-fusion` 或 `claude-qwable-fusion`。

---

## 🛠️ 故障排除

### 問題：`pytest` 跑不起來

```bash
# 確認虛擬環境啟動
source .venv/bin/activate
which python3   # 應該是 .venv/bin/python3
which pytest   # 應該是 .venv/bin/pytest

# 重裝依賴
pip install -r requirements.txt --force-reinstall
```

### 問題：Gateway 啟動失敗

```bash
# 看 log
tail -50 ~/Library/Logs/qwable-gateway.err.log

# 常見問題：
# 1. port 8088 被佔用
lsof -i :8088

# 2. LM Studio 沒跑
~/.lmstudio/bin/lms server start
curl http://127.0.0.1:1234/v1/models   # 應該回 200

# 3. Python 找不到 qwable
source .venv/bin/activate
python3 -c "import qwable; print(qwable.__file__)"
```

### 問題：Fusion 跑出 fallback 文字（沒有 5-section markdown）

這是 **正常的** — 某些 reasoning model 跑 fallback 路徑（不遵循 5-section 格式但給出 raw text）。
看 `action.trace["fusion"]["structured_had_fallback"]` 判斷。
要強制 5-section：把 judge 換成 `qwen/qwen3-coder-next`（coding preset）或用 gemma-3n 等更小模型。

### 問題：Gateway 卡住 / hang

```bash
# 檢查是否有 stuck request
ps aux | grep uvicorn

# 看 log 看是不是 sync IO blocking
tail -30 ~/Library/Logs/qwable-gateway.err.log

# 重啟
kill -TERM $(pgrep -f "uvicorn qwable.server")
launchctl kickstart -k "gui/$(id -u)/io.github.henrylinyy.qwable.gateway"
```

### 問題：ds4 timeout

`fusion preset=heavy` 用 ds4。如果 ds4 沒跑或 timeout：
- 確認 `ds4` 跑在 `http://127.0.0.1:8000/v1`
- 設 `DS4_BASE_URL` env 變數
- 或暫時改用其他 preset

### 問題：模型沒載入 / 自動下載失敗

```bash
# 手動下載必要 model
~/.lmstudio/bin/lms get qwen/qwen3.6-35b-a3b@mlx-8bit
~/.lmstudio/bin/lms get google/gemma-4-26b-a4b-qat@mlx-4bit
~/.lmstudio/bin/lms get qwen/qwen3-coder-next@mlx
~/.lmstudio/bin/lms get deepseek-r1-distill-qwen-32b@mlx
```

---

## 📊 常用指令

```bash
# 啟動 / 停止
launchctl start io.github.henrylinyy.qwable.gateway
launchctl stop io.github.henrylinyy.qwable.gateway
launchctl kickstart -k io.github.henrylinyy.qwable.gateway   # 強制重啟

# 看 log
tail -f ~/Library/Logs/qwable-gateway.out.log
tail -f ~/Library/Logs/qwable-gateway.err.log

# 跑測試
cd qwable-agent-gateway-m5
source .venv/bin/activate
pytest tests/ -q              # 完整 suite (~5s)
pytest tests/test_fusion_core.py -v   # 單一檔案

# 手動 warmup models（避免第一次請求的 30s 載入）
./scripts/warmup_now.sh quality   # 預載 quality preset 的 3 個 models

# E2E 測試
./scripts/test_fusion_quality.sh
./scripts/test_fusion_budget.sh
./scripts/test_fusion_coding.sh
./scripts/test_fusion_heavy.sh
./scripts/test_fusion_custom.sh
./scripts/test_fusion_bad_preset.sh

# 持久化對話（multi-turn）
CONV_ID=$(curl -s -X POST http://127.0.0.1:8088/v1/conversations \
  -H "Content-Type: application/json" -d '{}' \
  | python3 -c "import sys, json; print(json.load(sys.stdin)['id'])")
echo "Created: $CONV_ID"
curl -s http://127.0.0.1:8088/v1/conversations/$CONV_ID | python3 -m json.tool
curl -s -X DELETE http://127.0.0.1:8088/v1/conversations/$CONV_ID

# System / optimization introspection
curl http://127.0.0.1:8088/v1/system/optimizations | python3 -m json.tool
```

---

## 📁 檔案結構

```
qwable-agent-gateway-m5/
├── HANDOFF.md                  ← 完整交接文件（最重要的文件）
├── README.md                   ← 公開 README
├── INSTALL.md                  ← 本文件
├── requirements.txt            ← Python 依賴
├── pyproject.toml              ← 專案 metadata
├── .env.example                ← 環境變數範本
├── .gitignore                  ← Git 排除規則
├── qwable/               ← 主程式 source
│   ├── __init__.py
│   ├── server.py               ← FastAPI server (4 個 endpoint + /dashboard)
│   ├── fusion_core.py          ← Profile dispatch
│   ├── fusion_deliberation.py  ← Serial panel + streaming runner
│   ├── fusion_presets.py       ← 4 個 preset (quality/budget/coding/heavy)
│   ├── fusion_synthesis.py     ← Synthesis prompt + 5-section parser
│   ├── fusion_request.py       ← Plugin / top-level fusion block extractor
│   ├── fusion_retry.py         ← chat_with_retry helper
│   ├── fusion_schemas.py        ← FusionRequest, PanelResponse, SynthesisInput
│   ├── conversation_store.py   ← Persistent cross-request context
│   ├── mlx_optimizations.py    ← LM Studio settings enabler
│   ├── streaming_events.py     ← SSE event types
│   ├── message_parsing.py      ← OpenAI / Anthropic / Responses parsers
│   ├── render_*.py             ← Response renderers
│   ├── cli.py                  ← CLI entry point
│   ├── web/dashboard.html      ← Web UI (single-page)
│   └── ...                     ← (其他輔助模組)
├── qwable_sdk/           ← Python client library
│   ├── __init__.py
│   ├── client.py               ← LocalFusionClient
│   ├── events.py               ← Event dataclasses
│   └── types.py                ← FusionPreset enum + FusionResult
├── scripts/                    ← 啟動 / warmup / E2E
│   ├── start_server.sh
│   ├── install_warmup_launchd.sh
│   ├── warmup_fusion_quality.sh
│   ├── warmup_fusion_budget.sh
│   ├── warmup_now.sh
│   ├── test_fusion_*.sh        ← 5 preset E2E + bad_preset
│   ├── benchmark_speculative.py
│   └── ...                     ← (其他輔助 script)
├── tests/                      ← 287 個 unit tests
│   ├── test_fusion_*.py
│   ├── test_qwable_sdk.py
│   ├── test_mlx_optimizations.py
│   ├── test_conversation_store.py
│   └── ...                     ← (其他測試)
├── Library/LaunchAgents/       ← launchd plist templates
│   ├── io.github.henrylinyy.qwable.gateway.plist
│   ├── io.github.henrylinyy.qwable.warmup.quality.plist
│   └── io.github.henrylinyy.qwable.warmup.budget.plist
└── .github/workflows/          ← CI
    └── ci.yml
```

---

## 🤝 同事間的設定差異

每個同事的差異都寫在 `.env` 和 launchd plist：

| 項目 | 你需要改的 |
|---|---|
| 你的 macOS username | `/Users/yourname/` 出現的地方 |
| `OLLAMA_BASE_URL` | 如果 LM Studio 跑非預設 port |
| `DS4_BASE_URL` | 如果有裝 ds4 |
| `FUSION_DEFAULT_PRESET` | 你最常用的 preset（quality / budget / coding / heavy） |
| `QWABLE_WARMUP_ON_BOOT` | 是否要開機自動預載 |
| launchd plist 內 `__REPLACE_ME__` | 你的絕對路徑 |

其他全部用預設值就能跑。

---

## 🎁 Bonus：能怎麼玩

1. **Web UI** — `http://127.0.0.1:8088/dashboard` 互動式 streaming
2. **Multi-turn 對話** — 用 `/v1/conversations` 存 context，跨請求記得
3. **Streaming** — OpenAI Chat + Responses + Anthropic 三個協議都支援
4. **Judge fallback** — primary judge 失敗自動 retry 到 gemma / r1
5. **Last-model-resident** — 連續請求省 30s 載入時間
6. **CI** — `.github/workflows/ci.yml` 跑 287 個 tests 自動驗證

---

## 📞 還有問題？

1. 先看 `HANDOFF.md` — 包含所有架構決策、開發歷史、known limits
2. 跑 `pytest tests/ -q` 看哪個 test fail
3. 查 `~/Library/Logs/qwable-gateway.err.log`
4. 提 issue / 問同事 / 在 Discord 問

享受本地 multi-model deliberation！🎉

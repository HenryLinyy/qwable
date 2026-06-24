# Qwable

[English](README.md) · [繁體中文](README.zh-TW.md) · [Installation guide](INSTALL.en.md)

[![CI](https://github.com/HenryLinyy/qwable/actions/workflows/ci.yml/badge.svg)](https://github.com/HenryLinyy/qwable/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/Python-3.11%2B-blue.svg)](https://www.python.org/)

**Point Claude Code *and* Codex at your own Mac — local specialist models, a multi-model council that cross-checks itself, and a Qwable-first coding agent runtime. 128GB, fully offline, $0/month.**

Qwable is a single local gateway (`127.0.0.1:8088`) that speaks three agent protocols at once — OpenAI **Responses** (Codex), Anthropic **Messages** (Claude Code), and OpenAI **Chat Completions** (any OpenAI-compatible client). Behind that one endpoint sits a fleet of specialized local models, plus a multi-model **deliberation** mode that makes a single Apple Silicon machine behave like a cloud router with an opinionated panel of experts.

> Most "local LLM" setups hand you one model that's good at text and mediocre at everything else. Qwable does the opposite: it routes each request to the right specialist (coder, reasoner, vision, fast-formatter) and, when you ask for it, convenes several of them into a panel that a judge model synthesizes. You get cloud-style orchestration without the cloud.

---

## Why this exists

A raw local model is narrow. One checkpoint is good at chat, another only at code; reasoning and vision usually need their own dedicated weights. So in practice you end up babysitting five different models and manually picking which one to talk to.

Cloud providers hide all of that behind one smart endpoint. Qwable brings that same experience to your own machine:

- **One endpoint, the agents you already use.** Claude Code, Codex, and OpenAI-SDK clients connect with zero code changes — just a base URL swap.
- **Specialization, not replacement.** Each job is routed to the model built for it instead of forcing one generalist to do everything.
- **A council on a single box.** The deliberation router runs a panel of models *serially* (load → answer → unload) so a "5-model review" fits inside one 128GB Mac without the RAM ever exceeding the largest single model.

If you've ever wanted "Claude Code, but 100% local, with a second and third opinion baked in" — that's the whole idea.

---

## What you get

| Capability | What it means |
| --- | --- |
| 🔌 **Tri-protocol gateway** | `/v1/responses` (Codex), `/v1/messages` (Claude Code), `/v1/chat/completions` (OpenAI clients) — all on one port |
| 🧠 **Task-aware routing** | Each request goes to a specialist: coder, tool-runner, critic, judge, vision, or a fast formatter |
| 🤝 **Fusion deliberation** | OpenRouter-style multi-model panel → judge synthesis → `Final Answer / Consensus / Contradictions / Blind Spots / Per-model Notes` |
| 👁️ **Vision pipeline** | Two-stage: extract auditable visual evidence on a local VLM, then hand off to the heavy reasoner |
| 🪶 **MLX fast path** | Short, tool-free prompts auto-route to a lightweight MLX model — skips ~30s of model-load time |
| 📡 **Streaming** | SSE streaming for deliberation, with per-panel progress events |
| 🧰 **Tool-aware** | `tool_call` / `tool_use` handled and validated across all three protocols |
| 🖥️ **Live dashboard** | Single-file web UI that streams panel/judge events in real time |
| 🐍 **Python SDK** | Call the gateway in-process without going through HTTP |

---

## Architecture

```
                 Claude Code        Codex         Any OpenAI client
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
                │  (OpenAI-compat)  │        │  heavy reasoner  │
                │  coder / vision / │        │  (long context,  │
                │  reasoner / fast  │        │   big repos)     │
                └───────────────────┘        └──────────────────┘
```

Everything is **serial by design**. A single global request lock guarantees one model is resident at a time, so peak memory stays bounded — the price you pay is no parallel requests, which is the right trade on a single workstation.

---

## Quickstart

**Requirements:** Apple Silicon with large unified memory (built and tuned on an **M5 Max, 128GB**), [LM Studio](https://lmstudio.ai) as the local backend, Python 3.11+. The optional `ds4` heavy backend is only needed for long-context / large-repo work.

```bash
git clone https://github.com/HenryLinyy/qwable.git
cd qwable

python3 -m venv .venv
./.venv/bin/pip install -r requirements.txt
cp .env.example .env          # then edit paths/models for your machine

bash scripts/start_server.sh  # binds 127.0.0.1:8088
```

For model setup, launchd, and troubleshooting, see the [installation guide](INSTALL.en.md).

Check it's alive:

```bash
curl http://127.0.0.1:8088/health
curl http://127.0.0.1:8088/v1/models
```

> Security note: the gateway binds to `127.0.0.1` on purpose. Don't bind `0.0.0.0` unless you intend to expose it.

---

## Connect your agent

### Claude Code

```bash
export ANTHROPIC_BASE_URL=http://127.0.0.1:8088
export ANTHROPIC_AUTH_TOKEN=local
export CLAUDE_CODE_ENABLE_GATEWAY_MODEL_DISCOVERY=1
# models: claude-qwable-fast | -full | -heavy | -fusion
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

### Any OpenAI-compatible client

```
Base URL: http://127.0.0.1:8088/v1
API Key:  local
Model:    qwable-chat
```

---

## The council: fusion deliberation

This is the feature worth the repo. Call the `*-fusion` model and Qwable runs a **panel of models**, then a **judge** synthesizes their answers into a structured verdict:

```bash
curl -fsS http://127.0.0.1:8088/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwable-fusion",
    "messages": [{"role":"user","content":"Compare mergesort vs quicksort for our workload"}],
    "plugins": [{"id":"fusion","preset":"quality"}]
  }'
```

The judge returns five sections:

```markdown
## Final Answer        ← the synthesized 1–3 sentence verdict
## Consensus           ← what ≥2 panelists agreed on
## Contradictions      ← where panelists disagreed
## Blind Spots         ← what everyone may have missed
## Per-model Notes     ← a one-line take from each panelist
```

Four presets, each picked so **peak RAM never exceeds the largest single model** (serial load/unload):

| Preset | Panel | Judge | Peak RAM |
| --- | --- | --- | --- |
| `quality` | coder + agentic + reasoner | agentic | ~66GB |
| `budget` | fast + agentic | agentic | ~38GB |
| `coding` | coder + agentic + reasoner | coder | ~66GB |
| `heavy` | coder + reasoner | ds4 heavy | ~66GB + ds4 |

You can also override the panel inline:

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

Add `"stream": true` to watch each panelist finish and the judge's tokens arrive live over SSE.

---

## Agent runtime (v1.8)

Beyond single answers and the council, Qwable ships a multi-step **agent runtime** — a planner → executor → repair → critic → judge loop with tool calls, context compaction, a repo index, and replayable run traces persisted to SQLite. Three profiles drive it:

| Model | What it does |
| --- | --- |
| `qwable-agent` | General long-horizon agent — planning, multi-step tool flows, research |
| `qwable-code-agent` | Coding / repo-patch / test / repair workflow |
| `qwable-review-agent` | Reviews plans, patches, architecture and risk without large rewrites |

```bash
curl http://127.0.0.1:8088/v1/responses \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwable-code-agent","input":"Add retries to the fetch helper. Use tests."}'
```

Each stage resolves a model **role** with its own fallback chain (`MODEL_ROLE_*` in `.env`), so a missing model degrades gracefully instead of failing the run. Two optional Fable/Mythos-style local workers can join the loop: **Qwable** (coding executor/repair, on by default) and **Qwythos** (long-context worker, opt-in). Toggle both in `.env`.

---

## Model specialization

v1.8 routes each profile to the model built for that job (LM Studio model IDs shown; swap for your own in `.env`):

| Role | Model | Approx RAM |
| --- | --- | --- |
| Fast / formatter / fast-vision | `gemma-4-26b-a4b-qat` | ~16GB |
| Coder / tool-runner | `qwen3-coder-next` | ~65GB |
| Vision (OCR / UI / visual-coding) | `qwen3-vl-30b` | ~34GB |
| Agentic reasoning / pro chat | `qwen3.6-35b-a3b` | ~38GB |
| Critic / judge | `deepseek-r1-distill-qwen-32b` | ~66GB |
| Heavy backend (ds4) | `deepseek-v4-flash` | ~90GB |

**Profiles:** `chat-agent` (plain chat), `fast-agent` (daily coding / tool loops), `full-agent` (coder → tooler → critic → judge), `heavy-agent` (ds4 primary with local fallback), `fusion-agent` (the council).

---

## Live dashboard

A single self-contained `qwable/web/dashboard.html` streams the deliberation in real time — panel start/done events, judge tokens, and the final synthesis — so you can actually watch the council think instead of staring at a spinner.

---

## Honest limits

No overselling — these are real and intentional:

- **Single workstation, serial execution.** No parallel requests; one model resident at a time. This is the trade that keeps RAM bounded.
- **One tool call per turn.** Fine for most agent loops, not for heavy fan-out tool use.
- **Streaming v1** is keepalive + chunk/event streaming, not raw token streaming on every path (the fusion judge path *does* stream tokens).
- **Stateless server** — no stored `previous_response_id`.
- **Local models won't always pick the right tool.** Schema validation reduces the blast radius; it doesn't eliminate it.
- **`ds4` heavy backend is beta** and always has a fallback to the local `full-agent`.
- **Hardware is real.** This was built for 128GB unified memory. It will run on less if you shrink the model plan, but the default presets assume headroom.

---

## Status

`v1.8.0` · **521 tests passing** · CI on macOS (pytest + ruff) across Python 3.11 / 3.12.

```bash
./.venv/bin/pytest tests/ -q
```

---

## License

Released under the [MIT License](LICENSE).

Contributions are welcome; see [CONTRIBUTING.md](CONTRIBUTING.md). Please report vulnerabilities through the process in [SECURITY.md](SECURITY.md).

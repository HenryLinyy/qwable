# Qwable — Setup Guide (for new installs)

A local, OpenAI/Anthropic-compatible **AI gateway** that fuses several local
models (via LM Studio) into agent profiles, multi-model "fusion" deliberation,
and coding/agentic/review **workflows**. Runs entirely on your machine.

## 1. Prerequisites

| Need | Notes |
|------|-------|
| **macOS + Apple Silicon** | Built/tested on M-series. Large unified memory recommended (≈128 GB) for the `heavy`/`quality` presets; smaller RAM works for the light profiles. |
| **Python 3.11+** | `python3 --version` |
| **LM Studio** | https://lmstudio.ai — then `lms bootstrap` so the `lms` CLI is on PATH, and `lms server start` (serves on `:1234`). |
| **ds4 service** (optional) | Only for the `heavy` preset / long-context. An OpenAI-compatible server on `:8000` serving `deepseek-v4-flash`. Skip if you don't need heavy. |

## 2. Install

```bash
unzip qwable-*.zip && cd qwable-*
./setup.sh            # venv + deps + .env + backend checks
```

## 3. Download the models (in LM Studio)

These are the model **ids** the gateway expects (override any in `.env`). Pull
them in the LM Studio UI or with `lms get`:

| Model id | Used by | ~Size |
|----------|---------|-------|
| `google/gemma-4-26b-a4b-qat` | fast / formatter / vision-fast / agentic profiles | ~16 GB |
| `qwen/qwen3-coder-next` | coder / executor / repair | ~65 GB |
| `qwen/qwen3.6-35b-a3b` | planner / judge / fusion | ~38 GB |
| `qwen/qwen3-vl-30b` | vision-pro | ~34 GB |
| `deepseek-r1-distill-qwen-32b` | critic / fusion panel | ~66 GB |
| `deepseek-v4-flash` (via ds4 `:8000`) | heavy / long-context | external |

**v1.8-qwable build only** also uses (download from `empero-ai` on Hugging Face):
`qwable-9b-claude-fable-5` (default executor) and, optionally,
`qwythos-9b-claude-mythos-5-1m` (long-context, off by default).

> You don't need every model to start — the light profiles only need
> `gemma-4-26b`. Pull the rest as you use heavier profiles.

## 4. Configure

`.env` is created from `.env.example` with sensible defaults (host `127.0.0.1`,
port `8088`, model ids above). Edit only if your ports/model ids differ.
`LMSTUDIO_CLI_PATH` defaults to `$HOME/.lmstudio/bin/lms`.

## 5. Start

```bash
./.venv/bin/python -m uvicorn qwable.server:app --host 127.0.0.1 --port 8088
```

(Optional auto-start on login: copy a plist from `Library/LaunchAgents/` into
`~/Library/LaunchAgents/`, edit the absolute paths inside to your checkout, then
`launchctl load` it.)

## 6. Verify

```bash
bash scripts/verify_all.sh          # structural + per-profile + fusion + vision
bash scripts/verify_all.sh --full   # also runs the 3 agent workflows (slower)
```

All green = working.

## 7. Use it

Point any OpenAI/Anthropic client at `http://127.0.0.1:8088`. Pick behavior via
the model name:

```bash
# OpenAI Chat — fast profile
curl http://127.0.0.1:8088/v1/chat/completions -H "Content-Type: application/json" \
  -d '{"model":"qwable-fast","messages":[{"role":"user","content":"hi"}]}'

# Fusion deliberation (panel -> judge). Do NOT set a tiny max_tokens — the
# thinking-model judge needs room.
curl http://127.0.0.1:8088/v1/chat/completions -H "Content-Type: application/json" \
  -d '{"model":"qwable-fusion","fusion":{"preset":"budget"},
       "messages":[{"role":"user","content":"explain X in one sentence"}]}'

# Coding workflow (plan -> critic -> execute -> ...)
curl http://127.0.0.1:8088/v1/chat/completions -H "Content-Type: application/json" \
  -d '{"model":"qwable-code-agent","messages":[{"role":"user","content":"write clamp(x,lo,hi)"}]}'
```

Model-name families: `qwable-{fast,full,chat,heavy}`,
`qwable-vision-{fast,pro}`, `qwable-{agentic-pro,hermes-pro,agentic-mlx,formatter-mlx}`,
`qwable-fusion`, `qwable-{agent,code-agent,review-agent}`.
Anthropic clients use the `claude-qwable-*` aliases on `/v1/messages`.

### Choosing a fusion preset (and memory)

Pick the preset right in the model name (handy for GUI clients):
`qwable-fusion-{budget,quality,coding,heavy}` (or pass `{"fusion":{"preset":"..."}}`).

- **budget** (default) — gemma + qwen3.6, ~light. Use for everyday questions.
- **quality** — 3 large models (qwen3-coder + qwen3.6 + deepseek-r1). Deep
  analysis, but **memory-heavy** (~150 GB resident if all stay loaded).
- **heavy** — uses the ds4 service (`:8000`).

⚠️ Fusion/heavy load several big models. They run sequentially but LM Studio
keeps them resident (now auto-unloaded after `LMSTUDIO_TTL_SECONDS`, default
600s). Don't run `quality`/`heavy` while also running ds4 + Docker + other big
models, or the machine will swap. Use `qwable-fast`/`-chat` for plain
Q&A; reserve `Qwable Agent` (a multi-step task workflow) for actual tasks,
not questions.

## 8. Troubleshooting

- **Memory**: `heavy` (and `quality`) load multiple large models; on a busy box
  free memory can dip low. `scripts/stress_test.sh` auto-runs `lms unload --all`
  when free memory drops below a threshold. If LM Studio thrashes, unload models
  (`lms unload --all`) between heavy calls.
- **Thinking models** (qwen3.6, Qwable): they emit reasoning separately and need
  a real token budget — don't cap fusion/workflow `max_tokens` too low.
- **`heavy` preset fails**: ensure the ds4 service is up on `:8000`.
- **Empty/Ë errors**: run `bash scripts/verify_all.sh` to localize which
  profile/backend is the problem.

Full design/architecture details live in `README.md`.

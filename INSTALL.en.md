# Installing Qwable

[English](INSTALL.en.md) · [繁體中文](INSTALL.md)

Qwable is built for Apple Silicon and uses LM Studio as its local OpenAI-compatible model backend.

## Requirements

- Apple Silicon Mac running macOS 13 or newer
- Python 3.11 or 3.12
- [LM Studio](https://lmstudio.ai/) 0.4.16 or newer
- 32 GB unified memory minimum; 64 GB or more recommended
- 128 GB for the default `quality` and `coding` model plans

The optional `ds4` backend is only required by the `heavy` preset.

## 1. Clone and install

```bash
git clone https://github.com/HenryLinyy/qwable.git
cd qwable

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

## 2. Configure

```bash
cp .env.example .env
```

The defaults expect:

- LM Studio at `http://127.0.0.1:1234/v1`
- Qwable at `http://127.0.0.1:8088`
- LM Studio CLI at `~/.lmstudio/bin/lms`

Edit `.env` if your ports, model IDs, or CLI path differ. Qwable reads this file automatically when started from the repository directory.

## 3. Prepare LM Studio

Start the LM Studio local server, then verify its API:

```bash
~/.lmstudio/bin/lms --version
curl http://127.0.0.1:1234/v1/models
```

The smallest default preset is `budget`, which uses:

| Role | Default model | Approximate memory |
| --- | --- | ---: |
| Fast panelist | `google/gemma-4-26b-a4b-qat` | 16 GB |
| Agentic panelist and judge | `qwen/qwen3.6-35b-a3b` | 38 GB |

Model IDs are configurable in `.env`; you do not need to use the defaults.

## 4. Test and start

```bash
pytest tests/ -q
bash scripts/start_server.sh
```

In another terminal:

```bash
curl http://127.0.0.1:8088/health
curl http://127.0.0.1:8088/v1/models
```

Open `http://127.0.0.1:8088/dashboard` for the live deliberation dashboard.

## 5. Connect a client

### Claude Code

```bash
export ANTHROPIC_BASE_URL=http://127.0.0.1:8088
export ANTHROPIC_AUTH_TOKEN=local
export CLAUDE_CODE_ENABLE_GATEWAY_MODEL_DISCOVERY=1
```

Use `claude-qwable-fast`, `claude-qwable-full`, `claude-qwable-heavy`, or `claude-qwable-fusion`.

### Codex

Add this provider to `~/.codex/config.toml`:

```toml
model = "qwable-fast"
model_provider = "qwable"

[model_providers.qwable]
name = "Qwable"
base_url = "http://127.0.0.1:8088/v1"
wire_api = "responses"
env_key = "QWABLE_API_KEY"
```

Then set any local placeholder value:

```bash
export QWABLE_API_KEY=local
```

### OpenAI-compatible clients

Use:

```text
Base URL: http://127.0.0.1:8088/v1
API key:  local
Model:    qwable-chat
```

## Optional: launchd

The repository includes portable launchd templates. Replace the example paths before installing the gateway template:

```bash
mkdir -p ~/Library/LaunchAgents ~/Library/Logs
sed \
  -e "s|/Users/yourname/Documents/qwable|$(pwd)|g" \
  -e "s|/Users/yourname|$HOME|g" \
  Library/LaunchAgents/io.github.henrylinyy.qwable.gateway.plist \
  > ~/Library/LaunchAgents/io.github.henrylinyy.qwable.gateway.plist

launchctl load -w ~/Library/LaunchAgents/io.github.henrylinyy.qwable.gateway.plist
```

Install the optional weekday model warmups with:

```bash
./scripts/install_warmup_launchd.sh
```

The installer renders your home and repository paths into the installed plist files.

## Troubleshooting

### Gateway starts but model calls fail

Confirm that LM Studio is running and that `.env` uses the same port:

```bash
curl http://127.0.0.1:1234/v1/models
```

### A fusion response has no visible output

Reasoning models can spend their output budget on reasoning. Send `max_tokens: 4000` or higher.

### Port 8088 is already in use

Set another port in `.env`:

```dotenv
QWABLE_PORT=8090
```

### Uninstall launchd jobs

```bash
launchctl unload ~/Library/LaunchAgents/io.github.henrylinyy.qwable.gateway.plist
launchctl unload ~/Library/LaunchAgents/io.github.henrylinyy.qwable.warmup.quality.plist
launchctl unload ~/Library/LaunchAgents/io.github.henrylinyy.qwable.warmup.budget.plist
```

The gateway binds to `127.0.0.1` by default. Do not expose it on `0.0.0.0` without adding authentication and network controls.

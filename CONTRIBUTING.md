# Contributing

Bug reports and focused pull requests are welcome.

## Development setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

## Before opening a pull request

```bash
pytest tests/ -q
ruff check qwable qwable_sdk tests
ruff format --check qwable qwable_sdk tests
```

Keep changes small, include one regression test for behavior changes, and do not commit model weights, `.env`, logs, or local conversation data.

Real-model integration tests require LM Studio and are intentionally separate from the mocked CI suite.

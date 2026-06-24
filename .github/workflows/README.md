# CI / GitHub Actions

G14-3: GitHub Actions workflow at `.github/workflows/ci.yml` runs:

- **Test job** (Python 3.11, 3.12 matrix on macOS-latest for MLX/Apple Silicon):
  - Install dependencies (with pip cache)
  - Cache LM Studio / Ollama (saves 5+ min per run)
  - Run `pytest tests/ -q --tb=short --maxfail=5`
  - Coverage report (`xml` + `term-missing`) — uploads to Codecov
- **Lint job** (Python 3.12):
  - ruff check + format

Triggers:
- push to main / develop / gate/*
- pull_request to main / develop

Note: CI tests do NOT require real LM Studio / ds4 services — all unit
tests are mock-based. E2E tests against real services are run manually
on the local M5 Max via `scripts/test_fusion_*.sh`.

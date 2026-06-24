"""Tests for advertised gateway model aliases."""

from fastapi.testclient import TestClient

from qwable.config import FusionConfig
from qwable.server import MODELS_LIST


def test_models_list_advertises_v15_profiles():
    ids = {item["id"] for item in MODELS_LIST["data"]}

    assert {
        "qwable-vision-fast",
        "qwable-vision-pro",
        "qwable-vision-heavy",
        "qwable-agentic-pro",
        "qwable-hermes-pro",
        "qwable-agentic-mlx",
        "qwable-formatter-mlx",
        "claude-qwable-vision-fast",
        "claude-qwable-vision-pro",
        "claude-qwable-vision-heavy",
        "claude-qwable-agentic-pro",
        "claude-qwable-hermes-pro",
        "claude-qwable-agentic-mlx",
        "claude-qwable-formatter-mlx",
    }.issubset(ids)


def test_models_list_advertises_g10_fusion_profiles():
    """G10: OpenRouter-style fusion deliberation router model ids."""
    ids = {item["id"] for item in MODELS_LIST["data"]}

    assert "qwable-fusion" in ids
    assert "claude-qwable-fusion" in ids


def test_models_list_advertises_v17_agent_workflow_profiles():
    ids = {item["id"] for item in MODELS_LIST["data"]}

    assert {
        "qwable-agent",
        "qwable-code-agent",
        "qwable-review-agent",
        "claude-qwable-agent",
        "claude-qwable-code-agent",
        "claude-qwable-review-agent",
    }.issubset(ids)


def test_health_advertises_agent_runtime_metadata_without_dropping_existing_fields():
    import qwable.server as server_mod

    original_config = server_mod.config
    server_mod.config = FusionConfig(agent_store_path=".qwable_agent_runs.sqlite3")
    try:
        client = TestClient(server_mod.app)

        response = client.get("/health")
    finally:
        server_mod.config = original_config

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["version"] == "1.5.0"
    assert body["agent_runtime_enabled"] is True
    assert body["agent_store_path"] == ".qwable_agent_runs.sqlite3"
    assert body["agent_profiles"] == [
        "qwable-agent",
        "qwable-code-agent",
        "qwable-review-agent",
    ]
    assert body["model_roles"] == {
        "planner": "qwen/qwen3.6-35b-a3b",
        "executor": "qwen/qwen3-coder-next",
        "repair": "qwen/qwen3-coder-next",
        "critic": "deepseek-r1-distill-qwen-32b",
        "judge": "qwen/qwen3.6-35b-a3b",
    }


def test_v1_health_matches_agent_runtime_metadata_contract():
    import qwable.server as server_mod

    original_config = server_mod.config
    server_mod.config = FusionConfig(agent_store_path=".qwable_agent_runs.sqlite3")
    try:
        client = TestClient(server_mod.app)

        response = client.get("/v1/health")
    finally:
        server_mod.config = original_config

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["version"] == "1.5.0"
    assert body["agent_runtime_enabled"] is True
    assert body["agent_profiles"] == [
        "qwable-agent",
        "qwable-code-agent",
        "qwable-review-agent",
    ]

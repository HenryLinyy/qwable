"""G12-4: tests for GET /v1/fusion/presets endpoint."""

from unittest.mock import MagicMock


def test_v1_fusion_presets_returns_4_presets(monkeypatch):
    """Endpoint should return all 4 presets with panel/judge info."""
    import qwable.server as server_mod
    from qwable.config import FusionConfig
    from qwable.fusion_core import FusionCore

    # Stub out external probes
    def fake_subprocess_run(*args, **kwargs):
        m = MagicMock()
        m.stdout = "No models are currently loaded.\n"
        return m

    monkeypatch.setattr("subprocess.run", fake_subprocess_run)

    config = FusionConfig()
    fc = FusionCore(config)

    server_mod.config = config
    server_mod.fusion_core = fc

    from fastapi.testclient import TestClient

    client = TestClient(server_mod.app)
    response = client.get("/v1/fusion/presets")
    assert response.status_code == 200
    data = response.json()
    assert set(data["presets"]) == {"quality", "budget", "coding", "heavy"}
    assert data["presets"]["quality"]["judge"] == "qwen/qwen3.6-35b-a3b"
    assert data["presets"]["heavy"]["judge"] == "deepseek-v4-flash"
    assert data["presets"]["heavy"]["judge_backend"] == "ds4"
    assert data["presets"]["budget"]["judge_backend"] == "ollama"
    assert (
        data["default_preset"] == "budget"
    )  # default lowered to budget for memory safety


def test_v1_fusion_presets_loaded_returns_quick(monkeypatch):
    """Lightweight /loaded endpoint should not probe ds4."""
    import qwable.server as server_mod
    from qwable.config import FusionConfig
    from qwable.fusion_core import FusionCore

    def fake_subprocess_run(*args, **kwargs):
        m = MagicMock()
        m.stdout = "qwen/qwen3.6-35b-a3b    qwen/qwen3.6-35b-a3b    IDLE    37.75GB\n"
        return m

    monkeypatch.setattr("subprocess.run", fake_subprocess_run)

    config = FusionConfig()
    fc = FusionCore(config)
    server_mod.config = config
    server_mod.fusion_core = fc

    from fastapi.testclient import TestClient

    client = TestClient(server_mod.app)
    response = client.get("/v1/fusion/presets/loaded")
    assert response.status_code == 200
    data = response.json()
    assert "qwen/qwen3.6-35b-a3b" in data["loaded_models"]


def test_v1_fusion_presets_includes_ds4_reachability(monkeypatch):
    """ds4_reachable should reflect whether /v1/models probe succeeded."""
    import qwable.server as server_mod
    from qwable.config import FusionConfig
    from qwable.fusion_core import FusionCore

    def fake_subprocess_run(*args, **kwargs):
        m = MagicMock()
        m.stdout = ""
        return m

    # Mock httpx.Client to return 200 for ds4 probe
    class FakeResp:
        status_code = 200

    class FakeClient:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def get(self, url):
            return FakeResp()

    monkeypatch.setattr("subprocess.run", fake_subprocess_run)
    monkeypatch.setattr("httpx.Client", FakeClient)

    config = FusionConfig()
    fc = FusionCore(config)
    server_mod.config = config
    server_mod.fusion_core = fc

    from fastapi.testclient import TestClient

    client = TestClient(server_mod.app)
    response = client.get("/v1/fusion/presets")
    data = response.json()
    assert data["ds4_reachable"] is True

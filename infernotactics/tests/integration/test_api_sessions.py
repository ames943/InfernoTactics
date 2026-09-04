from pathlib import Path
import sys

import pytest


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
LEGACY_GRID = REPOSITORY_ROOT / "infernotactics" / "data" / "grid_static_legacy_heuristic.npy"
CHECKPOINT = (
    REPOSITORY_ROOT
    / "infernotactics"
    / "models"
    / "containment_policy_v10.pt"
)


@pytest.mark.skipif(
    not LEGACY_GRID.exists() or not CHECKPOINT.exists(),
    reason="local legacy world/checkpoint integration assets are unavailable",
)
def test_api_sessions_are_isolated_and_share_read_only_assets():
    sys.path.insert(0, str(REPOSITORY_ROOT))
    from fastapi.testclient import TestClient
    import infernotactics.api.server as server

    client = TestClient(server.app)
    response = client.post("/api/sessions")
    assert response.status_code == 201
    session_id = response.json()["session_id"]
    child = server.sessions.get(session_id)
    try:
        assert child.env is not server.session.env
        assert child.model is server.session.model
        assert child.env.routing_context is server.session.env.routing_context
        assert client.post(
            "/api/step",
            json={"session_id": session_id, "mode": "manual", "actions": []},
        ).json()["tick"] == 1
        assert client.get("/api/state").json()["tick"] == 0
        assert client.post(
            "/api/reset",
            json={"session_id": session_id, "ignition_point": [-1, 0]},
        ).status_code == 422
        assert client.post(
            "/api/step",
            json={"session_id": session_id, "mode": "unsupported"},
        ).status_code == 422
    finally:
        assert client.delete(f"/api/sessions/{session_id}").status_code == 200
    assert client.get("/api/state", params={"session_id": session_id}).status_code == 404

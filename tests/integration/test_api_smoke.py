from fastapi.testclient import TestClient

from distillshield_api.main import app


client = TestClient(app)


def test_health_and_simulation_flow():
    health = client.get("/health")
    assert health.status_code == 200
    simulate = client.post("/simulate/session", json={"num_users": 4, "sessions_per_user": 1, "seed": 7, "persist": True})
    assert simulate.status_code == 200
    sessions = client.get("/sessions")
    assert sessions.status_code == 200
    assert len(sessions.json()) >= 1

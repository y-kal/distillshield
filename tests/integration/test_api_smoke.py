from distillshield_api.main import SessionIdRequest, get_session_detail, health, score_session, simulate_session
from distillshield_core.schemas import SimulationRequest
from distillshield_storage.database import create_db_and_tables


def test_health_and_simulation_flow():
    create_db_and_tables()
    health_payload = health()
    assert health_payload["status"] == "ok"

    simulate = simulate_session(SimulationRequest(num_users=4, sessions_per_user=1, seed=7, persist=True))
    assert simulate["generated_sessions"] == 4
    assert simulate["sample_session_ids"]

    session_id = simulate["sample_session_ids"][0]
    scored = score_session(SessionIdRequest(session_id=session_id))
    assert 0.0 <= scored.risk_score <= 1.0
    assert scored.explainability.category_scores
    assert scored.top_reasons

    detail = get_session_detail(session_id)
    assert detail["risk_assessment"]["predicted_class"] == scored.predicted_class.value
    assert "triggered_rules" in detail["risk_assessment"]

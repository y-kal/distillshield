from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sqlmodel import select

from distillshield_core.config import get_settings
from distillshield_core.schemas import (
    EvaluateRequest,
    ExperimentSummary,
    IngestSessionRequest,
    PolicyDecisionResult,
    RiskAssessmentResult,
    SessionRecord,
    SimulationRequest,
    TransformationResult,
)
from distillshield_eval.runner import EvaluationRunner
from distillshield_llm_adapter.transform import TransformationEngine
from distillshield_models import DistillShieldEngine
from distillshield_models.policy import PolicyEngine
from distillshield_storage.database import create_db_and_tables, get_session
from distillshield_storage.models import ExperimentRunEntity, FeatureVectorEntity, SessionEntity, User
from distillshield_storage.repository import (
    get_session_bundle,
    list_sessions,
    save_experiment_run,
    save_feature_vector,
    save_policy_decision,
    save_risk_assessment,
    save_transformation,
    upsert_session,
)
from distillshield_synthetic_data.generator import SyntheticDataGenerator


settings = get_settings()
synthetic_generator = SyntheticDataGenerator(seed=7)
engine = DistillShieldEngine()
policy_engine = PolicyEngine()
transformer = TransformationEngine()
evaluator = EvaluationRunner()


@asynccontextmanager
async def lifespan(app: FastAPI):
    create_db_and_tables()
    yield


app = FastAPI(title="DistillShield API", version="0.1.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[settings.frontend_origin, "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class SessionIdRequest(BaseModel):
    session_id: str


class PolicyRequest(BaseModel):
    session: SessionRecord
    predicted_class: str
    risk_score: float
    confidence: float


class TransformRequest(BaseModel):
    session: SessionRecord
    policy: str
    raw_output: str | None = None


def _session_from_db(session_id: str) -> SessionRecord:
    with get_session() as db:
        bundle = get_session_bundle(db, session_id)
    if bundle["session"] is None:
        raise KeyError(session_id)
    session_entity = bundle["session"]
    queries = bundle["queries"]
    responses = bundle["responses"]
    api_context = bundle["api_context"]
    return SessionRecord(
        id=session_entity.id,
        user_id=session_entity.user_id,
        session_started_at=session_entity.started_at,
        session_ended_at=session_entity.ended_at,
        queries=[
            {
                "id": query.id,
                "text": query.text,
                "timestamp": query.timestamp,
                "max_tokens": query.max_tokens,
                "streamed": query.streamed,
            }
            for query in queries
        ],
        responses=[
            {
                "id": response.id,
                "query_id": response.query_id,
                "raw_output": response.raw_output,
                "completion_tokens": response.completion_tokens,
                "truncated": response.truncated,
                "refusal": response.refusal,
                "created_at": response.created_at,
            }
            for response in responses
        ],
        api_context={
            "api_key_id": api_context.api_key_id,
            "org_id": api_context.org_id,
            "user_agent": api_context.user_agent,
            "ip_address": api_context.ip_address,
            "geo_region": api_context.geo_region,
            "quota_used": api_context.quota_used,
            "source": api_context.source,
        },
        label=session_entity.label,
        metadata=session_entity.session_metadata,
    )


def _ensure_analysis(session: SessionRecord) -> None:
    with get_session() as db:
        bundle = get_session_bundle(db, session.id)
        if bundle["risk_assessment"] and bundle["policy_decision"] and bundle["transformation"]:
            return

    assessment = engine.assess(session)
    decision = policy_engine.decide(
        session,
        assessment.predicted_class,
        assessment.risk_score,
        assessment.confidence,
        category_scores=assessment.category_scores,
        triggered_rules=assessment.triggered_rules,
    )
    transformed = transformer.transform(session, decision.chosen_policy)
    with get_session() as db:
        save_feature_vector(db, session.id, {item.name: item.model_dump() for item in assessment.feature_values})
        save_risk_assessment(db, assessment)
        save_policy_decision(db, decision)
        save_transformation(db, transformed)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": "distillshield-api"}


@app.post("/simulate/session")
def simulate_session(request: SimulationRequest) -> dict[str, Any]:
    generator = SyntheticDataGenerator(seed=request.seed)
    sessions = generator.generate_sessions(num_users=request.num_users, sessions_per_user=request.sessions_per_user)
    if request.persist:
        with get_session() as db:
            for session in sessions:
                upsert_session(db, session)
    return {
        "generated_sessions": len(sessions),
        "class_distribution": generator.class_distribution(sessions),
        "sample_session_ids": [session.id for session in sessions[:5]],
    }


@app.post("/ingest/session")
def ingest_session(request: IngestSessionRequest) -> dict[str, str]:
    with get_session() as db:
        upsert_session(db, request.session)
    return {"status": "stored", "session_id": request.session.id}


@app.post("/score/session", response_model=RiskAssessmentResult)
def score_session(request: SessionIdRequest) -> RiskAssessmentResult:
    try:
        session = _session_from_db(request.session_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Session not found") from exc
    assessment = engine.assess(session)
    with get_session() as db:
        save_feature_vector(db, session.id, {item.name: item.model_dump() for item in assessment.feature_values})
        save_risk_assessment(db, assessment)
    return assessment


@app.post("/classify/session", response_model=RiskAssessmentResult)
def classify_session(request: SessionIdRequest) -> RiskAssessmentResult:
    return score_session(request)


@app.post("/policy/decide", response_model=PolicyDecisionResult)
def policy_decide(request: PolicyRequest) -> PolicyDecisionResult:
    from distillshield_core.enums import BehaviorClass

    predicted_class = BehaviorClass(request.predicted_class)
    decision = policy_engine.decide(request.session, predicted_class, request.risk_score, request.confidence)
    with get_session() as db:
        save_policy_decision(db, decision)
    return decision


@app.post("/transform/output", response_model=TransformationResult)
def transform_output(request: TransformRequest) -> TransformationResult:
    from distillshield_core.enums import OutputPolicy

    result = transformer.transform(request.session, OutputPolicy(request.policy), raw_output=request.raw_output)
    with get_session() as db:
        save_transformation(db, result)
    return result


@app.post("/evaluate")
def evaluate(request: EvaluateRequest) -> dict[str, Any]:
    result = evaluator.run(seed=request.seed, num_users=request.num_users, sessions_per_user=request.sessions_per_user)
    with get_session() as db:
        save_experiment_run(db, result["experiment_id"], result["metrics"], result["artifact_paths"])
    return result


@app.get("/sessions")
def get_sessions() -> list[dict[str, Any]]:
    with get_session() as db:
        return [
            {
                "id": session.id,
                "user_id": session.user_id,
                "started_at": session.started_at,
                "ended_at": session.ended_at,
                "label": session.label,
                "metadata": session.session_metadata,
            }
            for session in list_sessions(db)
        ]


@app.get("/sessions/{session_id}")
def get_session_detail(session_id: str) -> dict[str, Any]:
    try:
        session = _session_from_db(session_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Session not found") from exc
    _ensure_analysis(session)
    with get_session() as db:
        bundle = get_session_bundle(db, session_id)
    return {
        "session": {
            "id": bundle["session"].id,
            "user_id": bundle["session"].user_id,
            "started_at": bundle["session"].started_at.isoformat(),
            "ended_at": bundle["session"].ended_at.isoformat(),
            "label": bundle["session"].label,
            "metadata": bundle["session"].session_metadata,
        },
        "queries": [
            {
                "id": query.id,
                "text": query.text,
                "timestamp": query.timestamp.isoformat(),
                "max_tokens": query.max_tokens,
                "streamed": query.streamed,
            }
            for query in bundle["queries"]
        ],
        "responses": [
            {
                "id": response.id,
                "query_id": response.query_id,
                "raw_output": response.raw_output,
                "completion_tokens": response.completion_tokens,
                "truncated": response.truncated,
                "refusal": response.refusal,
                "created_at": response.created_at.isoformat(),
            }
            for response in bundle["responses"]
        ],
        "features": {
            "feature_payload": bundle["features"].feature_payload if bundle["features"] else {},
            "created_at": bundle["features"].created_at.isoformat() if bundle["features"] else None,
        },
        "risk_assessment": {
            "risk_score": bundle["risk_assessment"].risk_score,
            "predicted_class": bundle["risk_assessment"].predicted_class,
            "confidence": bundle["risk_assessment"].confidence,
            "explainability": bundle["risk_assessment"].explainability,
            "category_scores": bundle["risk_assessment"].category_scores,
            "top_reasons": bundle["risk_assessment"].top_reasons,
            "triggered_rules": bundle["risk_assessment"].triggered_rules,
            "risk_reducers": bundle["risk_assessment"].risk_reducers,
            "reasons": bundle["risk_assessment"].reasons,
        } if bundle["risk_assessment"] else None,
        "policy_decision": {
            "chosen_policy": bundle["policy_decision"].chosen_policy,
            "policy_reason": bundle["policy_decision"].policy_reason,
            "override_applied": bundle["policy_decision"].override_applied,
            "thresholds": bundle["policy_decision"].thresholds,
        } if bundle["policy_decision"] else None,
        "transformation": {
            "policy": bundle["transformation"].policy,
            "raw_output": bundle["transformation"].raw_output,
            "transformed_output": bundle["transformation"].transformed_output,
            "leakage_proxy_score": bundle["transformation"].leakage_proxy_score,
            "leakage_factors": bundle["transformation"].leakage_factors,
        } if bundle["transformation"] else None,
        "api_context": {
            "api_key_id": bundle["api_context"].api_key_id,
            "org_id": bundle["api_context"].org_id,
            "user_agent": bundle["api_context"].user_agent,
            "ip_address": bundle["api_context"].ip_address,
            "geo_region": bundle["api_context"].geo_region,
            "quota_used": bundle["api_context"].quota_used,
            "source": bundle["api_context"].source,
        } if bundle["api_context"] else None,
    }


@app.get("/users")
def get_users() -> list[dict[str, Any]]:
    with get_session() as db:
        users = db.exec(select(User)).all()
    return [{"id": user.id, "label": user.label, "created_at": user.created_at, "metadata": user.user_metadata} for user in users]


@app.get("/features/{session_id}")
def get_features(session_id: str) -> dict[str, Any]:
    with get_session() as db:
        feature_row = db.exec(select(FeatureVectorEntity).where(FeatureVectorEntity.session_id == session_id).order_by(FeatureVectorEntity.created_at.desc())).first()
    if not feature_row:
        raise HTTPException(status_code=404, detail="Features not found")
    return {"session_id": session_id, "features": feature_row.feature_payload, "created_at": feature_row.created_at}


@app.get("/experiments/latest", response_model=ExperimentSummary | None)
def latest_experiment():
    with get_session() as db:
        experiment = db.exec(select(ExperimentRunEntity).order_by(ExperimentRunEntity.created_at.desc())).first()
    if not experiment:
        return None
    return ExperimentSummary(
        experiment_id=experiment.id,
        created_at=experiment.created_at,
        metrics=experiment.metrics,
        artifact_paths=experiment.artifact_paths,
    )

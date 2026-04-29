from __future__ import annotations

from datetime import datetime, UTC
from typing import Iterable

from sqlmodel import Session, select

from distillshield_core.schemas import (
    PolicyDecisionResult,
    RiskAssessmentResult,
    SessionRecord,
    TransformationResult,
)

from .models import (
    APIContextEntity,
    ExperimentRunEntity,
    FeatureVectorEntity,
    PolicyDecisionEntity,
    QueryEntity,
    ResponseEntity,
    RiskAssessmentEntity,
    SessionEntity,
    TransformationArtifactEntity,
    User,
)


def upsert_session(session: Session, record: SessionRecord) -> None:
    user = session.get(User, record.user_id)
    if not user:
        user = User(id=record.user_id, label=record.label.value if record.label else None, created_at=record.session_started_at, user_metadata={})
        session.add(user)

    existing_session = session.get(SessionEntity, record.id)
    if not existing_session:
        session.add(
            SessionEntity(
                id=record.id,
                user_id=record.user_id,
                started_at=record.session_started_at,
                ended_at=record.session_ended_at,
                label=record.label.value if record.label else None,
                session_metadata=record.metadata,
            )
        )

    for query in record.queries:
        if not session.get(QueryEntity, query.id):
            session.add(
                QueryEntity(
                    id=query.id,
                    session_id=record.id,
                    text=query.text,
                    timestamp=query.timestamp,
                    max_tokens=query.max_tokens,
                    streamed=query.streamed,
                )
            )
    for response in record.responses:
        if not session.get(ResponseEntity, response.id):
            session.add(
                ResponseEntity(
                    id=response.id,
                    session_id=record.id,
                    query_id=response.query_id,
                    raw_output=response.raw_output,
                    completion_tokens=response.completion_tokens,
                    truncated=response.truncated,
                    refusal=response.refusal,
                    created_at=response.created_at,
                )
            )

    existing_ctx = session.exec(select(APIContextEntity).where(APIContextEntity.session_id == record.id)).first()
    if not existing_ctx:
        session.add(
            APIContextEntity(
                session_id=record.id,
                api_key_id=record.api_context.api_key_id,
                org_id=record.api_context.org_id,
                user_agent=record.api_context.user_agent,
                ip_address=record.api_context.ip_address,
                geo_region=record.api_context.geo_region,
                quota_used=record.api_context.quota_used,
                source=record.api_context.source,
            )
        )

    session.commit()


def save_feature_vector(session: Session, session_id: str, feature_payload: dict) -> None:
    session.add(FeatureVectorEntity(session_id=session_id, feature_payload=feature_payload, created_at=datetime.now(UTC)))
    session.commit()


def save_risk_assessment(session: Session, assessment: RiskAssessmentResult) -> None:
    session.add(
        RiskAssessmentEntity(
            session_id=assessment.session_id,
            risk_score=assessment.risk_score,
            predicted_class=assessment.predicted_class.value,
            confidence=assessment.confidence,
            explainability=assessment.explainability.model_dump(mode="json"),
            category_scores=assessment.category_scores,
            top_reasons=assessment.top_reasons,
            triggered_rules=[rule.model_dump() for rule in assessment.triggered_rules],
            risk_reducers=assessment.risk_reducers,
            reasons=[reason.model_dump() for reason in assessment.reasons],
            created_at=datetime.now(UTC),
        )
    )
    session.commit()


def save_policy_decision(session: Session, decision: PolicyDecisionResult) -> None:
    session.add(
        PolicyDecisionEntity(
            session_id=decision.session_id,
            chosen_policy=decision.chosen_policy.value,
            policy_reason=decision.policy_reason,
            override_applied=decision.override_applied,
            thresholds=decision.thresholds,
            created_at=datetime.now(UTC),
        )
    )
    session.commit()


def save_transformation(session: Session, transformation: TransformationResult) -> None:
    session.add(
        TransformationArtifactEntity(
            session_id=transformation.session_id,
            policy=transformation.policy.value,
            raw_output=transformation.raw_output,
            transformed_output=transformation.transformed_output,
            leakage_proxy_score=transformation.leakage_proxy_score,
            leakage_factors=transformation.leakage_factors,
            created_at=datetime.now(UTC),
        )
    )
    session.commit()


def save_experiment_run(session: Session, experiment_id: str, metrics: dict, artifact_paths: dict, notes: str | None = None) -> None:
    session.add(
        ExperimentRunEntity(
            id=experiment_id,
            created_at=datetime.now(UTC),
            metrics=metrics,
            artifact_paths=artifact_paths,
            notes=notes,
        )
    )
    session.commit()


def list_sessions(session: Session) -> Iterable[SessionEntity]:
    return session.exec(select(SessionEntity).order_by(SessionEntity.started_at.desc())).all()


def get_session_bundle(session: Session, session_id: str) -> dict:
    entity = session.get(SessionEntity, session_id)
    if not entity:
        raise KeyError(session_id)
    queries = session.exec(select(QueryEntity).where(QueryEntity.session_id == session_id).order_by(QueryEntity.timestamp)).all()
    responses = session.exec(select(ResponseEntity).where(ResponseEntity.session_id == session_id).order_by(ResponseEntity.created_at)).all()
    features = session.exec(select(FeatureVectorEntity).where(FeatureVectorEntity.session_id == session_id).order_by(FeatureVectorEntity.created_at.desc())).first()
    risk = session.exec(select(RiskAssessmentEntity).where(RiskAssessmentEntity.session_id == session_id).order_by(RiskAssessmentEntity.created_at.desc())).first()
    policy = session.exec(select(PolicyDecisionEntity).where(PolicyDecisionEntity.session_id == session_id).order_by(PolicyDecisionEntity.created_at.desc())).first()
    artifact = session.exec(select(TransformationArtifactEntity).where(TransformationArtifactEntity.session_id == session_id).order_by(TransformationArtifactEntity.created_at.desc())).first()
    api_context = session.exec(select(APIContextEntity).where(APIContextEntity.session_id == session_id)).first()
    return {
        "session": entity,
        "queries": queries,
        "responses": responses,
        "features": features,
        "risk_assessment": risk,
        "policy_decision": policy,
        "transformation": artifact,
        "api_context": api_context,
    }

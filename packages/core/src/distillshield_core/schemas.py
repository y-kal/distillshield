from __future__ import annotations

from datetime import datetime
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field

from .enums import BehaviorClass, FeatureProvenance, OutputPolicy


class QueryRecord(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    text: str
    timestamp: datetime
    max_tokens: int = 512
    streamed: bool = False


class ResponseRecord(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    query_id: str
    raw_output: str
    completion_tokens: int = 0
    truncated: bool = False
    refusal: bool = False
    created_at: datetime


class APIContextRecord(BaseModel):
    api_key_id: str
    org_id: str
    user_agent: str
    ip_address: str
    geo_region: str
    quota_used: int = 0
    source: str = "synthetic"


class SessionRecord(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    user_id: str
    session_started_at: datetime
    session_ended_at: datetime
    queries: list[QueryRecord]
    responses: list[ResponseRecord]
    api_context: APIContextRecord
    label: BehaviorClass | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class FeatureValue(BaseModel):
    name: str
    group: str
    value: float
    provenance: FeatureProvenance
    description: str


class ScoreExplanation(BaseModel):
    reason: str
    contribution: float


class TriggeredRule(BaseModel):
    id: str
    description: str
    effect: str


class AssessmentExplainability(BaseModel):
    risk_score: float
    predicted_class: BehaviorClass
    confidence: float
    category_scores: dict[str, float] = Field(default_factory=dict)
    risk_reducers: list[str] = Field(default_factory=list)
    top_reasons: list[str] = Field(default_factory=list)
    triggered_rules: list[TriggeredRule] = Field(default_factory=list)


class RiskAssessmentResult(BaseModel):
    session_id: str
    risk_score: float
    predicted_class: BehaviorClass
    confidence: float
    explainability: AssessmentExplainability
    category_scores: dict[str, float] = Field(default_factory=dict)
    top_reasons: list[str] = Field(default_factory=list)
    triggered_rules: list[TriggeredRule] = Field(default_factory=list)
    risk_reducers: list[str] = Field(default_factory=list)
    reasons: list[ScoreExplanation] = Field(default_factory=list)
    feature_values: list[FeatureValue]


class PolicyDecisionResult(BaseModel):
    session_id: str
    chosen_policy: OutputPolicy
    policy_reason: str
    override_applied: bool = False
    thresholds: dict[str, float] = Field(default_factory=dict)


class TransformationResult(BaseModel):
    session_id: str
    policy: OutputPolicy
    raw_output: str
    transformed_output: str
    leakage_proxy_score: float
    leakage_factors: dict[str, float]


class SimulationRequest(BaseModel):
    num_users: int = 20
    sessions_per_user: int = 3
    seed: int = 7
    persist: bool = True


class EvaluateRequest(BaseModel):
    seed: int = 11
    num_users: int = 40
    sessions_per_user: int = 3


class IngestSessionRequest(BaseModel):
    session: SessionRecord


class ExperimentMetrics(BaseModel):
    scenario_count: int | None = None
    mean_risk_score: float | None = None
    leakage_proxy_reduction_mean: float | None = None
    utility_preservation_mean: float | None = None
    false_block_rate_normal: float | None = None
    false_block_rate_laboratory_legitimate: float | None = None
    adaptive_degradation_rate_high_risk: float | None = None
    reason_coverage: float | None = None
    threshold_sanity_checks: dict[str, float] = Field(default_factory=dict)
    policy_distribution_by_class: dict[str, dict[str, int]] = Field(default_factory=dict)
    leakage_proxy_reduction_by_class: dict[str, float] = Field(default_factory=dict)
    utility_preservation_by_class: dict[str, float] = Field(default_factory=dict)


class ExperimentSummary(BaseModel):
    experiment_id: str
    created_at: datetime
    metrics: ExperimentMetrics
    artifact_paths: dict[str, str]

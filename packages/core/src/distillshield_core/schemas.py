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


class RiskAssessmentResult(BaseModel):
    session_id: str
    risk_score: float
    predicted_class: BehaviorClass
    confidence: float
    reasons: list[ScoreExplanation]
    feature_values: list[FeatureValue]
    model_contributions: dict[str, float] = Field(default_factory=dict)


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


class TrainingRequest(BaseModel):
    seed: int = 7
    num_users: int = 60
    sessions_per_user: int = 4


class EvaluateRequest(BaseModel):
    seed: int = 11
    num_users: int = 40
    sessions_per_user: int = 3


class IngestSessionRequest(BaseModel):
    session: SessionRecord


class ModelSummary(BaseModel):
    name: str
    available: bool
    version: str
    metrics: dict[str, float] = Field(default_factory=dict)


class ExperimentMetrics(BaseModel):
    accuracy: float | None = None
    precision: float | None = None
    recall: float | None = None
    f1: float | None = None
    utility_mean: float | None = None
    leakage_proxy_mean: float | None = None
    confusion_matrix: list[list[float]] | None = None
    class_report: dict[str, Any] | None = None


class ExperimentSummary(BaseModel):
    experiment_id: str
    created_at: datetime
    metrics: ExperimentMetrics
    artifact_paths: dict[str, str]

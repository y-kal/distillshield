from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy import JSON, Column, Text
from sqlmodel import Field, SQLModel


class User(SQLModel, table=True):
    id: str = Field(primary_key=True)
    label: str | None = None
    created_at: datetime
    user_metadata: dict[str, Any] = Field(default_factory=dict, sa_column=Column("metadata", JSON))


class SessionEntity(SQLModel, table=True):
    id: str = Field(primary_key=True)
    user_id: str = Field(index=True)
    started_at: datetime
    ended_at: datetime
    label: str | None = Field(default=None, index=True)
    session_metadata: dict[str, Any] = Field(default_factory=dict, sa_column=Column("metadata", JSON))


class QueryEntity(SQLModel, table=True):
    id: str = Field(primary_key=True)
    session_id: str = Field(index=True)
    text: str = Field(sa_column=Column(Text))
    timestamp: datetime
    max_tokens: int = 512
    streamed: bool = False


class ResponseEntity(SQLModel, table=True):
    id: str = Field(primary_key=True)
    session_id: str = Field(index=True)
    query_id: str = Field(index=True)
    raw_output: str = Field(sa_column=Column(Text))
    completion_tokens: int = 0
    truncated: bool = False
    refusal: bool = False
    created_at: datetime


class APIContextEntity(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    session_id: str = Field(index=True)
    api_key_id: str
    org_id: str = Field(index=True)
    user_agent: str
    ip_address: str
    geo_region: str
    quota_used: int = 0
    source: str = "synthetic"


class FeatureVectorEntity(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    session_id: str = Field(index=True)
    feature_payload: dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSON))
    created_at: datetime


class RiskAssessmentEntity(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    session_id: str = Field(index=True)
    risk_score: float
    predicted_class: str = Field(index=True)
    confidence: float
    reasons: list[dict[str, Any]] = Field(default_factory=list, sa_column=Column(JSON))
    model_contributions: dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSON))
    created_at: datetime


class PolicyDecisionEntity(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    session_id: str = Field(index=True)
    chosen_policy: str = Field(index=True)
    policy_reason: str
    override_applied: bool = False
    thresholds: dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSON))
    created_at: datetime


class TransformationArtifactEntity(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    session_id: str = Field(index=True)
    policy: str = Field(index=True)
    raw_output: str = Field(sa_column=Column(Text))
    transformed_output: str = Field(sa_column=Column(Text))
    leakage_proxy_score: float
    leakage_factors: dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSON))
    created_at: datetime


class ExperimentRunEntity(SQLModel, table=True):
    id: str = Field(primary_key=True)
    created_at: datetime
    metrics: dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSON))
    artifact_paths: dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSON))
    notes: str | None = None


class ModelVersionEntity(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    name: str = Field(index=True)
    version: str
    artifact_path: str
    metrics: dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSON))
    created_at: datetime

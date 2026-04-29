from __future__ import annotations

from distillshield_core.config import get_settings
from distillshield_core.schemas import RiskAssessmentResult, SessionRecord
from distillshield_feature_pipeline.pipeline import FeaturePipeline

from .rule_based import RuleBasedRiskEngine


class DistillShieldEngine:
    def __init__(self) -> None:
        self.settings = get_settings()
        self.feature_pipeline = FeaturePipeline()
        self.rule_engine = RuleBasedRiskEngine()

    def assess(self, session: SessionRecord) -> RiskAssessmentResult:
        feature_values = self.feature_pipeline.extract(session)
        trusted_lab = session.api_context.org_id in self.settings.trusted_lab_orgs
        return self.rule_engine.score(session.id, feature_values, trusted_lab=trusted_lab)

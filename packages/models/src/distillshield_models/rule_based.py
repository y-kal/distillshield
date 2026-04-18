from __future__ import annotations

from distillshield_core.enums import BehaviorClass
from distillshield_core.schemas import FeatureValue, RiskAssessmentResult, ScoreExplanation


RULE_WEIGHTS = {
    "consecutive_query_similarity": 0.10,
    "prompt_template_fingerprint_score": 0.12,
    "knowledge_domain_sweep_score": 0.08,
    "turn_dependency_score": -0.10,
    "followup_naturalness_proxy": -0.12,
    "session_restart_frequency": 0.08,
    "burst_regularity_proxy": 0.06,
    "retry_on_refusal_rate": 0.12,
    "max_token_utilisation_rate": 0.08,
    "truncation_stop_indicator": 0.06,
    "response_to_next_query_latency": -0.03,
    "key_rotation_frequency": 0.09,
    "geographic_implausibility_proxy": 0.07,
    "org_quota_burn_rate": 0.10,
    "reference_to_prior_response_rate": -0.08,
    "emotional_personal_context_presence": -0.10,
    "question_form_rate": -0.03,
}


class RuleBasedScorer:
    def score(self, session_id: str, feature_values: list[FeatureValue]) -> RiskAssessmentResult:
        feature_map = {feature.name: feature.value for feature in feature_values}
        reasons: list[ScoreExplanation] = []
        weighted_score = 0.35

        for name, weight in RULE_WEIGHTS.items():
            value = feature_map.get(name, 0.0)
            normalized = self._normalize(name, value)
            contribution = normalized * weight
            weighted_score += contribution
            if abs(contribution) >= 0.04:
                direction = "increased" if contribution > 0 else "reduced"
                reasons.append(ScoreExplanation(reason=f"{name} {direction} risk", contribution=round(contribution, 4)))

        risk_score = max(0.0, min(weighted_score, 1.0))
        predicted_class = self._class_from_score(risk_score)
        confidence = min(0.99, 0.55 + abs(risk_score - 0.5))
        reasons = sorted(reasons, key=lambda item: abs(item.contribution), reverse=True)[:5]
        return RiskAssessmentResult(
            session_id=session_id,
            risk_score=risk_score,
            predicted_class=predicted_class,
            confidence=confidence,
            reasons=reasons,
            feature_values=feature_values,
            model_contributions={"rule_based": risk_score},
        )

    def _normalize(self, name: str, value: float) -> float:
        if name in {"inter_query_time_mean", "response_to_next_query_latency", "session_length_query_ratio", "query_length_variance"}:
            return min(value / 300.0, 1.0)
        if name in {"org_quota_burn_rate", "knowledge_domain_sweep_score", "domain_coverage_breadth", "instruction_verb_diversity"}:
            return min(value / 4.0, 1.0)
        return min(value, 1.0)

    def _class_from_score(self, score: float) -> BehaviorClass:
        if score < 0.30:
            return BehaviorClass.NORMAL
        if score < 0.48:
            return BehaviorClass.LABORATORY_LEGITIMATE
        if score < 0.75:
            return BehaviorClass.SUSPICIOUS
        return BehaviorClass.HIGH_THREAT

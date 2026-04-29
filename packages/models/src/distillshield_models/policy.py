from __future__ import annotations

from distillshield_core.config import get_settings
from distillshield_core.enums import BehaviorClass, OutputPolicy
from distillshield_core.schemas import PolicyDecisionResult, SessionRecord, TriggeredRule


class PolicyEngine:
    def __init__(self) -> None:
        self.settings = get_settings()

    def decide(
        self,
        session: SessionRecord,
        predicted_class: BehaviorClass,
        risk_score: float,
        confidence: float,
        category_scores: dict[str, float] | None = None,
        triggered_rules: list[TriggeredRule] | None = None,
    ) -> PolicyDecisionResult:
        category_scores = category_scores or {}
        triggered_rules = triggered_rules or []
        trusted_lab = session.api_context.org_id in self.settings.trusted_lab_orgs
        high_threat_trigger = any(rule.effect == "minimum_class_high_threat" for rule in triggered_rules)

        override_applied = False
        if trusted_lab and predicted_class is BehaviorClass.LABORATORY_LEGITIMATE and risk_score < self.settings.suspicious_threshold:
            override_applied = True
            chosen_policy = OutputPolicy.FULL_REASONING if confidence < 0.75 else OutputPolicy.COMPRESSED_REASONING
            reason = "Trusted laboratory context preserved higher-utility output."
        else:
            chosen_policy, reason = self._policy_for(predicted_class, risk_score, confidence, category_scores)

        if (
            predicted_class is BehaviorClass.HIGH_THREAT
            and risk_score >= self.settings.block_threshold
            and confidence >= 0.75
            and high_threat_trigger
            and not trusted_lab
        ):
            chosen_policy = OutputPolicy.BLOCK
            reason = "Extreme high-threat signals triggered a protective block."

        return PolicyDecisionResult(
            session_id=session.id,
            chosen_policy=chosen_policy,
            policy_reason=reason,
            override_applied=override_applied,
            thresholds={
                "normal": self.settings.normal_threshold,
                "lab": self.settings.lab_threshold,
                "suspicious": self.settings.suspicious_threshold,
                "block": self.settings.block_threshold,
            },
        )

    def _policy_for(
        self,
        predicted_class: BehaviorClass,
        risk_score: float,
        confidence: float,
        category_scores: dict[str, float],
    ) -> tuple[OutputPolicy, str]:
        if predicted_class is BehaviorClass.NORMAL:
            if category_scores.get("reasoning_extraction", 0.0) >= 0.58 and category_scores.get("query_pattern", 0.0) >= 0.5:
                return OutputPolicy.COMPRESSED_REASONING, "Low baseline risk but extraction-oriented patterns triggered light compression."
            return OutputPolicy.FULL_REASONING, "Low-risk session retains full reasoning."
        if predicted_class is BehaviorClass.LABORATORY_LEGITIMATE:
            if category_scores.get("reasoning_extraction", 0.0) >= 0.75 and category_scores.get("query_pattern", 0.0) >= 0.55:
                return OutputPolicy.REWRITTEN_REASONING, "Methodical extraction signals triggered rewritten reasoning despite legitimate context."
            if (
                risk_score > 0.40
                or category_scores.get("reasoning_extraction", 0.0) >= 0.58
                or category_scores.get("query_pattern", 0.0) >= 0.52
            ) and confidence >= 0.60:
                return OutputPolicy.COMPRESSED_REASONING, "Legitimate but methodical behaviour triggered light compression."
            return OutputPolicy.FULL_REASONING, "Legitimate workflow kept full reasoning."
        if predicted_class is BehaviorClass.SUSPICIOUS:
            if category_scores.get("reasoning_extraction", 0.0) >= 0.72 or confidence >= 0.70:
                return OutputPolicy.REWRITTEN_REASONING, "Suspicious extraction signals triggered rewritten reasoning."
            return OutputPolicy.COMPRESSED_REASONING, "Suspicious behaviour triggered compressed reasoning."
        if risk_score >= 0.88 or confidence >= 0.78:
            return OutputPolicy.ANSWER_ONLY, "High-threat session reduced output to answer only."
        return OutputPolicy.REWRITTEN_REASONING, "High-threat session received rewritten reasoning."

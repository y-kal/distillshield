from __future__ import annotations

from distillshield_core.config import get_settings
from distillshield_core.enums import BehaviorClass, OutputPolicy
from distillshield_core.schemas import PolicyDecisionResult, SessionRecord


class PolicyEngine:
    def __init__(self) -> None:
        self.settings = get_settings()

    def decide(self, session: SessionRecord, predicted_class: BehaviorClass, risk_score: float, confidence: float) -> PolicyDecisionResult:
        override_applied = False
        if session.api_context.org_id in self.settings.trusted_lab_orgs and predicted_class is not BehaviorClass.HIGH_THREAT:
            override_applied = True
            chosen_policy = OutputPolicy.FULL_REASONING if confidence >= 0.55 else OutputPolicy.COMPRESSED_REASONING
            reason = "Trusted laboratory override applied."
        else:
            chosen_policy, reason = self._policy_for(predicted_class, risk_score, confidence)

        if risk_score >= self.settings.block_threshold and predicted_class is BehaviorClass.HIGH_THREAT:
            chosen_policy = OutputPolicy.BLOCK
            reason = "Extreme risk and high-threat confidence triggered block."

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

    def _policy_for(self, predicted_class: BehaviorClass, risk_score: float, confidence: float) -> tuple[OutputPolicy, str]:
        if predicted_class is BehaviorClass.NORMAL:
            return OutputPolicy.FULL_REASONING, "Low-risk session retains full reasoning."
        if predicted_class is BehaviorClass.LABORATORY_LEGITIMATE:
            if risk_score > self.settings.lab_threshold and confidence > 0.65:
                return OutputPolicy.COMPRESSED_REASONING, "Legitimate but methodical behavior triggered mild compression."
            return OutputPolicy.FULL_REASONING, "Legitimate laboratory behavior allowed full reasoning."
        if predicted_class is BehaviorClass.SUSPICIOUS:
            return OutputPolicy.COMPRESSED_REASONING, "Suspicious harvesting pattern triggered reasoning compression."
        if confidence > 0.75 or risk_score > self.settings.suspicious_threshold:
            return OutputPolicy.REWRITTEN_REASONING, "High-threat activity triggered rewritten reasoning."
        return OutputPolicy.ANSWER_ONLY, "High-threat uncertainty reduced output to answer only."

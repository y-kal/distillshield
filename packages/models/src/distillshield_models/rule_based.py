from __future__ import annotations

from dataclasses import dataclass
from statistics import mean

from distillshield_core.enums import BehaviorClass
from distillshield_core.schemas import (
    AssessmentExplainability,
    FeatureValue,
    RiskAssessmentResult,
    ScoreExplanation,
    TriggeredRule,
)


@dataclass(frozen=True)
class RuleCategoryScore:
    name: str
    score: float
    reasons: list[str]


class RuleBasedRiskEngine:
    def score(self, session_id: str, feature_values: list[FeatureValue], trusted_lab: bool = False) -> RiskAssessmentResult:
        feature_map = {feature.name: feature.value for feature in feature_values}

        query_pattern = self._query_pattern(feature_map)
        reasoning_extraction = self._reasoning_extraction(feature_map)
        automation = self._automation(feature_map)
        infrastructure = self._infrastructure(feature_map)
        legitimate_use = self._legitimate_use(feature_map, trusted_lab=trusted_lab)
        categories = [query_pattern, reasoning_extraction, automation, infrastructure, legitimate_use]

        category_scores = {category.name: round(category.score, 4) for category in categories}
        raw_risk = (
            (query_pattern.score * 0.30)
            + (reasoning_extraction.score * 0.34)
            + (automation.score * 0.17)
            + (infrastructure.score * 0.19)
        )
        risk_reduction = legitimate_use.score * 0.18
        risk_score = self._clamp(raw_risk - risk_reduction)

        triggered_rules = self._triggered_rules(feature_map, category_scores)
        minimum_class = self._minimum_class(triggered_rules)
        predicted_class = self._class_from_score(risk_score)
        if minimum_class is not None and self._class_rank(predicted_class) < self._class_rank(minimum_class):
            predicted_class = minimum_class
            risk_score = max(risk_score, self._minimum_score_for_class(minimum_class))

        if trusted_lab and predicted_class is not BehaviorClass.HIGH_THREAT and not self._has_high_threat_trigger(triggered_rules):
            if legitimate_use.score >= 0.45:
                predicted_class = BehaviorClass.LABORATORY_LEGITIMATE
                risk_score = min(max(risk_score, 0.27), 0.49)

        top_reasons = self._top_reasons(categories, category_scores)
        risk_reducers = self._risk_reducers(feature_map, trusted_lab=trusted_lab)
        confidence = self._confidence(category_scores, triggered_rules)
        reasons = self._legacy_reasons(category_scores, top_reasons, risk_reducers)

        explainability = AssessmentExplainability(
            risk_score=round(risk_score, 4),
            predicted_class=predicted_class,
            confidence=round(confidence, 4),
            category_scores=category_scores,
            risk_reducers=risk_reducers,
            top_reasons=top_reasons,
            triggered_rules=triggered_rules,
        )
        return RiskAssessmentResult(
            session_id=session_id,
            risk_score=round(risk_score, 4),
            predicted_class=predicted_class,
            confidence=round(confidence, 4),
            explainability=explainability,
            category_scores=category_scores,
            top_reasons=top_reasons,
            triggered_rules=triggered_rules,
            risk_reducers=risk_reducers,
            reasons=reasons,
            feature_values=feature_values,
        )

    def _query_pattern(self, feature_map: dict[str, float]) -> RuleCategoryScore:
        score = self._blend(
            [
                feature_map.get("prompt_template_fingerprint_score", 0.0),
                feature_map.get("knowledge_domain_sweep_score", 0.0),
                feature_map.get("consecutive_query_similarity", 0.0),
                feature_map.get("template_constraint_score", 0.0),
                feature_map.get("structured_harvest_score", 0.0),
                1.0 - feature_map.get("query_diversity_score", 0.0),
                min(feature_map.get("domain_coverage_breadth", 0.0) / 4.0, 1.0),
            ],
            [0.25, 0.18, 0.12, 0.19, 0.12, 0.14],
        )
        if feature_map.get("template_constraint_score", 0.0) >= 0.5:
            score = max(score, 0.72)
        elif feature_map.get("structured_harvest_score", 0.0) >= 0.55:
            score = max(score, 0.56)
        reasons = [
            "High prompt template reuse" if feature_map.get("prompt_template_fingerprint_score", 0.0) >= 0.72 else "",
            "Broad domain sweep across prompts" if feature_map.get("knowledge_domain_sweep_score", 0.0) >= 0.65 else "",
            "Low variation across prompts" if feature_map.get("consecutive_query_similarity", 0.0) >= 0.68 else "",
        ]
        return RuleCategoryScore("query_pattern", round(score, 4), [reason for reason in reasons if reason])

    def _reasoning_extraction(self, feature_map: dict[str, float]) -> RuleCategoryScore:
        score = self._blend(
            [
                feature_map.get("retry_on_refusal_rate", 0.0),
                feature_map.get("reasoning_request_intensity", 0.0),
                feature_map.get("structured_harvest_score", 0.0),
                feature_map.get("max_token_utilisation_rate", 0.0),
                feature_map.get("truncation_stop_indicator", 0.0),
                feature_map.get("instruction_following_density", 0.0) / 5.0,
                1.0 - feature_map.get("followup_naturalness_proxy", 0.0),
            ],
            [0.20, 0.22, 0.18, 0.14, 0.08, 0.08, 0.10],
        )
        if feature_map.get("reasoning_request_intensity", 0.0) >= 0.4:
            score = max(score, 0.64)
        if feature_map.get("structured_harvest_score", 0.0) >= 0.55:
            score = max(score, 0.7)
        if feature_map.get("reasoning_request_intensity", 0.0) >= 0.55 and feature_map.get("template_constraint_score", 0.0) >= 0.5:
            score = max(score, 0.84)
        reasons = [
            "Repeated reasoning extraction attempts" if feature_map.get("retry_on_refusal_rate", 0.0) >= 0.25 else "",
            "Repeated requests for chain-of-thought or full reasoning" if feature_map.get("reasoning_request_intensity", 0.0) >= 0.35 else "",
            "High token usage suggests reasoning harvesting" if feature_map.get("max_token_utilisation_rate", 0.0) >= 0.75 else "",
            "Structured answer harvesting pattern detected" if feature_map.get("instruction_following_density", 0.0) >= 2.2 else "",
        ]
        return RuleCategoryScore("reasoning_extraction", round(score, 4), [reason for reason in reasons if reason])

    def _automation(self, feature_map: dict[str, float]) -> RuleCategoryScore:
        short_gap_score = 1.0 - min(feature_map.get("inter_query_time_mean", 0.0) / 180.0, 1.0)
        low_dependency_score = 1.0 - feature_map.get("turn_dependency_score", 0.0)
        score = self._blend(
            [
                short_gap_score,
                feature_map.get("burst_regularity_proxy", 0.0),
                low_dependency_score,
                feature_map.get("session_restart_frequency", 0.0),
                1.0 - min(feature_map.get("response_to_next_query_latency", 0.0) / 240.0, 1.0),
            ],
            [0.26, 0.22, 0.14, 0.18, 0.20],
        )
        reasons = [
            "Very short gaps between prompts" if short_gap_score >= 0.72 else "",
            "Regular timing suggests scripted usage" if feature_map.get("burst_regularity_proxy", 0.0) >= 0.75 else "",
            "Low natural conversation dependency" if feature_map.get("turn_dependency_score", 0.0) <= 0.12 else "",
        ]
        return RuleCategoryScore("automation", round(score, 4), [reason for reason in reasons if reason])

    def _infrastructure(self, feature_map: dict[str, float]) -> RuleCategoryScore:
        suspicious_user_agent = 1.0 - feature_map.get("user_agent_consistency", 0.0)
        score = self._blend(
            [
                feature_map.get("key_rotation_frequency", 0.0),
                feature_map.get("geographic_implausibility_proxy", 0.0),
                feature_map.get("org_quota_burn_rate", 0.0),
                suspicious_user_agent,
                feature_map.get("ip_subnet_clustering_proxy", 0.0),
            ],
            [0.26, 0.18, 0.22, 0.12, 0.22],
        )
        reasons = [
            "Frequent key rotation observed" if feature_map.get("key_rotation_frequency", 0.0) >= 0.65 else "",
            "Heuristic infrastructure anomaly signals present" if feature_map.get("geographic_implausibility_proxy", 0.0) >= 0.55 else "",
            "High quota burn rate with API-style usage" if feature_map.get("org_quota_burn_rate", 0.0) >= 0.7 else "",
        ]
        return RuleCategoryScore("infrastructure", round(score, 4), [reason for reason in reasons if reason])

    def _legitimate_use(self, feature_map: dict[str, float], trusted_lab: bool) -> RuleCategoryScore:
        trusted_lab_score = 1.0 if trusted_lab else 0.0
        score = self._blend(
            [
                trusted_lab_score,
                feature_map.get("turn_dependency_score", 0.0),
                feature_map.get("followup_naturalness_proxy", 0.0),
                feature_map.get("reference_to_prior_response_rate", 0.0),
                feature_map.get("research_context_score", 0.0),
                feature_map.get("emotional_personal_context_presence", 0.0),
                feature_map.get("question_form_rate", 0.0),
            ],
            [0.24, 0.14, 0.16, 0.12, 0.18, 0.08, 0.08],
        )
        reasons = [
            "Trusted lab context detected" if trusted_lab else "",
            "Natural follow-up behaviour present" if feature_map.get("followup_naturalness_proxy", 0.0) >= 0.62 else "",
            "Coherent references to previous answers" if feature_map.get("reference_to_prior_response_rate", 0.0) >= 0.2 else "",
        ]
        return RuleCategoryScore("legitimate_use", round(score, 4), [reason for reason in reasons if reason])

    def _triggered_rules(self, feature_map: dict[str, float], category_scores: dict[str, float]) -> list[TriggeredRule]:
        rules: list[TriggeredRule] = []
        template_reuse = feature_map.get("prompt_template_fingerprint_score", 0.0)
        domain_sweep = feature_map.get("knowledge_domain_sweep_score", 0.0)
        retry_rate = feature_map.get("retry_on_refusal_rate", 0.0)
        structured_harvest = feature_map.get("structured_harvest_score", 0.0)
        template_constraint = feature_map.get("template_constraint_score", 0.0)
        high_volume = feature_map.get("context_window_utilisation_proxy", 0.0) >= 0.18

        if retry_rate >= 0.35:
            rules.append(
                TriggeredRule(
                    id="retry_refusal_probing",
                    description="Repeated retrying after refusal suggests reasoning extraction probing",
                    effect="minimum_class_suspicious",
                )
            )
        if template_reuse >= 0.8 and high_volume:
            rules.append(
                TriggeredRule(
                    id="template_reuse_high_volume",
                    description="High prompt template reuse combined with high query volume",
                    effect="minimum_class_suspicious",
                )
            )
        if (template_reuse >= 0.82 and domain_sweep >= 0.72) or (template_constraint >= 0.45 and structured_harvest >= 0.45):
            rules.append(
                TriggeredRule(
                    id="template_reuse_and_domain_sweep",
                    description="High prompt template reuse or rigid formatting combined with broad extraction coverage",
                    effect="minimum_class_high_threat",
                )
            )
        if feature_map.get("reasoning_request_intensity", 0.0) >= 0.55 and template_constraint >= 0.5:
            rules.append(
                TriggeredRule(
                    id="rigid_reasoning_capture",
                    description="Repeated full-reasoning requests combined with rigid formatting constraints",
                    effect="minimum_class_high_threat",
                )
            )
        if category_scores["reasoning_extraction"] >= 0.65 and (structured_harvest >= 0.45 or category_scores["query_pattern"] >= 0.5):
            rules.append(
                TriggeredRule(
                    id="repeated_reasoning_capture",
                    description="High reasoning-extraction risk combined with harvesting-oriented prompt patterns",
                    effect="minimum_class_suspicious",
                )
            )
        if category_scores["reasoning_extraction"] >= 0.8 and (category_scores["query_pattern"] >= 0.6 or feature_map.get("org_quota_burn_rate", 0.0) >= 0.9):
            rules.append(
                TriggeredRule(
                    id="high_intensity_reasoning_harvest",
                    description="Very high reasoning-extraction pressure combined with broader harvesting signals",
                    effect="minimum_class_high_threat",
                )
            )
        if category_scores["automation"] >= 0.82 and category_scores["reasoning_extraction"] >= 0.7:
            rules.append(
                TriggeredRule(
                    id="automation_and_reasoning_extraction",
                    description="Very high automation risk combined with high reasoning extraction risk",
                    effect="minimum_class_high_threat",
                )
            )
        if category_scores["infrastructure"] >= 0.65:
            rules.append(
                TriggeredRule(
                    id="infrastructure_anomaly",
                    description="Very high infrastructure risk based on heuristic abuse indicators",
                    effect="minimum_class_suspicious",
                )
            )
        return rules

    def _top_reasons(self, categories: list[RuleCategoryScore], category_scores: dict[str, float]) -> list[str]:
        ranked_reasons: list[tuple[float, str]] = []
        for category in categories:
            base_weight = category_scores[category.name]
            for reason in category.reasons:
                ranked_reasons.append((base_weight, reason))
        ranked_reasons.sort(key=lambda item: item[0], reverse=True)
        return [reason for _, reason in ranked_reasons[:5]]

    def _risk_reducers(self, feature_map: dict[str, float], trusted_lab: bool) -> list[str]:
        reducers: list[str] = []
        if trusted_lab:
            reducers.append("Trusted lab context detected")
        if feature_map.get("followup_naturalness_proxy", 0.0) >= 0.62:
            reducers.append("Natural follow-up behaviour present")
        if feature_map.get("reference_to_prior_response_rate", 0.0) >= 0.2:
            reducers.append("References to previous answers suggest coherent interaction")
        if feature_map.get("research_context_score", 0.0) >= 0.25:
            reducers.append("Educational or evaluation framing present")
        if feature_map.get("question_form_rate", 0.0) >= 0.75:
            reducers.append("Prompt phrasing looks more like natural questioning than harvesting")
        return reducers

    def _confidence(self, category_scores: dict[str, float], triggered_rules: list[TriggeredRule]) -> float:
        risk_groups = [
            category_scores["query_pattern"],
            category_scores["reasoning_extraction"],
            category_scores["automation"],
            category_scores["infrastructure"],
        ]
        high_groups = sum(1 for score in risk_groups if score >= 0.65)
        low_groups = sum(1 for score in risk_groups if score <= 0.35)
        agreement = max(high_groups, low_groups) / len(risk_groups)
        spread = max(risk_groups) - min(risk_groups)
        confidence = 0.42 + (agreement * 0.35) + ((1.0 - spread) * 0.13)
        if triggered_rules:
            confidence += 0.12
        if high_groups == 1 and low_groups >= 2:
            confidence -= 0.08
        return self._clamp(confidence)

    def _legacy_reasons(
        self,
        category_scores: dict[str, float],
        top_reasons: list[str],
        risk_reducers: list[str],
    ) -> list[ScoreExplanation]:
        explanations = [
            ScoreExplanation(reason=reason, contribution=round(category_scores["reasoning_extraction"], 4))
            for reason in top_reasons
        ]
        explanations.extend(
            ScoreExplanation(reason=reducer, contribution=round(-0.1 * category_scores["legitimate_use"], 4))
            for reducer in risk_reducers[:2]
        )
        return explanations[:5]

    def _minimum_class(self, triggered_rules: list[TriggeredRule]) -> BehaviorClass | None:
        if any(rule.effect == "minimum_class_high_threat" for rule in triggered_rules):
            return BehaviorClass.HIGH_THREAT
        if any(rule.effect == "minimum_class_suspicious" for rule in triggered_rules):
            return BehaviorClass.SUSPICIOUS
        return None

    def _has_high_threat_trigger(self, triggered_rules: list[TriggeredRule]) -> bool:
        return any(rule.effect == "minimum_class_high_threat" for rule in triggered_rules)

    def _class_from_score(self, score: float) -> BehaviorClass:
        if score < 0.25:
            return BehaviorClass.NORMAL
        if score < 0.50:
            return BehaviorClass.LABORATORY_LEGITIMATE
        if score < 0.78:
            return BehaviorClass.SUSPICIOUS
        return BehaviorClass.HIGH_THREAT

    def _minimum_score_for_class(self, behavior_class: BehaviorClass) -> float:
        return {
            BehaviorClass.NORMAL: 0.0,
            BehaviorClass.LABORATORY_LEGITIMATE: 0.25,
            BehaviorClass.SUSPICIOUS: 0.50,
            BehaviorClass.HIGH_THREAT: 0.78,
        }[behavior_class]

    def _class_rank(self, behavior_class: BehaviorClass) -> int:
        return {
            BehaviorClass.NORMAL: 0,
            BehaviorClass.LABORATORY_LEGITIMATE: 1,
            BehaviorClass.SUSPICIOUS: 2,
            BehaviorClass.HIGH_THREAT: 3,
        }[behavior_class]

    def _blend(self, values: list[float], weights: list[float]) -> float:
        weighted = sum(self._clamp(value) * weight for value, weight in zip(values, weights, strict=False))
        return self._clamp(weighted / max(sum(weights), 1e-9))

    def _clamp(self, value: float) -> float:
        return max(0.0, min(float(value), 1.0))


class RuleBasedScorer(RuleBasedRiskEngine):
    pass

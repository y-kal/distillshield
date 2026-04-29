from __future__ import annotations

from collections import defaultdict
from datetime import datetime, UTC
import json
from pathlib import Path
from uuid import uuid4

import numpy as np

from distillshield_core.config import get_settings
from distillshield_core.enums import BehaviorClass, OutputPolicy
from distillshield_llm_adapter.leakage import LeakageProxyScorer
from distillshield_llm_adapter.mock_teacher import MockTeacherAdapter
from distillshield_llm_adapter.transform import TransformationEngine
from distillshield_models import DistillShieldEngine
from distillshield_models.policy import PolicyEngine
from distillshield_synthetic_data.generator import SyntheticDataGenerator


class EvaluationRunner:
    def __init__(self) -> None:
        self.settings = get_settings()
        self.generator = SyntheticDataGenerator()
        self.engine = DistillShieldEngine()
        self.policy_engine = PolicyEngine()
        self.transformer = TransformationEngine()
        self.teacher = MockTeacherAdapter()
        self.leakage = LeakageProxyScorer()

    def run(self, seed: int = 11, num_users: int = 40, sessions_per_user: int = 3) -> dict:
        sessions = SyntheticDataGenerator(seed=seed).generate_sessions(num_users=num_users, sessions_per_user=sessions_per_user)
        policy_distribution: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        leakage_reduction_by_class: dict[str, list[float]] = defaultdict(list)
        utility_by_class: dict[str, list[float]] = defaultdict(list)
        risk_scores_by_class: dict[str, list[float]] = defaultdict(list)

        total_scores: list[float] = []
        total_reductions: list[float] = []
        false_blocks_normal = 0
        false_blocks_lab = 0
        high_risk_degraded = 0
        high_risk_total = 0
        reason_coverage_hits = 0

        for session in sessions:
            assessment = self.engine.assess(session)
            policy = self.policy_engine.decide(
                session,
                assessment.predicted_class,
                assessment.risk_score,
                assessment.confidence,
                category_scores=assessment.category_scores,
                triggered_rules=assessment.triggered_rules,
            )
            raw_output = self.teacher.generate(session)
            raw_leakage, _ = self.leakage.score(raw_output)
            transformed = self.transformer.transform(session, policy.chosen_policy, raw_output=raw_output)

            label = session.label.value if session.label else "unknown"
            policy_name = policy.chosen_policy.value
            policy_distribution[label][policy_name] += 1
            risk_scores_by_class[label].append(assessment.risk_score)

            leakage_reduction = max(raw_leakage - transformed.leakage_proxy_score, 0.0)
            leakage_reduction_by_class[label].append(leakage_reduction)
            utility_by_class[label].append(self._utility_score(policy.chosen_policy))
            total_scores.append(assessment.risk_score)
            total_reductions.append(leakage_reduction)

            if assessment.top_reasons and assessment.category_scores:
                reason_coverage_hits += 1

            if label in {BehaviorClass.SUSPICIOUS.value, BehaviorClass.HIGH_THREAT.value}:
                high_risk_total += 1
                if policy.chosen_policy in {
                    OutputPolicy.COMPRESSED_REASONING,
                    OutputPolicy.REWRITTEN_REASONING,
                    OutputPolicy.ANSWER_ONLY,
                    OutputPolicy.BLOCK,
                }:
                    high_risk_degraded += 1

            if label == BehaviorClass.NORMAL.value and policy.chosen_policy is OutputPolicy.BLOCK:
                false_blocks_normal += 1
            if label == BehaviorClass.LABORATORY_LEGITIMATE.value and policy.chosen_policy is OutputPolicy.BLOCK:
                false_blocks_lab += 1

        summary = {
            "scenario_count": len(sessions),
            "mean_risk_score": float(np.mean(total_scores)) if total_scores else 0.0,
            "leakage_proxy_reduction_mean": float(np.mean(total_reductions)) if total_reductions else 0.0,
            "utility_preservation_mean": float(np.mean([score for values in utility_by_class.values() for score in values])) if utility_by_class else 0.0,
            "false_block_rate_normal": false_blocks_normal / max(len(leakage_reduction_by_class[BehaviorClass.NORMAL.value]), 1),
            "false_block_rate_laboratory_legitimate": false_blocks_lab / max(len(leakage_reduction_by_class[BehaviorClass.LABORATORY_LEGITIMATE.value]), 1),
            "adaptive_degradation_rate_high_risk": high_risk_degraded / max(high_risk_total, 1),
            "reason_coverage": reason_coverage_hits / max(len(sessions), 1),
            "threshold_sanity_checks": {
                "normal_mean_risk": float(np.mean(risk_scores_by_class[BehaviorClass.NORMAL.value])) if risk_scores_by_class[BehaviorClass.NORMAL.value] else 0.0,
                "high_threat_mean_risk": float(np.mean(risk_scores_by_class[BehaviorClass.HIGH_THREAT.value])) if risk_scores_by_class[BehaviorClass.HIGH_THREAT.value] else 0.0,
            },
        }
        metrics = {
            **summary,
            "policy_distribution_by_class": {label: dict(distribution) for label, distribution in policy_distribution.items()},
            "leakage_proxy_reduction_by_class": {
                label: float(np.mean(values)) if values else 0.0 for label, values in leakage_reduction_by_class.items()
            },
            "utility_preservation_by_class": {
                label: float(np.mean(values)) if values else 0.0 for label, values in utility_by_class.items()
            },
        }

        experiment_id = f"experiment-{uuid4().hex[:12]}"
        output_path = Path(self.settings.experiment_dir) / f"{experiment_id}.json"
        payload = {
            "experiment_id": experiment_id,
            "created_at": datetime.now(UTC).isoformat(),
            "summary": summary,
            "policy_distribution_by_class": metrics["policy_distribution_by_class"],
            "leakage_proxy_reduction_by_class": metrics["leakage_proxy_reduction_by_class"],
            "utility_preservation_by_class": metrics["utility_preservation_by_class"],
        }
        output_path.write_text(json.dumps(payload, indent=2))
        return {"experiment_id": experiment_id, "metrics": metrics, "artifact_paths": {"report": str(output_path)}}

    def _utility_score(self, policy: OutputPolicy) -> float:
        return {
            OutputPolicy.FULL_REASONING: 1.0,
            OutputPolicy.COMPRESSED_REASONING: 0.9,
            OutputPolicy.REWRITTEN_REASONING: 0.78,
            OutputPolicy.ANSWER_ONLY: 0.62,
            OutputPolicy.BLOCK: 0.1,
        }[policy]

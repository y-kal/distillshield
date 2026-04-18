from __future__ import annotations

import pandas as pd

from distillshield_core.enums import BehaviorClass
from distillshield_core.schemas import RiskAssessmentResult, ScoreExplanation, SessionRecord
from distillshield_feature_pipeline.pipeline import FeaturePipeline
from distillshield_models.ml import BaselineModelTrainer
from distillshield_models.optional import GraphModelStub, SequenceModelStub
from distillshield_models.rule_based import RuleBasedScorer


class DistillShieldEngine:
    def __init__(self) -> None:
        self.feature_pipeline = FeaturePipeline()
        self.rule_scorer = RuleBasedScorer()
        self.model_trainer = BaselineModelTrainer()
        self.sequence_stub = SequenceModelStub()
        self.graph_stub = GraphModelStub()

    def assess(self, session: SessionRecord, artifact_paths: dict[str, str] | None = None, graph=None) -> RiskAssessmentResult:
        feature_values = self.feature_pipeline.extract(session)
        rule_result = self.rule_scorer.score(session.id, feature_values)
        contributions = dict(rule_result.model_contributions)
        final_score = rule_result.risk_score
        predicted_class = rule_result.predicted_class
        confidence = rule_result.confidence
        reasons = list(rule_result.reasons)

        frame = pd.DataFrame([{feature.name: feature.value for feature in feature_values}])
        if artifact_paths:
            ml_scores = []
            class_votes = []
            for model_name, artifact_path in artifact_paths.items():
                if not artifact_path:
                    continue
                prediction = self.model_trainer.predict(artifact_path, frame)
                model_score = self._risk_from_prediction(prediction["predicted_class"], prediction["confidence"])
                ml_scores.append(model_score)
                class_votes.append(prediction["predicted_class"])
                contributions[model_name] = model_score
            if ml_scores:
                final_score = (0.45 * rule_result.risk_score) + (0.55 * sum(ml_scores) / len(ml_scores))
                predicted_class = self._majority_class(class_votes, fallback=rule_result.predicted_class.value)
                confidence = min(0.99, (rule_result.confidence + max(ml_scores)) / 2)

        sequence_output = self.sequence_stub.score(session)
        if sequence_output.available:
            contributions["sequence_stub"] = sequence_output.score
            final_score = min(1.0, final_score * 0.9 + sequence_output.score * 0.1)
        if graph is not None:
            graph_output = self.graph_stub.score(session, graph)
            if graph_output.available:
                contributions["graph_stub"] = graph_output.score
                final_score = min(1.0, final_score * 0.9 + graph_output.score * 0.1)

        predicted_class = self._class_from_score(final_score, predicted_class)
        reasons.append(ScoreExplanation(reason="Ensemble blended rule and model signals", contribution=round(final_score, 4)))
        return RiskAssessmentResult(
            session_id=session.id,
            risk_score=final_score,
            predicted_class=predicted_class,
            confidence=confidence,
            reasons=sorted(reasons, key=lambda item: abs(item.contribution), reverse=True)[:5],
            feature_values=feature_values,
            model_contributions=contributions,
        )

    def _risk_from_prediction(self, predicted_class: str, confidence: float) -> float:
        base = {
            BehaviorClass.NORMAL.value: 0.18,
            BehaviorClass.LABORATORY_LEGITIMATE.value: 0.35,
            BehaviorClass.SUSPICIOUS.value: 0.65,
            BehaviorClass.HIGH_THREAT.value: 0.88,
        }[predicted_class]
        return min(1.0, base * 0.7 + confidence * 0.3)

    def _class_from_score(self, score: float, fallback: BehaviorClass | str) -> BehaviorClass:
        if score < 0.30:
            return BehaviorClass.NORMAL
        if score < 0.48:
            return BehaviorClass.LABORATORY_LEGITIMATE
        if score < 0.75:
            return BehaviorClass.SUSPICIOUS
        return BehaviorClass.HIGH_THREAT

    def _majority_class(self, class_votes: list[str], fallback: str) -> BehaviorClass:
        if not class_votes:
            return BehaviorClass(fallback)
        counts = {vote: class_votes.count(vote) for vote in set(class_votes)}
        return BehaviorClass(max(counts, key=counts.get))

from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
from uuid import uuid4

import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score

from distillshield_core.config import get_settings
from distillshield_feature_pipeline.pipeline import FeaturePipeline
from distillshield_llm_adapter.transform import TransformationEngine
from distillshield_models.ensemble import DistillShieldEngine
from distillshield_models.ml import BaselineModelTrainer
from distillshield_models.policy import PolicyEngine
from distillshield_synthetic_data.generator import SyntheticDataGenerator


class EvaluationRunner:
    def __init__(self) -> None:
        self.settings = get_settings()
        self.generator = SyntheticDataGenerator()
        self.feature_pipeline = FeaturePipeline()
        self.trainer = BaselineModelTrainer()
        self.engine = DistillShieldEngine()
        self.policy_engine = PolicyEngine()
        self.transformer = TransformationEngine()

    def run(self, seed: int = 11, num_users: int = 40, sessions_per_user: int = 3) -> dict:
        generator = SyntheticDataGenerator(seed=seed)
        splits = generator.dataset_splits(num_users=num_users, sessions_per_user=sessions_per_user)
        train_rows = [self.feature_pipeline.to_frame_row(session) for session in splits["train"]]
        artifacts = self.trainer.train_all(train_rows, seed=seed)
        model_paths = {artifact.name: artifact.artifact_path for artifact in artifacts if artifact.available}

        y_true = []
        y_pred = []
        utility_scores = []
        leakage_scores = []

        for session in splits["test"]:
            assessment = self.engine.assess(session, artifact_paths=model_paths)
            policy = self.policy_engine.decide(session, assessment.predicted_class, assessment.risk_score, assessment.confidence)
            transformed = self.transformer.transform(session, policy.chosen_policy)
            y_true.append(session.label.value)
            y_pred.append(assessment.predicted_class.value)
            utility_scores.append(1.0 if policy.chosen_policy.value in {"full_reasoning", "compressed_reasoning"} else 0.7)
            leakage_scores.append(transformed.leakage_proxy_score)

        metrics = {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "precision": float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
            "f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
            "utility_mean": float(np.mean(utility_scores)),
            "leakage_proxy_mean": float(np.mean(leakage_scores)),
            "confusion_matrix": confusion_matrix(y_true, y_pred, labels=sorted(set(y_true))).tolist(),
            "class_report": classification_report(y_true, y_pred, output_dict=True, zero_division=0),
        }
        experiment_id = f"experiment-{uuid4().hex[:12]}"
        output_path = Path(self.settings.experiment_dir) / f"{experiment_id}.json"
        output_path.write_text(json.dumps({"experiment_id": experiment_id, "created_at": datetime.utcnow().isoformat(), "metrics": metrics}, indent=2))
        return {"experiment_id": experiment_id, "metrics": metrics, "artifact_paths": {"report": str(output_path)}}

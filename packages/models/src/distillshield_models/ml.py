from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from distillshield_core.config import get_settings
from distillshield_core.enums import BehaviorClass

try:
    from lightgbm import LGBMClassifier
except Exception:  # pragma: no cover
    LGBMClassifier = None


LABEL_TO_INDEX = {label.value: idx for idx, label in enumerate(BehaviorClass)}
INDEX_TO_LABEL = {idx: label for label, idx in LABEL_TO_INDEX.items()}


@dataclass
class TrainedModelArtifact:
    name: str
    artifact_path: str
    metrics: dict[str, float]
    feature_importance: dict[str, float]
    available: bool = True


class BaselineModelTrainer:
    def __init__(self, model_dir: Path | None = None) -> None:
        self.settings = get_settings()
        self.model_dir = model_dir or self.settings.model_dir
        self.model_dir.mkdir(parents=True, exist_ok=True)

    def prepare_dataframe(self, rows: list[dict[str, Any]]) -> pd.DataFrame:
        frame = pd.DataFrame(rows)
        feature_columns = [column for column in frame.columns if column not in {"label", "session_id"}]
        frame[feature_columns] = frame[feature_columns].fillna(0.0)
        return frame

    def train_all(self, rows: list[dict[str, Any]], seed: int = 7) -> list[TrainedModelArtifact]:
        frame = self.prepare_dataframe(rows)
        X = frame.drop(columns=["label", "session_id"])
        y = frame["label"].map(LABEL_TO_INDEX)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=seed, stratify=y)

        artifacts = [
            self._train_logistic(X_train, X_test, y_train, y_test),
            self._train_random_forest(X_train, X_test, y_train, y_test, seed),
            self._train_lightgbm(X_train, X_test, y_train, y_test, seed),
        ]
        return artifacts

    def predict(self, artifact_path: str, frame: pd.DataFrame) -> dict[str, Any]:
        model = joblib.load(artifact_path)
        probabilities = model.predict_proba(frame)[0]
        predicted_index = int(np.argmax(probabilities))
        return {
            "predicted_class": INDEX_TO_LABEL[predicted_index],
            "confidence": float(np.max(probabilities)),
            "class_probabilities": {INDEX_TO_LABEL[idx]: float(prob) for idx, prob in enumerate(probabilities)},
        }

    def _train_logistic(self, X_train, X_test, y_train, y_test) -> TrainedModelArtifact:
        pipeline = Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("model", LogisticRegression(max_iter=2000)),
            ]
        )
        pipeline.fit(X_train, y_train)
        return self._finalize_artifact("logistic_regression", pipeline, X_test, y_test)

    def _train_random_forest(self, X_train, X_test, y_train, y_test, seed: int) -> TrainedModelArtifact:
        model = RandomForestClassifier(n_estimators=300, random_state=seed, class_weight="balanced")
        model.fit(X_train, y_train)
        return self._finalize_artifact("random_forest", model, X_test, y_test)

    def _train_lightgbm(self, X_train, X_test, y_train, y_test, seed: int) -> TrainedModelArtifact:
        if LGBMClassifier is None:
            return TrainedModelArtifact(name="lightgbm", artifact_path="", metrics={"available": 0.0}, feature_importance={}, available=False)
        model = LGBMClassifier(n_estimators=250, learning_rate=0.05, random_state=seed)
        model.fit(X_train, y_train)
        return self._finalize_artifact("lightgbm", model, X_test, y_test)

    def _finalize_artifact(self, name: str, model, X_test, y_test) -> TrainedModelArtifact:
        artifact_path = str(self.model_dir / f"{name}.joblib")
        joblib.dump(model, artifact_path)
        predictions = model.predict(X_test)
        probabilities = model.predict_proba(X_test)
        precision, recall, f1, _ = precision_recall_fscore_support(y_test, predictions, average="weighted", zero_division=0)
        metrics = {
            "accuracy": float(accuracy_score(y_test, predictions)),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
        }
        try:
            metrics["auroc"] = float(roc_auc_score(y_test, probabilities, multi_class="ovr"))
        except ValueError:
            metrics["auroc"] = 0.0

        if hasattr(model, "feature_importances_"):
            importance = dict(zip(X_test.columns, model.feature_importances_.tolist()))
        elif hasattr(model, "named_steps"):
            coefficients = model.named_steps["model"].coef_
            importance = dict(zip(X_test.columns, np.abs(coefficients).mean(axis=0).tolist()))
        else:
            importance = {}
        return TrainedModelArtifact(name=name, artifact_path=artifact_path, metrics=metrics, feature_importance=importance, available=True)

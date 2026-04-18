from datetime import UTC, datetime

from distillshield_core.schemas import ExperimentSummary


def test_experiment_summary_accepts_nested_metrics():
    summary = ExperimentSummary(
        experiment_id="experiment-test",
        created_at=datetime.now(UTC),
        metrics={
            "accuracy": 0.8,
            "f1": 0.75,
            "confusion_matrix": [[4, 1], [0, 3]],
            "class_report": {
                "normal": {"precision": 0.8, "recall": 1.0, "f1-score": 0.89, "support": 4.0},
                "suspicious": {"precision": 0.75, "recall": 0.75, "f1-score": 0.75, "support": 4.0},
                "accuracy": 0.8,
            },
        },
        artifact_paths={"report": "data/experiments/experiment-test.json"},
    )

    assert summary.metrics.confusion_matrix == [[4.0, 1.0], [0.0, 3.0]]
    assert summary.metrics.class_report["accuracy"] == 0.8

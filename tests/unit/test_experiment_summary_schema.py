from datetime import UTC, datetime

from distillshield_core.schemas import ExperimentSummary


def test_experiment_summary_accepts_nested_metrics():
    summary = ExperimentSummary(
        experiment_id="experiment-test",
        created_at=datetime.now(UTC),
        metrics={
            "scenario_count": 32,
            "mean_risk_score": 0.53,
            "policy_distribution_by_class": {
                "normal": {"full_reasoning": 8},
                "high_threat": {"answer_only": 5, "block": 1},
            },
        },
        artifact_paths={"report": "data/experiments/experiment-test.json"},
    )

    assert summary.metrics.scenario_count == 32
    assert summary.metrics.policy_distribution_by_class["high_threat"]["block"] == 1

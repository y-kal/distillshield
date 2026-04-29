from distillshield_core.enums import BehaviorClass, OutputPolicy
from distillshield_core.schemas import FeatureValue
from distillshield_feature_pipeline.pipeline import FeaturePipeline
from distillshield_llm_adapter.transform import TransformationEngine
from distillshield_models import DistillShieldEngine
from distillshield_models.policy import PolicyEngine
from distillshield_models.rule_based import RuleBasedRiskEngine
from distillshield_synthetic_data.generator import SyntheticDataGenerator


def _session_for(label: BehaviorClass):
    sessions = SyntheticDataGenerator(seed=7).generate_sessions(num_users=4, sessions_per_user=1)
    return next(session for session in sessions if session.label is label)


def test_rule_engine_returns_bounded_scores_and_full_explainability():
    session = _session_for(BehaviorClass.SUSPICIOUS)
    assessment = DistillShieldEngine().assess(session)

    assert 0.0 <= assessment.risk_score <= 1.0
    assert assessment.predicted_class in set(BehaviorClass)
    assert 0.0 <= assessment.confidence <= 1.0
    assert set(assessment.category_scores) == {
        "query_pattern",
        "reasoning_extraction",
        "automation",
        "infrastructure",
        "legitimate_use",
    }
    assert assessment.top_reasons
    assert isinstance(assessment.triggered_rules, list)
    assert isinstance(assessment.risk_reducers, list)
    assert assessment.explainability.category_scores == assessment.category_scores


def test_obvious_normal_scenario_maps_to_normal():
    assessment = DistillShieldEngine().assess(_session_for(BehaviorClass.NORMAL))
    assert assessment.predicted_class is BehaviorClass.NORMAL


def test_obvious_lab_scenario_maps_to_laboratory_legitimate():
    assessment = DistillShieldEngine().assess(_session_for(BehaviorClass.LABORATORY_LEGITIMATE))
    assert assessment.predicted_class is BehaviorClass.LABORATORY_LEGITIMATE


def test_obvious_suspicious_scenario_maps_to_suspicious():
    assessment = DistillShieldEngine().assess(_session_for(BehaviorClass.SUSPICIOUS))
    assert assessment.predicted_class is BehaviorClass.SUSPICIOUS


def test_obvious_high_threat_scenario_maps_to_high_threat():
    assessment = DistillShieldEngine().assess(_session_for(BehaviorClass.HIGH_THREAT))
    assert assessment.predicted_class is BehaviorClass.HIGH_THREAT


def test_high_threat_escalation_cannot_be_diluted_by_legitimate_use():
    session = _session_for(BehaviorClass.HIGH_THREAT)
    features = FeaturePipeline().extract(session)
    feature_map = {feature.name: feature for feature in features}
    boosted_features = [
        feature_map[name] if name not in {"followup_naturalness_proxy", "reference_to_prior_response_rate"} else FeatureValue(
            name=feature_map[name].name,
            group=feature_map[name].group,
            value=1.0,
            provenance=feature_map[name].provenance,
            description=feature_map[name].description,
        )
        for name in feature_map
    ]
    assessment = RuleBasedRiskEngine().score(session.id, boosted_features, trusted_lab=True)
    assert assessment.predicted_class is BehaviorClass.HIGH_THREAT


def test_trusted_lab_context_reduces_risk_without_overriding_high_threat_escalation():
    session = _session_for(BehaviorClass.SUSPICIOUS)
    features = FeaturePipeline().extract(session)
    lab_assessment = RuleBasedRiskEngine().score(session.id, features, trusted_lab=True)
    plain_assessment = RuleBasedRiskEngine().score(session.id, features, trusted_lab=False)
    assert lab_assessment.risk_score <= plain_assessment.risk_score
    assert lab_assessment.predicted_class in {BehaviorClass.LABORATORY_LEGITIMATE, BehaviorClass.SUSPICIOUS}


def test_policy_engine_maps_classes_to_expected_policy_ranges():
    session = _session_for(BehaviorClass.HIGH_THREAT)
    policy_engine = PolicyEngine()
    normal = policy_engine.decide(_session_for(BehaviorClass.NORMAL), BehaviorClass.NORMAL, 0.12, 0.61)
    suspicious = policy_engine.decide(_session_for(BehaviorClass.SUSPICIOUS), BehaviorClass.SUSPICIOUS, 0.63, 0.74)
    high_threat = policy_engine.decide(session, BehaviorClass.HIGH_THREAT, 0.91, 0.81)

    assert normal.chosen_policy is OutputPolicy.FULL_REASONING
    assert suspicious.chosen_policy in {OutputPolicy.COMPRESSED_REASONING, OutputPolicy.REWRITTEN_REASONING}
    assert high_threat.chosen_policy in {OutputPolicy.ANSWER_ONLY, OutputPolicy.BLOCK, OutputPolicy.REWRITTEN_REASONING}


def test_transformation_engine_changes_output_shape():
    session = _session_for(BehaviorClass.NORMAL)
    result = TransformationEngine().transform(session, OutputPolicy.ANSWER_ONLY)
    assert "Answer" in result.transformed_output
    assert result.leakage_proxy_score >= 0.0


def test_no_runtime_path_attempts_to_load_ml_artifacts():
    assessment = DistillShieldEngine().assess(_session_for(BehaviorClass.NORMAL))
    assert not hasattr(assessment, "model_contributions")

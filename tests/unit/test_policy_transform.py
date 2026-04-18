from distillshield_core.enums import BehaviorClass, OutputPolicy
from distillshield_llm_adapter.transform import TransformationEngine
from distillshield_models.policy import PolicyEngine
from distillshield_synthetic_data.generator import SyntheticDataGenerator


def test_policy_engine_maps_high_threat_to_protective_policy():
    session = SyntheticDataGenerator(seed=9).generate_sessions(num_users=4, sessions_per_user=1)[-1]
    decision = PolicyEngine().decide(session, BehaviorClass.HIGH_THREAT, 0.84, 0.83)
    assert decision.chosen_policy in {OutputPolicy.REWRITTEN_REASONING, OutputPolicy.ANSWER_ONLY, OutputPolicy.BLOCK}


def test_transformation_engine_changes_output_shape():
    session = SyntheticDataGenerator(seed=7).generate_sessions(num_users=1, sessions_per_user=1)[0]
    result = TransformationEngine().transform(session, OutputPolicy.ANSWER_ONLY)
    assert "Answer" in result.transformed_output
    assert result.leakage_proxy_score >= 0.0

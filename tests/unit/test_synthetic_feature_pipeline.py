from distillshield_feature_pipeline.pipeline import FeaturePipeline
from distillshield_synthetic_data.generator import SyntheticDataGenerator


def test_synthetic_generator_is_reproducible():
    first = SyntheticDataGenerator(seed=13).generate_sessions(num_users=4, sessions_per_user=1)
    second = SyntheticDataGenerator(seed=13).generate_sessions(num_users=4, sessions_per_user=1)
    assert [session.id for session in first] != [session.id for session in second]
    assert [session.label for session in first] == [session.label for session in second]
    assert [session.queries[0].text for session in first] == [session.queries[0].text for session in second]


def test_feature_pipeline_extracts_requested_features():
    session = SyntheticDataGenerator(seed=7).generate_sessions(num_users=1, sessions_per_user=1)[0]
    features = FeaturePipeline().extract(session)
    feature_names = {feature.name for feature in features}
    assert "consecutive_query_similarity" in feature_names
    assert "key_rotation_frequency" in feature_names
    assert "question_form_rate" in feature_names
    assert len(features) >= 30

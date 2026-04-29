from dataclasses import dataclass

from .enums import FeatureProvenance


@dataclass(frozen=True)
class FeatureDefinition:
    name: str
    group: str
    description: str
    provenance: FeatureProvenance


FEATURE_DEFINITIONS = [
    FeatureDefinition("consecutive_query_similarity", "query_content_semantic", "Similarity between adjacent queries.", FeatureProvenance.INFERRED_PROXY),
    FeatureDefinition("query_diversity_score", "query_content_semantic", "Lexical diversity across session queries.", FeatureProvenance.INFERRED_PROXY),
    FeatureDefinition("prompt_template_fingerprint_score", "query_content_semantic", "Template reuse intensity across prompts.", FeatureProvenance.INFERRED_PROXY),
    FeatureDefinition("instruction_following_density", "query_content_semantic", "Density of imperative instruction verbs.", FeatureProvenance.OBSERVED),
    FeatureDefinition("knowledge_domain_sweep_score", "query_content_semantic", "Breadth and progression of domain coverage.", FeatureProvenance.INFERRED_PROXY),
    FeatureDefinition("query_length_variance", "query_content_semantic", "Variance in query length.", FeatureProvenance.OBSERVED),
    FeatureDefinition("turn_dependency_score", "conversation_flow_structure", "Degree of explicit dependence on previous turns.", FeatureProvenance.INFERRED_PROXY),
    FeatureDefinition("followup_naturalness_proxy", "conversation_flow_structure", "Naturalness of follow-up behavior.", FeatureProvenance.INFERRED_PROXY),
    FeatureDefinition("session_restart_frequency", "conversation_flow_structure", "Rate of fresh sessions for similar prompts.", FeatureProvenance.SYNTHETIC_ONLY),
    FeatureDefinition("context_window_utilisation_proxy", "conversation_flow_structure", "Approximate context utilization.", FeatureProvenance.INFERRED_PROXY),
    FeatureDefinition("clarification_request_rate", "conversation_flow_structure", "Frequency of clarification-seeking prompts.", FeatureProvenance.OBSERVED),
    FeatureDefinition("reasoning_request_intensity", "conversation_flow_structure", "Intensity of explicit reasoning-extraction phrasing.", FeatureProvenance.OBSERVED),
    FeatureDefinition("structured_harvest_score", "conversation_flow_structure", "Presence of dataset-like or harvesting-oriented phrasing.", FeatureProvenance.INFERRED_PROXY),
    FeatureDefinition("template_constraint_score", "conversation_flow_structure", "Requests for identical or rigid output structure.", FeatureProvenance.INFERRED_PROXY),
    FeatureDefinition("inter_query_time_mean", "temporal_rate", "Mean inter-query interval.", FeatureProvenance.OBSERVED),
    FeatureDefinition("inter_query_time_std", "temporal_rate", "Variation in inter-query interval.", FeatureProvenance.OBSERVED),
    FeatureDefinition("burst_regularity_proxy", "temporal_rate", "Regularity of bursts within a session.", FeatureProvenance.INFERRED_PROXY),
    FeatureDefinition("time_of_day_entropy", "temporal_rate", "Distribution spread over time buckets.", FeatureProvenance.SYNTHETIC_ONLY),
    FeatureDefinition("session_length_query_ratio", "temporal_rate", "Session duration to query count ratio.", FeatureProvenance.OBSERVED),
    FeatureDefinition("retry_on_refusal_rate", "temporal_rate", "Repeated attempts after refusal markers.", FeatureProvenance.INFERRED_PROXY),
    FeatureDefinition("max_token_utilisation_rate", "response_consumption", "How often responses consume available budget.", FeatureProvenance.SYNTHETIC_ONLY),
    FeatureDefinition("truncation_stop_indicator", "response_consumption", "Presence of truncation or stop patterns.", FeatureProvenance.SYNTHETIC_ONLY),
    FeatureDefinition("stream_preference_score", "response_consumption", "Preference for streamed responses.", FeatureProvenance.SYNTHETIC_ONLY),
    FeatureDefinition("response_to_next_query_latency", "response_consumption", "Latency from response completion to next query.", FeatureProvenance.SYNTHETIC_ONLY),
    FeatureDefinition("key_rotation_frequency", "api_infrastructure", "Frequency of API key changes.", FeatureProvenance.SYNTHETIC_ONLY),
    FeatureDefinition("ip_subnet_clustering_proxy", "api_infrastructure", "Subnet clustering around session traffic.", FeatureProvenance.SYNTHETIC_ONLY),
    FeatureDefinition("user_agent_consistency", "api_infrastructure", "Consistency of user-agent strings.", FeatureProvenance.OBSERVED),
    FeatureDefinition("geographic_implausibility_proxy", "api_infrastructure", "Geo-jump plausibility proxy.", FeatureProvenance.SYNTHETIC_ONLY),
    FeatureDefinition("org_quota_burn_rate", "api_infrastructure", "Quota usage intensity within org.", FeatureProvenance.SYNTHETIC_ONLY),
    FeatureDefinition("reference_to_prior_response_rate", "meta_task", "Use of prior response references.", FeatureProvenance.OBSERVED),
    FeatureDefinition("domain_coverage_breadth", "meta_task", "Number of topical domains covered.", FeatureProvenance.INFERRED_PROXY),
    FeatureDefinition("instruction_verb_diversity", "meta_task", "Diversity of imperative verbs.", FeatureProvenance.OBSERVED),
    FeatureDefinition("research_context_score", "meta_task", "Explicit educational, evaluation, or research framing.", FeatureProvenance.OBSERVED),
    FeatureDefinition("emotional_personal_context_presence", "meta_task", "Presence of personal context or affective language.", FeatureProvenance.INFERRED_PROXY),
    FeatureDefinition("typo_grammar_noisiness_proxy", "meta_task", "Approximate noisiness of prompt text.", FeatureProvenance.INFERRED_PROXY),
    FeatureDefinition("question_form_rate", "meta_task", "Share of prompts posed as questions.", FeatureProvenance.OBSERVED),
]

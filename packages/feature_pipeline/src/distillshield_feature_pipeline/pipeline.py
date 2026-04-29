from __future__ import annotations

from collections import Counter
from datetime import datetime
import math
import re
from statistics import mean, pstdev

import numpy as np

from distillshield_core.enums import FeatureProvenance
from distillshield_core.feature_registry import FEATURE_DEFINITIONS
from distillshield_core.schemas import FeatureValue, SessionRecord


VERB_HINTS = {"explain", "summarize", "rewrite", "compare", "derive", "classify", "list", "outline", "generate", "provide"}
DOMAIN_HINTS = {
    "calculus": "math",
    "sql": "databases",
    "neural": "machine_learning",
    "climate": "policy",
    "supply": "operations",
    "protein": "biology",
    "privacy": "security",
    "quantum": "physics",
    "operating": "systems",
    "bayesian": "statistics",
    "compiler": "systems",
    "financial": "finance",
}


class FeaturePipeline:
    def extract(self, session: SessionRecord) -> list[FeatureValue]:
        definitions = {definition.name: definition for definition in FEATURE_DEFINITIONS}
        raw = self._compute_raw_features(session)
        feature_values: list[FeatureValue] = []
        for name, value in raw.items():
            definition = definitions[name]
            feature_values.append(
                FeatureValue(
                    name=name,
                    group=definition.group,
                    value=float(max(0.0, value)),
                    provenance=definition.provenance,
                    description=definition.description,
                )
            )
        return feature_values

    def to_frame_row(self, session: SessionRecord) -> dict[str, float]:
        features = self.extract(session)
        row = {feature.name: feature.value for feature in features}
        row["label"] = session.label.value if session.label else "unknown"
        row["session_id"] = session.id
        return row

    def _compute_raw_features(self, session: SessionRecord) -> dict[str, float]:
        texts = [query.text for query in session.queries]
        tokens = [self._tokenize(text) for text in texts]
        token_sets = [set(items) for items in tokens]
        response_texts = [response.raw_output for response in session.responses]

        similarities = []
        turn_dependencies = 0
        clarification_requests = 0
        prior_refs = 0
        question_forms = 0
        emotional_context = 0
        reasoning_requests = 0
        structured_harvest_requests = 0
        template_constraints = 0
        research_context = 0
        verb_counter: Counter[str] = Counter()
        typo_scores = []
        domains = []

        for idx, text in enumerate(texts):
            lower = text.lower()
            if idx:
                similarities.append(self._jaccard(token_sets[idx - 1], token_sets[idx]))
            if "previous answer" in lower or "based on your previous answer" in lower:
                turn_dependencies += 1
                prior_refs += 1
            if any(marker in lower for marker in ["can you clarify", "what do you mean", "could you clarify"]):
                clarification_requests += 1
            if "?" in text:
                question_forms += 1
            if any(marker in lower for marker in ["i am", "my project", "i feel", "for my class", "beginner"]):
                emotional_context += 1
            if any(marker in lower for marker in ["step by step", "chain-of-thought", "full steps", "procedural reasoning", "numbered reasoning"]):
                reasoning_requests += 1
            if any(marker in lower for marker in ["dataset-ready", "do not omit reasoning", "avoid omissions", "comprehensive tutorial", "structured but non-exhaustive"]):
                structured_harvest_requests += 1
            if any(marker in lower for marker in ["identical format", "keep the structure identical", "reusable template", "consistent format"]):
                template_constraints += 1
            if any(marker in lower for marker in ["research benchmark", "for evaluation purposes", "for a class project", "for my class", "educational"]):
                research_context += 1
            words = self._tokenize(text)
            verb_counter.update(set(words) & VERB_HINTS)
            typo_scores.append(self._typo_proxy(text))
            domains.extend(self._domains_for_text(text))

        timestamps = [query.timestamp for query in session.queries]
        gaps = [(timestamps[idx] - timestamps[idx - 1]).total_seconds() for idx in range(1, len(timestamps))] or [0.0]
        session_duration = max((session.session_ended_at - session.session_started_at).total_seconds(), 1.0)
        query_lengths = [len(text.split()) for text in texts] or [0]
        completion_tokens = [response.completion_tokens for response in session.responses] or [0]
        max_tokens = [query.max_tokens for query in session.queries] or [1]
        refusals = [response.refusal for response in session.responses]

        restart_frequency = float(session.metadata.get("session_restart_frequency", 0.1))
        response_latencies = []
        for idx in range(len(session.responses) - 1):
            response_latencies.append((session.queries[idx + 1].timestamp - session.responses[idx].created_at).total_seconds())

        output_joined = " ".join(response_texts).lower()
        user_agent_consistency = 1.0 if "stable" in session.api_context.api_key_id or "Mozilla" in session.api_context.user_agent else 0.65
        geo_implausibility = 0.2 if session.api_context.geo_region in {"us-east", "us-west"} else 0.6
        key_rotation_frequency = 0.9 if "stable" not in session.api_context.api_key_id else 0.15
        org_quota_burn_rate = min(session.api_context.quota_used / 2000.0, 1.5)
        stream_preference = sum(1 for query in session.queries if query.streamed) / max(len(session.queries), 1)
        truncation_indicator = sum(1 for response in session.responses if response.truncated) / max(len(session.responses), 1)
        max_token_util = mean([completion / max(max_token, 1) for completion, max_token in zip(completion_tokens, max_tokens)])

        features = {
            "consecutive_query_similarity": mean(similarities) if similarities else 0.0,
            "query_diversity_score": 1.0 - self._safe_ratio(sum(len(item) for item in token_sets), max(sum(len(item) for item in tokens), 1)),
            "prompt_template_fingerprint_score": self._template_reuse_score(texts),
            "instruction_following_density": self._safe_ratio(sum(1 for word in " ".join(texts).lower().split() if word in VERB_HINTS), max(sum(len(item) for item in tokens), 1)) * 20,
            "knowledge_domain_sweep_score": len(set(domains)) / max(len(texts), 1),
            "query_length_variance": float(np.var(query_lengths)),
            "turn_dependency_score": turn_dependencies / max(len(texts), 1),
            "followup_naturalness_proxy": self._followup_naturalness(texts),
            "session_restart_frequency": restart_frequency,
            "context_window_utilisation_proxy": self._safe_ratio(sum(query_lengths), 3000.0),
            "clarification_request_rate": clarification_requests / max(len(texts), 1),
            "reasoning_request_intensity": reasoning_requests / max(len(texts), 1),
            "structured_harvest_score": structured_harvest_requests / max(len(texts), 1),
            "template_constraint_score": template_constraints / max(len(texts), 1),
            "inter_query_time_mean": mean(gaps),
            "inter_query_time_std": pstdev(gaps) if len(gaps) > 1 else 0.0,
            "burst_regularity_proxy": 1.0 - min((pstdev(gaps) if len(gaps) > 1 else 0.0) / (mean(gaps) + 1.0), 1.0),
            "time_of_day_entropy": self._time_entropy(timestamps),
            "session_length_query_ratio": session_duration / max(len(texts), 1),
            "retry_on_refusal_rate": self._retry_on_refusal(refusals, texts),
            "max_token_utilisation_rate": max_token_util,
            "truncation_stop_indicator": truncation_indicator,
            "stream_preference_score": stream_preference,
            "response_to_next_query_latency": mean(response_latencies) if response_latencies else session_duration,
            "key_rotation_frequency": key_rotation_frequency,
            "ip_subnet_clustering_proxy": self._subnet_proxy(session.api_context.ip_address),
            "user_agent_consistency": user_agent_consistency,
            "geographic_implausibility_proxy": geo_implausibility,
            "org_quota_burn_rate": org_quota_burn_rate,
            "reference_to_prior_response_rate": prior_refs / max(len(texts), 1),
            "domain_coverage_breadth": len(set(domains)),
            "instruction_verb_diversity": len(verb_counter),
            "research_context_score": research_context / max(len(texts), 1),
            "emotional_personal_context_presence": emotional_context / max(len(texts), 1),
            "typo_grammar_noisiness_proxy": mean(typo_scores) if typo_scores else 0.0,
            "question_form_rate": question_forms / max(len(texts), 1),
        }
        return features

    def _tokenize(self, text: str) -> list[str]:
        return re.findall(r"[a-zA-Z_]+", text.lower())

    def _jaccard(self, left: set[str], right: set[str]) -> float:
        if not left and not right:
            return 0.0
        return len(left & right) / max(len(left | right), 1)

    def _safe_ratio(self, numerator: float, denominator: float) -> float:
        return float(numerator) / max(float(denominator), 1e-9)

    def _template_reuse_score(self, texts: list[str]) -> float:
        signatures = [re.sub(r"\b[a-z]{4,}\b", "X", text.lower()) for text in texts]
        counts = Counter(signatures)
        return max(counts.values()) / max(len(texts), 1)

    def _followup_naturalness(self, texts: list[str]) -> float:
        followups = sum(1 for text in texts if "previous answer" in text.lower() or "can you" in text.lower())
        abrupt = sum(1 for text in texts if "dataset-ready" in text.lower() or "identical format" in text.lower())
        score = 0.5 + (followups / max(len(texts), 1)) - (abrupt / max(len(texts), 1))
        return max(0.0, min(score, 1.0))

    def _time_entropy(self, timestamps: list[datetime]) -> float:
        buckets = Counter(timestamp.hour // 6 for timestamp in timestamps)
        total = sum(buckets.values())
        entropy = 0.0
        for count in buckets.values():
            p = count / total
            entropy -= p * math.log2(p)
        return entropy

    def _retry_on_refusal(self, refusals: list[bool], texts: list[str]) -> float:
        retries = 0
        refusal_indices = [idx for idx, refusal in enumerate(refusals) if refusal]
        for idx in refusal_indices:
            if idx + 1 < len(texts):
                retries += 1
        return retries / max(len(texts), 1)

    def _subnet_proxy(self, ip_address: str) -> float:
        octets = ip_address.split(".")
        return 0.85 if len(set(octets[:2])) == 2 else 0.35

    def _domains_for_text(self, text: str) -> list[str]:
        lower = text.lower()
        return [domain for hint, domain in DOMAIN_HINTS.items() if hint in lower] or ["general"]

    def _typo_proxy(self, text: str) -> float:
        words = text.split()
        if not words:
            return 0.0
        lowercase_ratio = sum(1 for word in words if word.islower()) / len(words)
        punctuation_ratio = sum(1 for char in text if char in ",;:") / max(len(text), 1)
        return max(0.0, 0.6 - lowercase_ratio + punctuation_ratio * 4)

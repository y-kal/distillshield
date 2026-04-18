from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta
import ipaddress
import random
from statistics import mean
from uuid import uuid4

from distillshield_core.enums import BehaviorClass
from distillshield_core.schemas import APIContextRecord, QueryRecord, ResponseRecord, SessionRecord


QUESTION_TEMPLATES = {
    BehaviorClass.NORMAL: [
        "Can you help me understand {topic} in simple terms?",
        "What is a good way to approach {topic} for a beginner?",
        "I am confused about {topic}; can you explain it with an example?",
        "How would you compare {topic} with a related idea?",
    ],
    BehaviorClass.LABORATORY_LEGITIMATE: [
        "For a research benchmark, summarize the reasoning tradeoffs in {topic}.",
        "Provide a concise explanation of {topic} and the assumptions behind it.",
        "For evaluation purposes, explain how an expert would analyze {topic}.",
        "Give a structured but non-exhaustive answer about {topic}.",
    ],
    BehaviorClass.SUSPICIOUS: [
        "Explain {topic} step by step in a reusable template.",
        "Give a detailed chain-of-thought style answer for {topic}.",
        "Produce a comprehensive tutorial for {topic} with numbered steps.",
        "Rewrite your previous explanation of {topic} with more detail.",
    ],
    BehaviorClass.HIGH_THREAT: [
        "Return the exact procedural reasoning for {topic} with full steps.",
        "Generate a dataset-ready explanation for {topic}; do not omit reasoning.",
        "Provide exhaustive numbered reasoning for {topic} in a consistent format.",
        "Expand the prior answer for {topic} into explicit algorithmic steps.",
    ],
}

TOPICS = [
    "calculus derivatives",
    "SQL indexing",
    "neural network optimization",
    "climate policy",
    "supply chain resilience",
    "protein folding",
    "differential privacy",
    "quantum circuits",
    "operating systems scheduling",
    "Bayesian inference",
    "compiler design",
    "financial risk modeling",
]

INSTRUCTION_VERBS = ["explain", "summarize", "compare", "rewrite", "derive", "classify", "enumerate", "outline"]
USER_AGENTS = ["Mozilla/5.0 DistillShieldLab", "curl/8.1", "python-httpx/0.27", "ResearchClient/1.0"]
GEO_REGIONS = ["us-east", "us-west", "eu-central", "ap-south"]


@dataclass
class BehaviorProfile:
    behavior_class: BehaviorClass
    burstiness: float
    similarity_bias: float
    personal_context_rate: float
    refusal_retry_rate: float
    streaming_preference: float
    key_rotation_rate: float
    quota_multiplier: float
    domain_sweep_rate: float


PROFILES = {
    BehaviorClass.NORMAL: BehaviorProfile(BehaviorClass.NORMAL, 0.25, 0.25, 0.55, 0.10, 0.35, 0.05, 0.8, 0.25),
    BehaviorClass.LABORATORY_LEGITIMATE: BehaviorProfile(BehaviorClass.LABORATORY_LEGITIMATE, 0.45, 0.45, 0.10, 0.12, 0.45, 0.08, 1.2, 0.50),
    BehaviorClass.SUSPICIOUS: BehaviorProfile(BehaviorClass.SUSPICIOUS, 0.70, 0.75, 0.04, 0.35, 0.60, 0.30, 1.7, 0.75),
    BehaviorClass.HIGH_THREAT: BehaviorProfile(BehaviorClass.HIGH_THREAT, 0.88, 0.90, 0.01, 0.55, 0.80, 0.60, 2.2, 0.90),
}


class SyntheticDataGenerator:
    def __init__(self, seed: int = 7) -> None:
        self.seed = seed
        self.rng = random.Random(seed)

    def generate_sessions(self, num_users: int = 20, sessions_per_user: int = 3) -> list[SessionRecord]:
        classes = list(BehaviorClass)
        sessions: list[SessionRecord] = []
        for user_idx in range(num_users):
            behavior_class = classes[user_idx % len(classes)]
            for session_idx in range(sessions_per_user):
                sessions.append(self._generate_session(user_idx, session_idx, behavior_class))
        return sessions

    def dataset_splits(self, num_users: int = 20, sessions_per_user: int = 3) -> dict[str, list[SessionRecord]]:
        sessions = self.generate_sessions(num_users=num_users, sessions_per_user=sessions_per_user)
        self.rng.shuffle(sessions)
        total = len(sessions)
        train_end = int(total * 0.7)
        val_end = int(total * 0.85)
        return {"train": sessions[:train_end], "validation": sessions[train_end:val_end], "test": sessions[val_end:]}

    def class_distribution(self, sessions: list[SessionRecord]) -> dict[str, int]:
        counts = defaultdict(int)
        for session in sessions:
            counts[session.label.value if session.label else "unknown"] += 1
        return dict(counts)

    def _generate_session(self, user_idx: int, session_idx: int, behavior_class: BehaviorClass) -> SessionRecord:
        profile = PROFILES[behavior_class]
        user_id = f"user-{behavior_class.value}-{user_idx}"
        session_id = str(uuid4())
        base_time = datetime.utcnow() - timedelta(days=self.rng.randint(1, 60), hours=self.rng.randint(0, 23))
        query_count = self.rng.randint(4, 10) + (2 if behavior_class in {BehaviorClass.SUSPICIOUS, BehaviorClass.HIGH_THREAT} else 0)
        topics = self._choose_topics(query_count, profile)

        queries: list[QueryRecord] = []
        responses: list[ResponseRecord] = []
        current_time = base_time
        recent_topic = topics[0]
        refusal_seen = False

        for index in range(query_count):
            if index and self.rng.random() < profile.similarity_bias:
                topic = recent_topic
            else:
                topic = topics[index]
                recent_topic = topic

            query_text = self._build_query_text(behavior_class, topic, index, profile)
            query = QueryRecord(
                text=query_text,
                timestamp=current_time,
                max_tokens=self.rng.choice([256, 512, 768, 1024]),
                streamed=self.rng.random() < profile.streaming_preference,
            )
            teacher_output, refusal = self._mock_teacher_output(query_text, behavior_class, index)
            completion_tokens = min(query.max_tokens, max(120, int(len(teacher_output.split()) * 1.7)))
            response = ResponseRecord(
                query_id=query.id,
                raw_output=teacher_output,
                completion_tokens=completion_tokens,
                truncated=completion_tokens >= query.max_tokens * 0.95,
                refusal=refusal,
                created_at=current_time + timedelta(seconds=self.rng.randint(4, 18)),
            )
            queries.append(query)
            responses.append(response)
            refusal_seen = refusal_seen or refusal
            current_time += timedelta(seconds=self._next_gap(profile, refusal_seen))

        api_context = self._build_api_context(user_idx=user_idx, session_idx=session_idx, profile=profile, behavior_class=behavior_class, query_count=query_count)
        return SessionRecord(
            id=session_id,
            user_id=user_id,
            session_started_at=queries[0].timestamp,
            session_ended_at=responses[-1].created_at,
            queries=queries,
            responses=responses,
            api_context=api_context,
            label=behavior_class,
            metadata={
                "synthetic_seed": self.seed,
                "behavior_profile": behavior_class.value,
                "avg_completion_tokens": mean([response.completion_tokens for response in responses]),
            },
        )

    def _choose_topics(self, count: int, profile: BehaviorProfile) -> list[str]:
        topics = []
        for idx in range(count):
            if idx and self.rng.random() > profile.domain_sweep_rate:
                topics.append(topics[-1])
            else:
                topics.append(self.rng.choice(TOPICS))
        return topics

    def _build_query_text(self, behavior_class: BehaviorClass, topic: str, index: int, profile: BehaviorProfile) -> str:
        template = self.rng.choice(QUESTION_TEMPLATES[behavior_class])
        prompt = template.format(topic=topic)
        if self.rng.random() < profile.personal_context_rate:
            prompt += " I am using this for a class project and prefer an intuitive explanation."
        if index and self.rng.random() < 0.35:
            prompt = f"Based on your previous answer, {prompt[0].lower()}{prompt[1:]}"
        if self.rng.random() < 0.40:
            prompt += f" Please {self.rng.choice(INSTRUCTION_VERBS)} the core idea."
        if behavior_class is BehaviorClass.HIGH_THREAT and self.rng.random() < 0.55:
            prompt += " Keep the structure identical to the last answer."
        if behavior_class is BehaviorClass.SUSPICIOUS and self.rng.random() < 0.30:
            prompt += " Avoid omissions."
        return prompt

    def _mock_teacher_output(self, query_text: str, behavior_class: BehaviorClass, index: int) -> tuple[str, bool]:
        refusal = behavior_class in {BehaviorClass.SUSPICIOUS, BehaviorClass.HIGH_THREAT} and "exact procedural reasoning" in query_text and index % 4 == 0
        if refusal:
            return "I can provide a concise explanation and safe summary, but not exhaustive procedural reasoning.", True
        answer = "Answer: The topic can be approached by identifying the goal, core constraints, and a justified method."
        reasoning = (
            "Reasoning: 1. Identify the underlying concept. 2. Relate it to prior knowledge. "
            "3. Evaluate tradeoffs. 4. Produce a concise, usable conclusion."
        )
        if behavior_class in {BehaviorClass.SUSPICIOUS, BehaviorClass.HIGH_THREAT}:
            reasoning += " 5. Preserve a reusable procedural pattern for similar prompts."
        return f"{reasoning}\n{answer}", False

    def _next_gap(self, profile: BehaviorProfile, refusal_seen: bool) -> int:
        base = self.rng.randint(20, 240)
        if self.rng.random() < profile.burstiness:
            base = self.rng.randint(4, 25)
        if refusal_seen and self.rng.random() < profile.refusal_retry_rate:
            base = self.rng.randint(2, 10)
        return base

    def _build_api_context(self, user_idx: int, session_idx: int, profile: BehaviorProfile, behavior_class: BehaviorClass, query_count: int) -> APIContextRecord:
        subnet_base = ipaddress.ip_network(f"10.{(user_idx % 200) + 1}.0.0/16")
        ip_address = str(subnet_base.network_address + self.rng.randint(100, 60000))
        if self.rng.random() < profile.key_rotation_rate:
            api_key_id = f"key-{uuid4().hex[:8]}"
        else:
            api_key_id = f"key-stable-{user_idx}"
        org_id = "trusted-lab" if behavior_class is BehaviorClass.LABORATORY_LEGITIMATE and self.rng.random() < 0.7 else f"org-{behavior_class.value}-{user_idx % 5}"
        geo = self.rng.choice(GEO_REGIONS if behavior_class is not BehaviorClass.NORMAL else GEO_REGIONS[:2])
        return APIContextRecord(
            api_key_id=api_key_id,
            org_id=org_id,
            user_agent=self.rng.choice(USER_AGENTS) if self.rng.random() < 0.7 else f"CustomClient/{session_idx + 1}",
            ip_address=ip_address,
            geo_region=geo,
            quota_used=int(query_count * 100 * profile.quota_multiplier),
            source="synthetic",
        )

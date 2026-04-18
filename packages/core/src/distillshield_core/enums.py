from enum import Enum


class BehaviorClass(str, Enum):
    NORMAL = "normal"
    LABORATORY_LEGITIMATE = "laboratory_legitimate"
    SUSPICIOUS = "suspicious"
    HIGH_THREAT = "high_threat"


class OutputPolicy(str, Enum):
    FULL_REASONING = "full_reasoning"
    COMPRESSED_REASONING = "compressed_reasoning"
    REWRITTEN_REASONING = "rewritten_reasoning"
    ANSWER_ONLY = "answer_only"
    BLOCK = "block"


class FeatureProvenance(str, Enum):
    OBSERVED = "observed"
    INFERRED_PROXY = "inferred_proxy"
    SYNTHETIC_ONLY = "synthetic_only"

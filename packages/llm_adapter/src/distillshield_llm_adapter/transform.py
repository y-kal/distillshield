from __future__ import annotations

import re

from distillshield_core.enums import OutputPolicy
from distillshield_core.schemas import SessionRecord, TransformationResult

from .leakage import LeakageProxyScorer
from .mock_teacher import MockTeacherAdapter


class TransformationEngine:
    def __init__(self) -> None:
        self.teacher = MockTeacherAdapter()
        self.leakage = LeakageProxyScorer()

    def transform(self, session: SessionRecord, policy: OutputPolicy, raw_output: str | None = None) -> TransformationResult:
        teacher_output = raw_output or self.teacher.generate(session)
        transformed = {
            OutputPolicy.FULL_REASONING: self._full_reasoning,
            OutputPolicy.COMPRESSED_REASONING: self._compressed_reasoning,
            OutputPolicy.REWRITTEN_REASONING: self._rewritten_reasoning,
            OutputPolicy.ANSWER_ONLY: self._answer_only,
            OutputPolicy.BLOCK: self._block,
        }[policy](teacher_output)
        leakage_score, leakage_factors = self.leakage.score(transformed)
        return TransformationResult(
            session_id=session.id,
            policy=policy,
            raw_output=teacher_output,
            transformed_output=transformed,
            leakage_proxy_score=leakage_score,
            leakage_factors=leakage_factors,
        )

    def _full_reasoning(self, output: str) -> str:
        return output

    def _compressed_reasoning(self, output: str) -> str:
        answer = self._extract_answer(output)
        return f"High-level explanation: The response is summarized to preserve the key concept and main tradeoff.\n\n{answer}"

    def _rewritten_reasoning(self, output: str) -> str:
        answer = self._extract_answer(output)
        abstracted = re.sub(r"(?m)^\d+\.\s*", "- ", output)
        abstracted = re.sub(r"Reasoning:\s*", "Abstract rationale:\n", abstracted)
        abstracted = re.sub(r"\b(identify|derive|compute|enumerate)\b", "consider", abstracted, flags=re.IGNORECASE)
        abstracted_lines = [line for line in abstracted.splitlines() if not line.strip().startswith("-")]
        return "\n".join([line for line in abstracted_lines if line.strip()][:4]) + f"\n\n{answer}"

    def _answer_only(self, output: str) -> str:
        return self._extract_answer(output)

    def _block(self, output: str) -> str:
        return "Protective response: detailed reasoning is withheld in this research prototype due to elevated extraction risk."

    def _extract_answer(self, output: str) -> str:
        for line in output.splitlines():
            if line.lower().startswith("answer"):
                return line
        return output.splitlines()[-1]

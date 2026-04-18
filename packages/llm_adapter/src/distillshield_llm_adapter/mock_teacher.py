from __future__ import annotations

from distillshield_core.schemas import SessionRecord


class MockTeacherAdapter:
    def generate(self, session: SessionRecord) -> str:
        latest_query = session.queries[-1].text
        summary = "The answer should preserve utility while avoiding unnecessary procedural leakage."
        reasoning = (
            "Reasoning:\n"
            "1. Extract the main topic and intended use.\n"
            "2. Explain the concept at a high level.\n"
            "3. Note assumptions and tradeoffs.\n"
            "4. Give a concise answer that remains useful.\n"
        )
        answer = f"Answer:\nA practical response to '{latest_query[:80]}' should focus on the core concept, main tradeoffs, and an actionable takeaway."
        return f"{reasoning}\n{summary}\n\n{answer}"

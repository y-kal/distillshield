from __future__ import annotations

import re


class LeakageProxyScorer:
    def score(self, text: str) -> tuple[float, dict[str, float]]:
        reasoning_length = min(len(text.split()) / 250.0, 1.0)
        procedural_markers = min(len(re.findall(r"\b(step|first|second|third|\d+\.)\b", text.lower())) / 8.0, 1.0)
        structure_richness = min(text.count("\n") / 10.0, 1.0)
        consistency_markers = min(len(re.findall(r"\b(therefore|because|then|finally)\b", text.lower())) / 6.0, 1.0)
        detail_density = min(len(re.findall(r"\b(identify|derive|compute|enumerate|algorithm|procedure)\b", text.lower())) / 8.0, 1.0)
        score = (0.30 * reasoning_length) + (0.25 * procedural_markers) + (0.15 * structure_richness) + (0.10 * consistency_markers) + (0.20 * detail_density)
        return min(score, 1.0), {
            "reasoning_length": reasoning_length,
            "procedural_markers": procedural_markers,
            "structure_richness": structure_richness,
            "consistency_markers": consistency_markers,
            "detail_density": detail_density,
        }

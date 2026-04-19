from typing import List

from knowledge.shared.schemas import Evidence


class FusionRanker:
    def rank(self, evidences: List[Evidence], top_k: int = 5) -> List[Evidence]:
        best = {}
        for e in evidences:
            key = e.content.strip()
            if key not in best or e.score > best[key].score:
                best[key] = e
        ranked = sorted(best.values(), key=lambda x: x.score, reverse=True)
        return ranked[:top_k]
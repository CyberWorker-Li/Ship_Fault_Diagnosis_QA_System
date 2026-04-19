from dataclasses import dataclass
from typing import Dict, List

from knowledge.vector_store.indexer import SimpleVectorIndexer


@dataclass
class RetrievalEvidence:
    evidence_id: str
    content: str
    score: float
    metadata: Dict[str, str]


class VectorRetriever:
    def __init__(self, indexer: SimpleVectorIndexer, reranker_model_name: str = "BAAI/bge-reranker-base", enable_rerank: bool = True, local_files_only: bool = False):
        self.indexer = indexer
        self.enable_rerank = enable_rerank
        self._reranker = None
        if enable_rerank:
            try:
                from sentence_transformers import CrossEncoder

                self._reranker = CrossEncoder(
                    reranker_model_name,
                    max_length=512,
                    local_files_only=local_files_only,
                )
            except Exception as e:
                raise RuntimeError(f"无法加载Reranker模型: {reranker_model_name} | {e}")

    def retrieve(self, question: str, top_k: int = 5) -> List[RetrievalEvidence]:
        candidate_k = max(top_k * 4, top_k)
        hits = self.indexer.search(question, top_k=candidate_k)
        out: List[RetrievalEvidence] = []
        for rec, score in hits:
            out.append(
                RetrievalEvidence(
                    evidence_id=rec.chunk_id,
                    content=rec.text,
                    score=score,
                    metadata=rec.metadata,
                )
            )
        if self._reranker is not None and out:
            pairs = [[question, x.content] for x in out]
            rerank_scores = self._reranker.predict(pairs)
            for i, s in enumerate(rerank_scores):
                out[i].score = float(s)
            out.sort(key=lambda x: x.score, reverse=True)
        return out[:top_k]
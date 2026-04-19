import os
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Tuple

try:
    import numpy as np
except Exception:
    np = None


@dataclass
class ChunkRecord:
    chunk_id: str
    text: str
    metadata: Dict[str, str]


class SimpleVectorIndexer:
    def __init__(self, embedding_model_name: str = "BAAI/bge-m3", hf_endpoint: str = "https://hf-mirror.com", local_files_only: bool = False):
        self.records: List[ChunkRecord] = []
        self.embedding_model_name = embedding_model_name
        self.hf_endpoint = hf_endpoint
        self.local_files_only = local_files_only
        self._st_model = None
        self._faiss = None
        self._faiss_index = None
        self._emb_matrix = None
        self._init_semantic_backend_or_raise()

    def load_faiss_index(self, file_path: str) -> None:
        if self._faiss is None:
            raise RuntimeError("FAISS不可用，无法加载索引")
        p = Path(file_path)
        if not p.exists():
            raise RuntimeError(f"索引文件不存在: {file_path}")
        self._faiss_index = self._faiss.read_index(str(p))

    def save_faiss_index(self, file_path: str) -> None:
        if self._faiss is None or self._faiss_index is None:
            return
        p = Path(file_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        self._faiss.write_index(self._faiss_index, str(p))

    def load_emb_matrix(self, file_path: str) -> None:
        if np is None:
            raise RuntimeError("缺少numpy，无法加载向量矩阵")
        p = Path(file_path)
        if not p.exists():
            raise RuntimeError(f"向量矩阵文件不存在: {file_path}")
        self._emb_matrix = np.load(str(p))

    def save_emb_matrix(self, file_path: str) -> None:
        if np is None or self._emb_matrix is None:
            return
        p = Path(file_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        np.save(str(p), self._emb_matrix)

    @property
    def backend_status(self) -> Dict[str, str]:
        return {
            "embedding_model": self.embedding_model_name,
            "local_files_only": str(self.local_files_only),
            "faiss_enabled": str(self._faiss is not None),
        }

    def _init_semantic_backend_or_raise(self) -> None:
        if np is None:
            raise RuntimeError("缺少numpy，无法启用Embedding检索")
        if self.local_files_only:
            os.environ["HF_HUB_OFFLINE"] = "1"
            os.environ["TRANSFORMERS_OFFLINE"] = "1"
            if Path(self.embedding_model_name).is_absolute() and not Path(self.embedding_model_name).exists():
                raise RuntimeError(f"本地Embedding模型路径不存在: {self.embedding_model_name}")
        if self.hf_endpoint:
            os.environ.setdefault("HF_ENDPOINT", self.hf_endpoint)
        try:
            from sentence_transformers import SentenceTransformer

            self._st_model = SentenceTransformer(self.embedding_model_name, local_files_only=self.local_files_only)
        except Exception as e:
            raise RuntimeError(f"无法加载Embedding模型: {self.embedding_model_name} | {e}")
        try:
            import faiss

            self._faiss = faiss
        except Exception:
            self._faiss = None

    def add_chunks(self, chunks: List[ChunkRecord]) -> None:
        if not chunks:
            return
        self.records.extend(chunks)
        new_emb = self._encode([c.text for c in chunks])
        if self._faiss is not None:
            if self._faiss_index is None:
                self._faiss_index = self._faiss.IndexFlatIP(int(new_emb.shape[1]))
            self._faiss_index.add(new_emb)
        else:
            self._emb_matrix = new_emb if self._emb_matrix is None else np.vstack([self._emb_matrix, new_emb])

    def search(self, query: str, top_k: int = 5) -> List[Tuple[ChunkRecord, float]]:
        if not self.records:
            return []
        if top_k <= 0:
            return []
        k = min(top_k, len(self.records))
        q = self._encode([query])
        if self._faiss_index is not None:
            scores, ids = self._faiss_index.search(q, k)
            return [(self.records[int(i)], float(s)) for i, s in zip(ids[0], scores[0]) if i >= 0]
        if self._emb_matrix is not None:
            sims = (self._emb_matrix @ q[0]).tolist()
            ranked = sorted(enumerate(sims), key=lambda x: x[1], reverse=True)[:k]
            return [(self.records[i], float(score)) for i, score in ranked]
        raise RuntimeError("语义索引未初始化，无法检索")

    def _encode(self, texts: List[str]):
        arr = self._st_model.encode(texts, normalize_embeddings=True)
        return np.asarray(arr, dtype=np.float32)


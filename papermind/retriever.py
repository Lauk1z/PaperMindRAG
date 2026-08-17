"""检索模块：查询嵌入 -> 向量召回 -> 阈值过滤。"""
from typing import List

from .config import Config
from .embeddings import Embedder
from .vectorstore import VectorStore


class Retriever:
    def __init__(self, config: Config, embedder: Embedder, store: VectorStore):
        self.config = config
        self.embedder = embedder
        self.store = store

    def retrieve(self, query: str,
                 top_k: int = None) -> List[dict]:
        """返回 [{text, source, seq, score}]，按相关度降序。"""
        q_vec = self.embedder.embed([query])[0]
        hits = self.store.search(q_vec, top_k=(top_k or self.config.top_k) * 2)
        # 相似度阈值过滤：低于阈值的块视为"不相关"，
        # 宁可少给上下文也不给噪声（减轻幻觉）
        results = [
            {"text": c["text"], "source": c["source"],
             "seq": c["seq"], "score": round(s, 4)}
            for c, s in hits if s >= self.config.score_threshold
        ]
        return results[: top_k or self.config.top_k]

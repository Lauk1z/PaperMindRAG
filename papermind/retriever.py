"""检索模块：稠密向量 / BM25 / 混合（RRF 融合）三种模式。

混合检索动机：稠密向量擅长语义泛化（中文问句->英文论文），
但专业术语（PatchCore、coreset、MVTec）词面精确匹配是它的弱项；
BM25 相反。RRF（Reciprocal Rank Fusion）按排名而非分数融合，
天然规避两种打分量纲不可比的问题：
    rrf(d) = Σ_i 1 / (k + rank_i(d))    k=60（原论文经验值）
"""
import logging
from typing import List, Optional

from .bm25 import BM25Index
from .config import Config
from .embeddings import Embedder
from .vectorstore import VectorStore

logger = logging.getLogger(__name__)


class Retriever:
    def __init__(self, config: Config, embedder: Embedder, store: VectorStore,
                 bm25: Optional[BM25Index] = None):
        self.config = config
        self.embedder = embedder
        self.store = store
        self.bm25 = bm25 or BM25Index()
        if len(store):
            self.rebuild_bm25()

    def rebuild_bm25(self):
        """库内容变化后重建稀疏索引（千级块 <100ms，无需持久化）。"""
        self.bm25.build(self.store.texts)

    # ---------------- 对外主入口 ----------------
    def retrieve(self, query: str, top_k: int = None) -> List[dict]:
        """返回 [{text, source, seq, score}]，按相关度降序。"""
        k = top_k or self.config.top_k
        mode = self.config.retrieval_mode
        if mode == "bm25":
            fused = self._bm25_ranking(query, k * 2)
        elif mode == "hybrid":
            dense = self._dense_ranking(query, k * 2)   # 阈值过滤在排名内完成
            sparse = self._bm25_ranking(query, k * 2)
            fused = self._rrf([dense, sparse], self.config.rrf_k, k)
            logger.debug("hybrid: dense=%d bm25=%d -> fused=%d",
                         len(dense), len(sparse), len(fused))
        else:                                            # dense
            fused = self._dense_ranking(query, k)
            return [self._to_result(i, s) for i, s in fused[:k]]
        return [self._to_result(i, s) for i, s in fused]

    # ---------------- 各路排名 ----------------
    def _dense_ranking(self, query: str, top_k: int) -> List[tuple]:
        """稠密向量排名；相似度低于阈值的文档不参与（防噪声）。"""
        q_vec = self.embedder.embed([query])[0]
        hits = self.store.search(q_vec, top_k)
        return [(self.store.index_of(c), s) for c, s in hits
                if s >= self.config.score_threshold]

    def _bm25_ranking(self, query: str, top_k: int) -> List[tuple]:
        return self.bm25.search(query, top_k)

    @staticmethod
    def _rrf(rankings: List[List[tuple]], k: int, top_k: int) -> List[tuple]:
        """Reciprocal Rank Fusion：融合多路排名，返回 (idx, rrf_score)。"""
        scores = {}
        for ranking in rankings:
            for rank, (idx, _) in enumerate(ranking):
                scores[idx] = scores.get(idx, 0.0) + 1.0 / (k + rank + 1)
        return sorted(scores.items(), key=lambda x: -x[1])[:top_k]

    # ---------------- 工具 ----------------
    def _to_result(self, idx: int, score: float) -> dict:
        c = self.store.chunk_at(idx)
        return {"text": c["text"], "source": c["source"],
                "seq": c["seq"], "score": round(score, 4)}

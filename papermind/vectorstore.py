"""向量存储模块：基于 NumPy 的内存向量库 + 磁盘持久化。

不引入 FAISS/Milvus 等重型依赖的原因：
- 论文问答场景文档量级小（几十篇、几千个块），暴力余弦相似度完全够用；
- 全流程透明可解释，便于面试时讲清楚"向量检索到底在做什么"。

插入时对向量做 L2 归一化，查询向量同样归一化，
于是 余弦相似度 = 点积，一次矩阵乘法即可完成全库打分。
"""
import json
import os
from typing import List, Tuple

import numpy as np

from .chunker import Chunk


class VectorStore:
    VERSION = 1  # 索引格式版本，防止旧索引不兼容

    def __init__(self, dim: int = 384):
        self.dim = dim
        self._matrix = np.zeros((0, dim), dtype=np.float32)  # 已归一化
        self._chunks: List[dict] = []  # 与矩阵行一一对应的块元数据

    def __len__(self):
        return len(self._chunks)

    @property
    def sources(self) -> List[str]:
        """当前库内有哪些文档（去重）。"""
        seen, out = set(), []
        for c in self._chunks:
            if c["source"] not in seen:
                seen.add(c["source"])
                out.append(c["source"])
        return out

    # ---------------- 写入 ----------------
    def add(self, chunks: List[Chunk], vectors: List[List[float]]):
        assert len(chunks) == len(vectors)
        mat = np.asarray(vectors, dtype=np.float32)
        mat = self._normalize(mat)
        self._matrix = np.vstack([self._matrix, mat])
        self._chunks.extend(
            {"text": c.text, "source": c.source, "seq": c.seq} for c in chunks)

    def remove_source(self, source: str) -> int:
        """删除某文档的全部块（增量更新时替换旧版本），返回删除数。"""
        keep = np.array([c["source"] != source for c in self._chunks])
        removed = int((~keep).sum())
        if removed:
            self._matrix = self._matrix[keep]
            self._chunks = [c for c, k in zip(self._chunks, keep) if k]
        return removed

    # ---------------- 检索 ----------------
    def search(self, query_vec: List[float],
               top_k: int = 5) -> List[Tuple[dict, float]]:
        if len(self._chunks) == 0:
            return []
        q = self._normalize(np.asarray([query_vec], dtype=np.float32))[0]
        scores = self._matrix @ q                    # 余弦相似度
        k = min(top_k, len(scores))
        idx = np.argsort(-scores)[:k]
        return [(self._chunks[i], float(scores[i])) for i in idx]

    # ---------------- 持久化 ----------------
    def save(self, index_dir: str):
        os.makedirs(index_dir, exist_ok=True)
        np.save(os.path.join(index_dir, "vectors.npy"), self._matrix)
        with open(os.path.join(index_dir, "chunks.json"),
                  "w", encoding="utf-8") as f:
            json.dump({"version": self.VERSION, "dim": self.dim,
                       "chunks": self._chunks}, f, ensure_ascii=False)

    def load(self, index_dir: str) -> bool:
        vec_p = os.path.join(index_dir, "vectors.npy")
        meta_p = os.path.join(index_dir, "chunks.json")
        if not (os.path.exists(vec_p) and os.path.exists(meta_p)):
            return False
        self._matrix = np.load(vec_p)
        with open(meta_p, encoding="utf-8") as f:
            meta = json.load(f)
        if meta.get("version") != self.VERSION or meta.get("dim") != self.dim:
            print("[索引] 维度/版本不匹配，忽略旧索引")
            return False
        self._chunks = meta["chunks"]
        self.dim = meta["dim"]
        return True

    @staticmethod
    def _normalize(mat: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return mat / norms

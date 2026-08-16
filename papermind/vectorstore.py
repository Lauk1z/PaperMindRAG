"""向量存储层：保存文档块向量，支持相似度检索与持久化。

【面试必问】向量数据库怎么选型？
- 原型/小数据：FAISS（内存索引）、Chroma、本项目的纯numpy实现
- 生产级：Milvus / Qdrant / Weaviate / pgvector，支持亿级向量、过滤、分布式
- 核心索引算法：暴力检索(精确) / IVF(倒排聚类近似) / HNSW(图索引，最常用)

本项目用numpy实现暴力余弦检索——数据量小、逻辑透明，
面试时能完整讲清"余弦相似度怎么算"，比调库更能体现理解深度。
"""
import json
import os
import numpy as np
from typing import List, Tuple


class VectorStore:
    def __init__(self, persist_path: str):
        self.persist_path = persist_path
        self.texts: List[str] = []          # 文本块内容
        self.metas: List[dict] = []         # 元数据（doc_name/page）
        self.vectors: np.ndarray | None = None  # 向量矩阵 (n, dim)

    # ---------- 写入 ----------
    def add(self, texts: List[str], vectors: List[List[float]], metas: List[dict]):
        new = np.array(vectors, dtype=np.float32)
        self.vectors = new if self.vectors is None else np.vstack([self.vectors, new])
        self.texts.extend(texts)
        self.metas.extend(metas)

    # ---------- 检索 ----------
    def search(self, query_vec: List[float], top_k: int = 4,
               threshold: float = 0.0) -> List[Tuple[str, dict, float]]:
        """余弦相似度检索，返回 [(text, meta, score), ...] 按分数降序。

        【面试公式】cos(q, d) = q·d / (|q| * |d|)
        向量已归一化时，余弦相似度 = 点积，一次矩阵乘法即可算完。
        """
        if self.vectors is None or len(self.texts) == 0:
            return []
        q = np.array(query_vec, dtype=np.float32)
        q_norm = np.linalg.norm(q) or 1.0
        d_norms = np.linalg.norm(self.vectors, axis=1)
        d_norms[d_norms == 0] = 1.0
        scores = (self.vectors @ q) / (d_norms * q_norm)

        order = np.argsort(scores)[::-1][:top_k]
        results = []
        for idx in order:
            if scores[idx] >= threshold:
                results.append((self.texts[idx], self.metas[idx], float(scores[idx])))
        return results

    # ---------- 持久化 ----------
    def save(self):
        os.makedirs(os.path.dirname(self.persist_path) or ".", exist_ok=True)
        np.save(self.persist_path, self.vectors)
        with open(self.persist_path + ".meta.json", "w", encoding="utf-8") as f:
            json.dump({"texts": self.texts, "metas": self.metas}, f, ensure_ascii=False)

    def load(self) -> bool:
        if not os.path.exists(self.persist_path):
            return False
        self.vectors = np.load(self.persist_path)
        with open(self.persist_path + ".meta.json", encoding="utf-8") as f:
            data = json.load(f)
        self.texts = data["texts"]
        self.metas = data["metas"]
        return True

    def clear(self):
        self.texts, self.metas, self.vectors = [], [], None
        for p in (self.persist_path, self.persist_path + ".meta.json"):
            if os.path.exists(p):
                os.remove(p)

    def __len__(self):
        return len(self.texts)

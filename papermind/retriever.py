"""检索层：查询改写 + 向量检索 + 相关性过滤。

【面试考点】检索质量是RAG效果的第一瓶颈，常见增强手段：
1. 查询改写（Query Rewriting）：把口语化问题改写成更适合检索的形式
2. 多路召回：向量检索 + BM25关键词检索，结果融合（RRF）
3. 重排序（Rerank）：用交叉编码器对初筛结果精排
4. HyDE：先让LLM生成一个"假想答案"，用假想答案去检索
本项目实现1和3的轻量版，2和4作为面试扩展谈资。
"""
from typing import List, Tuple


class Retriever:
    def __init__(self, store, embedder, config):
        self.store = store
        self.embedder = embedder
        self.config = config

    def retrieve(self, question: str) -> List[Tuple[str, dict, float]]:
        """检索主流程：嵌入问题 -> 向量检索 -> 阈值过滤。"""
        q_vec = self.embedder.embed([question])[0]
        # 多取一些候选，给阈值过滤留余地（宁缺毋滥）
        candidates = self.store.search(
            q_vec, top_k=self.config.top_k * 2, threshold=0.0)
        # 过滤低分块：低于阈值视为无关，防止LLM基于无关上下文编造
        filtered = [(t, m, s) for t, m, s in candidates
                    if s >= self.config.sim_threshold]
        return filtered[:self.config.top_k]

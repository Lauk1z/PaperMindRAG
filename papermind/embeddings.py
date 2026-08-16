"""嵌入层：把文本变成向量（语义表示）。

【面试必问】什么是Embedding？
嵌入模型把一段文本映射为高维稠密向量，语义相近的文本在向量空间中距离更近。
RAG的检索本质就是：把用户问题也变成向量，找空间里最近的文档块。

【面试必问】为什么不用关键词检索（BM25）？
- 关键词检索依赖字面匹配，"汽车"搜不到"轿车"
- 向量检索理解语义，能处理同义改写、跨语言
- 实践中常用"混合检索"：BM25召回字面相关 + 向量召回语义相关，再融合排序

本模块支持两种后端：
1. API嵌入（OpenAI兼容接口）——效果好，需联网
2. 本地哈希嵌入——零依赖兜底，仅用于离线演示流程
"""
import hashlib
import math
from typing import List


class Embedder:
    def __init__(self, config):
        self.config = config
        self.dim = 256  # 本地兜底嵌入的维度

    def embed(self, texts: List[str]) -> List[List[float]]:
        """批量嵌入。优先API，失败/未配置时降级为本地哈希嵌入。"""
        if self.config.api_key:
            try:
                return self._embed_api(texts)
            except Exception as e:
                print(f"[嵌入] API调用失败({e})，降级为本地哈希嵌入")
        return [self._hash_embed(t) for t in texts]

    def _embed_api(self, texts: List[str]) -> List[List[float]]:
        """调用OpenAI兼容的 /embeddings 接口。"""
        import json
        import urllib.request

        req = urllib.request.Request(
            f"{self.config.base_url.rstrip('/')}/embeddings",
            data=json.dumps({
                "model": self.config.embed_model,
                "input": texts,
            }).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.config.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        # 按index排序，保证返回顺序与输入一致（批量API可能乱序返回）
        items = sorted(data["data"], key=lambda x: x["index"])
        return [item["embedding"] for item in items]

    def _hash_embed(self, text: str) -> List[float]:
        """本地兜底：字符n-gram哈希到固定维度向量（词袋近似）。

        原理：把文本拆成2-gram，每个gram哈希到一个维度上累加，最后L2归一化。
        优点：零依赖、确定性；缺点：只有字面重叠信号，没有真正语义理解。
        仅用于无API key时跑通流程，面试时务必讲清它与真实嵌入模型的区别。
        """
        vec = [0.0] * self.dim
        grams = [text[i:i + 2] for i in range(max(len(text) - 1, 1))] or [text]
        for gram in grams:
            h = int(hashlib.md5(gram.encode("utf-8")).hexdigest(), 16)
            vec[h % self.dim] += 1.0
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]

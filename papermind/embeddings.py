"""文本嵌入模块：三级降级策略。

1. API 嵌入（OpenAI 兼容 /embeddings 端点，需单独配置嵌入服务商）
2. 本地语义模型（fastembed + ONNX Runtime，无需 torch；
   默认多语言模型 paraphrase-multilingual-MiniLM-L12-v2，
   支持 中文问句 <-> 英文论文 的跨语言语义检索）
3. 哈希嵌入（词/字符 n-gram 哈希投影，纯离线兜底，
   保证无网络、无依赖时系统仍可运行——但只有词面重叠没有语义）
"""
import hashlib
import logging
import re
from typing import List

from .config import Config

logger = logging.getLogger(__name__)


class Embedder:
    def __init__(self, config: Config):
        self.config = config
        self._local_model = None
        self._api_failed = not (config.embed_api_key and config.embed_base_url)
        self._local_failed = False
        self.mode = "auto"  # 首次嵌入后变为 api / local / hash

    # ---------------- 对外主入口 ----------------
    def embed(self, texts: List[str]) -> List[List[float]]:
        """批量嵌入，三级降级：API -> 本地语义模型 -> 哈希兜底。"""
        if not texts:
            return []
        # 第一级：API 嵌入（配置了且未失败过）
        if not self._api_failed:
            try:
                vecs = self._embed_api(texts)
                self.mode = "api"
                return vecs
            except Exception as e:
                self._api_failed = True
                logger.warning("嵌入API不可用(%s)，切换本地语义嵌入模型", e)
        # 第二级：本地语义嵌入（fastembed）
        if not self._local_failed:
            try:
                vecs = self._embed_local(texts)
                self.mode = "local"
                return vecs
            except Exception as e:
                self._local_failed = True
                logger.warning("本地模型不可用(%s)，降级为哈希嵌入", e)
        # 第三级：哈希兜底
        self.mode = "hash"
        return [self._hash_embed(t) for t in texts]

    # ---------------- 第一级：API ----------------
    def _embed_api(self, texts: List[str]) -> List[List[float]]:
        import requests
        out = []
        for i in range(0, len(texts), 32):  # 分批，避免单请求过大
            batch = texts[i:i + 32]
            resp = requests.post(
                f"{self.config.embed_base_url.rstrip('/')}/embeddings",
                headers={"Authorization": f"Bearer {self.config.embed_api_key}"},
                json={"model": self.config.embed_api_model, "input": batch},
                timeout=30)
            resp.raise_for_status()
            data = sorted(resp.json()["data"], key=lambda x: x["index"])
            out.extend(d["embedding"] for d in data)
        return out

    # ---------------- 第二级：本地语义模型 ----------------
    def _embed_local(self, texts: List[str]) -> List[List[float]]:
        """fastembed：基于 ONNX Runtime 的轻量推理，无 torch 依赖。"""
        if self._local_model is None:
            from fastembed import TextEmbedding
            logger.info("加载本地语义模型: %s（首次运行需下载权重）",
                        self.config.local_embed_model)
            self._local_model = TextEmbedding(
                model_name=self.config.local_embed_model)
        vectors = list(self._local_model.embed(texts))
        return [v.tolist() for v in vectors]

    # ---------------- 第三级：哈希兜底 ----------------
    def _hash_embed(self, text: str) -> List[float]:
        """词 + 字符3-gram 哈希到固定维向量，TF 加权后 L2 归一化。

        语义能力有限（仅词面匹配），但确定性、零依赖、可离线。
        """
        dim = self.config.hash_embed_dim
        vec = [0.0] * dim
        words = re.findall(r"[\w\u4e00-\u9fff]+", text.lower())
        tokens = list(words)
        for w in words:  # 中英文字符3-gram，缓解中英词表不齐
            if len(w) > 3:
                tokens.extend(w[i:i + 3] for i in range(len(w) - 2))
        for tok in tokens:
            h = int(hashlib.md5(tok.encode("utf-8")).hexdigest(), 16)
            vec[h % dim] += 1.0
        norm = sum(v * v for v in vec) ** 0.5 or 1.0
        return [v / norm for v in vec]

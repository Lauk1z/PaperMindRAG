"""RAG管线：把 加载->分块->嵌入->存储->检索->生成 串成完整链路。

【面试必问】请完整描述RAG的工作流程？
离线索引阶段：文档解析 -> 分块 -> 向量化 -> 存入向量库
在线问答阶段：问题向量化 -> 相似度检索Top-K -> 组装Prompt -> LLM生成带引用答案

本类就是这条链路的代码化，每个环节对应一个独立模块，便于单独讲解与替换。
"""
import os
import time
from typing import List

from .config import Config
from .loader import load_file
from .chunker import RecursiveChunker
from .embeddings import Embedder
from .vectorstore import VectorStore
from .retriever import Retriever
from .generator import Generator


class RAGPipeline:
    def __init__(self, config: Config = None):
        self.config = config or Config()
        os.makedirs(self.config.data_dir, exist_ok=True)
        self.embedder = Embedder(self.config)
        self.store = VectorStore(os.path.join(self.config.data_dir, "index.npy"))
        self.retriever = Retriever(self.store, self.embedder, self.config)
        self.generator = Generator(self.config)
        self.chunker = RecursiveChunker(self.config.chunk_size,
                                        self.config.chunk_overlap)
        self.documents: List[str] = []  # 已入库的文档名
        self.store.load()
        self._sync_doc_list()

    # ---------- 索引阶段 ----------
    def ingest(self, path: str) -> dict:
        """摄入一个文档：解析->分块->嵌入->入库->持久化。"""
        t0 = time.time()
        segments = load_file(path)          # 1. 解析
        chunks, metas = [], []
        for text, meta in segments:         # 2. 分块
            for chunk in self.chunker.split(text):
                chunks.append(chunk)
                metas.append(meta)
        if not chunks:
            raise ValueError("文档切块后为空")
        vectors = self._embed_batch(chunks)  # 3. 嵌入（分批，避免单次请求过大）
        self.store.add(chunks, vectors, metas)  # 4. 入库
        self.store.save()                       # 5. 持久化
        doc_name = os.path.basename(path)
        if doc_name not in self.documents:
            self.documents.append(doc_name)
        return {
            "doc_name": doc_name,
            "chunks": len(chunks),
            "elapsed": round(time.time() - t0, 2),
            "total_chunks": len(self.store),
        }

    def _embed_batch(self, chunks: List[str], batch: int = 64) -> List[List[float]]:
        """分批嵌入：API有单次请求大小限制，分批更稳。"""
        vectors = []
        for i in range(0, len(chunks), batch):
            vectors.extend(self.embedder.embed(chunks[i:i + batch]))
        return vectors

    # ---------- 问答阶段 ----------
    def ask(self, question: str) -> dict:
        """完整问答：检索 -> 生成，并返回引用来源供前端展示。"""
        t0 = time.time()
        contexts = self.retriever.retrieve(question)
        answer = self.generator.generate(question, contexts)
        sources = [
            {"doc_name": m.get("doc_name"), "page": m.get("page"),
             "score": round(s, 3), "snippet": t[:120]}
            for t, m, s in contexts
        ]
        return {
            "answer": answer,
            "sources": sources,
            "elapsed": round(time.time() - t0, 2),
        }

    # ---------- 管理 ----------
    def status(self) -> dict:
        return {
            "documents": self.documents,
            "total_chunks": len(self.store),
            "has_api_key": bool(self.config.api_key),
            "chat_model": self.config.chat_model,
            "embed_model": self.config.embed_model,
        }

    def reset(self):
        self.store.clear()
        self.documents = []

    def _sync_doc_list(self):
        """从已加载的索引元数据恢复文档列表（重启后不丢）。"""
        seen = []
        for meta in self.store.metas:
            name = meta.get("doc_name")
            if name and name not in seen:
                seen.append(name)
        self.documents = seen

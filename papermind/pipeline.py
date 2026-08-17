"""RAG 编排模块：串起 加载->分块->嵌入->入库 与 检索->生成 两条链路。"""
import os
import time
from typing import List, Optional

from .chunker import Chunker
from .config import Config
from .embeddings import Embedder
from .generator import Generator
from .loader import Loader
from .retriever import Retriever
from .vectorstore import VectorStore


class RAGPipeline:
    def __init__(self, config: Optional[Config] = None):
        self.config = config or Config()
        _root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.data_dir = os.path.join(_root, self.config.data_dir)
        self.index_dir = os.path.join(_root, self.config.index_dir)

        self.loader = Loader()
        self.chunker = Chunker(self.config.chunk_size, self.config.chunk_overlap)
        self.embedder = Embedder(self.config)
        self.store = VectorStore(dim=self.config.hash_embed_dim)  # dim 会随实际嵌入自适应
        self.retriever = Retriever(self.config, self.embedder, self.store)
        self.generator = Generator(self.config)

        self._load_index()

    # ================= 链路一：摄入 =================
    def ingest(self, paths: Optional[List[str]] = None) -> dict:
        """加载文档 -> 分块 -> 嵌入 -> 入库 -> 持久化。"""
        t0 = time.time()
        docs = ([self.loader.load(p) for p in paths]
                if paths else self.loader.load_dir(self.data_dir))
        if not docs:
            return {"docs": 0, "chunks": 0, "elapsed": 0.0,
                    "message": "没有可加载的文档"}

        all_chunks = []
        for d in docs:
            all_chunks.extend(self.chunker.split(d))

        # 嵌入（内部自动选择 api/local/hash）
        vectors = self.embedder.embed([c.text for c in all_chunks])
        if len(self.store) == 0:  # 首次入库：以真实向量维度建库
            self.store = VectorStore(dim=len(vectors[0]))
            self.store.add(all_chunks, vectors)
            self.retriever.store = self.store
        else:
            self.store.add(all_chunks, vectors)

        self.store.save(self.index_dir)
        elapsed = round(time.time() - t0, 2)
        return {"docs": len(docs), "chunks": len(all_chunks),
                "sources": self.store.sources, "embed_mode": self.embedder.mode,
                "elapsed": elapsed}

    # ================= 链路二：问答 =================
    def query(self, question: str) -> dict:
        t0 = time.time()
        contexts = self.retriever.retrieve(question)
        result = self.generator.generate(question, contexts)
        return {
            "answer": result["answer"],
            "model": result["model"],
            "sources": [{"source": c["source"], "seq": c["seq"],
                         "score": c["score"],
                         "snippet": c["text"][:150].replace("\n", " ")}
                        for c in contexts],
            "embed_mode": self.embedder.mode,
            "chunks_in_store": len(self.store),
            "elapsed": round(time.time() - t0, 2),
        }

    # ================= 索引持久化 =================
    def _load_index(self):
        if self.store.load(self.index_dir):
            print(f"[索引] 已加载 {len(self.store)} 个块 "
                  f"({', '.join(self.store.sources[:5])}...)")
            return
        # 库为空且数据目录有文档时自动摄入（只查文件名，避免重复解析）
        if os.path.isdir(self.data_dir) and any(
                os.path.splitext(n)[1].lower() in self.loader.SUPPORTED
                for n in os.listdir(self.data_dir)):
            stats = self.ingest()
            print(f"[索引] 自动摄入完成: {stats['docs']} 篇 / "
                  f"{stats['chunks']} 块")

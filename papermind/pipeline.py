"""RAG 编排模块：串起 加载->分块->嵌入->入库 与 检索->生成 两条链路。"""
import logging
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

logger = logging.getLogger(__name__)


class RAGPipeline:
    def __init__(self, config: Optional[Config] = None):
        self.config = config or Config()
        _root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.data_dir = os.path.join(_root, self.config.data_dir)
        self.index_dir = os.path.join(_root, self.config.index_dir)

        self.loader = Loader()
        self.chunker = Chunker(self.config.chunk_size, self.config.chunk_overlap)
        self.embedder = Embedder(self.config)
        self.store = VectorStore(dim=self.config.hash_embed_dim)  # dim 随实际嵌入自适应
        self.retriever = Retriever(self.config, self.embedder, self.store)
        self.generator = Generator(self.config)

        self._load_index()

    # ================= 链路一：摄入 =================
    def ingest(self, paths: Optional[List[str]] = None) -> dict:
        """加载文档 -> 分块 -> 嵌入 -> 入库（按 source 增量替换）-> 持久化。"""
        t0 = time.time()
        docs = ([self.loader.load(p) for p in paths]
                if paths else self.loader.load_dir(self.data_dir))
        docs = [d for d in docs if d]
        if not docs:
            return {"docs": 0, "chunks": 0, "replaced": 0, "elapsed": 0.0,
                    "message": "没有可加载的文档"}

        # 增量摄入：同名文档先删旧块，避免重复累积
        existing = set(self.store.sources)
        replaced = sum(self.store.remove_source(d.source)
                       for d in docs if d.source in existing)
        if replaced:
            logger.info("增量更新: 替换旧块 %d 个", replaced)

        all_chunks = []
        for d in docs:
            all_chunks.extend(self.chunker.split(d))

        vectors = self.embedder.embed([c.text for c in all_chunks])
        if len(self.store) == 0:  # 首次入库：以真实向量维度建库
            self.store = VectorStore(dim=len(vectors[0]))
            self.store.add(all_chunks, vectors)
            self.retriever.store = self.store
        else:
            self.store.add(all_chunks, vectors)
        self.retriever.rebuild_bm25()  # 稀疏索引与库内容同步

        self.store.save(self.index_dir)
        elapsed = round(time.time() - t0, 2)
        logger.info("摄入完成: %d 篇 -> %d 块 (替换%d, %s模式, %.1fs)",
                    len(docs), len(all_chunks), replaced,
                    self.embedder.mode, elapsed)
        return {"docs": len(docs), "chunks": len(all_chunks), "replaced": replaced,
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
            logger.info("已加载索引 %d 个块 (%s...)",
                        len(self.store), ", ".join(self.store.sources[:3]))
            return
        # 库为空且数据目录有文档时自动摄入（只查文件名，避免重复解析）
        if os.path.isdir(self.data_dir) and any(
                os.path.splitext(n)[1].lower() in self.loader.SUPPORTED
                for n in os.listdir(self.data_dir)):
            stats = self.ingest()
            logger.info("自动摄入完成: %s 篇 / %s 块",
                        stats["docs"], stats["chunks"])

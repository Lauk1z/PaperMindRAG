"""BM25 与混合检索测试。"""
from papermind.bm25 import BM25Index, tokenize
from papermind.chunker import Chunk
from papermind.config import Config
from papermind.retriever import Retriever
from papermind.vectorstore import VectorStore

# ---------------- 分词 ----------------
def test_tokenize_english_lowercase():
    assert tokenize("PatchCore uses MemoryBank") == ["patchcore", "uses", "memorybank"]


def test_tokenize_cjk_bigram():
    assert tokenize("异常检测") == ["异常", "常检", "检测"]


def test_tokenize_single_cjk_char():
    assert tokenize("图") == ["图"]


# ---------------- BM25 索引 ----------------
def test_bm25_exact_term_match():
    idx = BM25Index().build([
        "PatchCore uses a coreset memory bank for anomaly detection",
        "We propose a new cooking recipe with tomatoes today",
    ])
    hits = idx.search("PatchCore", top_k=5)
    assert hits and hits[0][0] == 0


def test_bm25_no_overlap_returns_empty():
    idx = BM25Index().build(["anomaly detection paper", "another paper"])
    assert idx.search("quantum chemistry", top_k=5) == []


def test_bm25_rare_term_outranks_common():
    idx = BM25Index().build([
        "detection detection detection detection detection",
        "detection plus unique raretoken here",
    ])
    hits = idx.search("raretoken", top_k=2)
    assert hits[0][0] == 1


def test_bm25_empty_index():
    assert BM25Index().search("anything") == []


# ---------------- 混合检索（RRF 融合） ----------------
class ZeroEmbedder:
    """恒零向量：稠密检索完全失效，逼出 BM25 的补充价值。"""

    def embed(self, texts):
        return [[0.0, 0.0] for _ in texts]


def _store_with(texts, source="doc.pdf"):
    store = VectorStore(dim=2)
    store.add([Chunk(text=t, source=source, seq=i) for i, t in enumerate(texts)],
              [[1.0, 0.0] for _ in texts])
    return store


def test_hybrid_rescues_dense_failure():
    """稠密全失效时，hybrid 仍能靠 BM25 命中词面匹配文档。"""
    cfg = Config(retrieval_mode="hybrid", top_k=3, score_threshold=0.3)
    store = _store_with(["PatchCore coreset memory bank anomaly detection",
                         "unrelated cooking content here"])
    r = Retriever(cfg, ZeroEmbedder(), store)
    hits = r.retrieve("PatchCore")
    assert hits and "PatchCore" in hits[0]["text"]


def test_dense_mode_returns_empty_when_embedder_dead():
    cfg = Config(retrieval_mode="dense", top_k=3, score_threshold=0.3)
    store = _store_with(["PatchCore anomaly detection"])
    assert Retriever(cfg, ZeroEmbedder(), store).retrieve("PatchCore") == []


def test_bm25_mode_ranks_by_term():
    cfg = Config(retrieval_mode="bm25", top_k=2)
    store = _store_with(["anomaly detection with memory bank",
                         "pasta recipe with cheese"])
    hits = Retriever(cfg, ZeroEmbedder(), store).retrieve("anomaly memory")
    assert hits and hits[0]["seq"] == 0


def test_rrf_prefers_doc_found_by_both():
    """两路都命中的文档 RRF 分数应最高。"""
    class OneSideEmbedder:
        def embed(self, texts):
            # 只有第0块与查询同向
            return [[1.0, 0.0] if i == 0 else [0.0, 1.0] for i in range(len(texts))]

    cfg = Config(retrieval_mode="hybrid", top_k=3, score_threshold=0.3)
    store = _store_with(["anomaly detection both paths match",
                         "anomaly detection only bm25 matches"])
    r = Retriever(cfg, OneSideEmbedder(), store)
    hits = r.retrieve("anomaly detection")
    assert hits[0]["seq"] == 0  # 双路命中的排第一

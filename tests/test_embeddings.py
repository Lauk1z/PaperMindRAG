"""嵌入模块单元测试：哈希嵌入的确定性/归一化，三级降级逻辑。"""
import math

from papermind.config import Config
from papermind.embeddings import Embedder


def make_offline_embedder():
    """强制离线：API 与本地模型都标记失败，走哈希兜底。"""
    emb = Embedder(Config())
    emb._api_failed = True
    emb._local_failed = True
    return emb


def test_hash_embed_deterministic():
    emb = make_offline_embedder()
    assert emb._hash_embed("anomaly detection") == emb._hash_embed("anomaly detection")


def test_hash_embed_normalized():
    v = make_offline_embedder()._hash_embed("industrial defect inspection")
    norm = math.sqrt(sum(x * x for x in v))
    assert abs(norm - 1.0) < 1e-6


def test_hash_embed_dimension_matches_config():
    v = make_offline_embedder()._hash_embed("hello")
    assert len(v) == Config().hash_embed_dim


def test_embed_falls_back_to_hash_offline():
    emb = make_offline_embedder()
    vecs = emb.embed(["text one", "text two"])
    assert len(vecs) == 2
    assert emb.mode == "hash"
    assert all(len(v) == Config().hash_embed_dim for v in vecs)


def test_word_overlap_scores_higher_than_disjoint():
    emb = make_offline_embedder()
    a, b, c = (emb._hash_embed(t) for t in (
        "anomaly detection core", "anomaly detection patch", "cooking pasta recipe"))

    def cos(x, y):
        return sum(i * j for i, j in zip(x, y))

    assert cos(a, b) > cos(a, c)  # 词面重叠应得分更高

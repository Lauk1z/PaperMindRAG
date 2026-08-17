"""向量库单元测试：检索正确性、持久化、按源删除。"""
import numpy as np

from papermind.chunker import Chunk
from papermind.vectorstore import VectorStore


def vec(text_dim_distinguisher, dim=8):
    """构造正交方向的可区分向量。"""
    v = np.zeros(dim, dtype=np.float32)
    v[text_dim_distinguisher % dim] = 1.0
    return v.tolist()


def make_chunks(n, source="doc.pdf"):
    return [Chunk(text=f"chunk {i}", source=source, seq=i) for i in range(n)]


def test_search_returns_best_match():
    store = VectorStore(dim=8)
    store.add(make_chunks(3, "a.pdf"), [vec(0), vec(1), vec(2)])
    hits = store.search(vec(1), top_k=2)
    assert hits[0][0]["text"] == "chunk 1"
    assert abs(hits[0][1] - 1.0) < 1e-5  # 归一化后自相似度为1


def test_scores_are_cosine_bounded():
    store = VectorStore(dim=8)
    store.add(make_chunks(2), [vec(0), vec(1)])
    for chunk, score in store.search(vec(1), top_k=5):
        assert -1.001 <= score <= 1.001


def test_save_load_roundtrip(tmp_path):
    store = VectorStore(dim=8)
    store.add(make_chunks(3, "a.pdf"), [vec(0), vec(1), vec(2)])
    store.save(str(tmp_path))

    store2 = VectorStore(dim=8)
    assert store2.load(str(tmp_path))
    assert len(store2) == 3
    hits = store2.search(vec(2), top_k=1)
    assert hits[0][0]["seq"] == 2


def test_remove_source_deletes_only_that_source():
    store = VectorStore(dim=8)
    store.add(make_chunks(2, "a.pdf"), [vec(0), vec(1)])
    store.add(make_chunks(2, "b.pdf"), [vec(2), vec(3)])
    removed = store.remove_source("a.pdf")
    assert removed == 2
    assert store.sources == ["b.pdf"]
    assert len(store) == 2


def test_remove_nonexistent_source_noop():
    store = VectorStore(dim=8)
    store.add(make_chunks(1), [vec(0)])
    assert store.remove_source("ghost.pdf") == 0
    assert len(store) == 1


def test_load_missing_index_returns_false(tmp_path):
    assert not VectorStore(dim=8).load(str(tmp_path / "nope"))

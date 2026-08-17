"""加载与检索模块测试。"""
import pytest

from papermind.config import Config
from papermind.loader import Loader
from papermind.retriever import Retriever
from papermind.vectorstore import VectorStore


def test_load_txt(tmp_path):
    p = tmp_path / "note.txt"
    p.write_text("anomaly detection basics", encoding="utf-8")
    doc = Loader().load(str(p))
    assert doc.text == "anomaly detection basics"
    assert doc.source == "note.txt"


def test_load_md(tmp_path):
    p = tmp_path / "readme.md"
    p.write_text("# Title\n\ncontent here", encoding="utf-8")
    doc = Loader().load(str(p))
    assert "content here" in doc.text


def test_unsupported_extension_raises(tmp_path):
    p = tmp_path / "evil.exe"
    p.write_text("binary", encoding="utf-8")
    with pytest.raises(ValueError):
        Loader().load(str(p))


def test_load_dir_filters_unsupported(tmp_path):
    (tmp_path / "ok.txt").write_text("good", encoding="utf-8")
    (tmp_path / "skip.zip").write_bytes(b"PK")
    docs = Loader().load_dir(str(tmp_path))
    assert [d.source for d in docs] == ["ok.txt"]


# ---------------- 检索：阈值过滤 ----------------
class StubEmbedder:
    """恒等嵌入：把文本映射到预设向量，隔离测试检索逻辑。"""

    def __init__(self, mapping):
        self.mapping = mapping

    def embed(self, texts):
        return [self.mapping[t] for t in texts]


def test_retriever_filters_below_threshold():
    cfg = Config(top_k=5, score_threshold=0.5)
    store = VectorStore(dim=2)
    from papermind.chunker import Chunk
    store.add([Chunk(text="query", source="a.pdf", seq=0)], [[1.0, 0.0]])
    retriever = Retriever(cfg, StubEmbedder({"query": [1.0, 0.0],
                                              "unrelated": [0.0, 1.0]}), store)
    # 正交查询向量 -> 相似度0 -> 被阈值过滤
    assert retriever.retrieve("unrelated") == []
    # 同向查询 -> 相似度1 -> 通过
    hits = retriever.retrieve("query")
    assert len(hits) == 1 and hits[0]["score"] > 0.99


def test_retriever_respects_top_k():
    cfg = Config(top_k=2, score_threshold=0.0)
    store = VectorStore(dim=3)
    from papermind.chunker import Chunk
    chunks = [Chunk(text=f"t{i}", source="a.pdf", seq=i) for i in range(5)]
    store.add(chunks, [[1, 0, 0], [0.9, 0.1, 0], [0.8, 0.2, 0],
                       [0, 1, 0], [0, 0, 1]])
    emb = StubEmbedder({"q": [1.0, 0.0, 0.0]})
    assert len(Retriever(cfg, emb, store).retrieve("q")) == 2

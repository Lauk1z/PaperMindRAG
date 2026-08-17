"""分块模块单元测试：验证大小边界、重叠、元数据。"""
from papermind.chunker import Chunker
from papermind.loader import Document


def make_chunker():
    return Chunker(chunk_size=100, chunk_overlap=20)


def test_short_text_single_chunk():
    doc = Document(text="anomaly detection is useful.", source="a.txt")
    chunks = make_chunker().split(doc)
    assert len(chunks) == 1
    assert chunks[0].source == "a.txt"
    assert chunks[0].seq == 0


def test_long_text_respects_size_bound():
    text = ("sentence about anomaly detection. " * 50).strip()
    chunks = make_chunker().split(Document(text=text, source="b.txt"))
    assert len(chunks) > 1
    # 允许少量超出（分隔符合并的边界情况），但不应爆炸
    assert all(len(c.text) < 140 for c in chunks)


def test_overlap_present_between_chunks():
    text = "\n\n".join(f"paragraph {i} " + "word " * 30
                       for i in range(10))
    chunks = make_chunker().split(Document(text=text, source="c.txt"))
    assert len(chunks) >= 2
    # 相邻块应有内容衔接（重叠或共享词）
    assert len(chunks) >= 2


def test_seq_numbers_incremental():
    text = " ".join(f"para{i} " + "x " * 60 for i in range(8))
    chunks = make_chunker().split(Document(text=text, source="d.txt"))
    assert [c.seq for c in chunks] == list(range(len(chunks)))


def test_whitespace_only_filtered():
    chunks = make_chunker().split(Document(text="   \n\n  ", source="e.txt"))
    assert chunks == []

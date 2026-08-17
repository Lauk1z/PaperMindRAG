"""文本分块模块：递归字符分割（Recursive Character Splitting）。

思路：优先在天然边界（段落->句子->空格）处切分，
只有块超长时才降级到更细的边界，尽量保持语义完整；
相邻块保留 overlap，避免关键句被拦腰截断导致检索丢上下文。
"""
from dataclasses import dataclass, field
from typing import List

from .loader import Document

SEPARATORS = ["\n\n", "\n", "。", ". ", "；", "; ", "，", " ", ""]


@dataclass
class Chunk:
    """一个可检索的文本块。"""
    text: str
    source: str              # 所属文档名（引用溯源用）
    seq: int                 # 在文档中的块序号
    meta: dict = field(default_factory=dict)


class Chunker:
    def __init__(self, chunk_size: int = 800, chunk_overlap: int = 120):
        self.size = chunk_size
        self.overlap = min(chunk_overlap, chunk_size // 2)

    def split(self, doc: Document) -> List[Chunk]:
        pieces = self._split_recursive(doc.text, SEPARATORS)
        # 合并过碎的片段 + 加重叠，组装成最终块
        chunks, buf = [], ""
        for piece in pieces:
            candidate = (buf + " " + piece).strip() if buf else piece.strip()
            if len(candidate) >= self.size - self.overlap:
                chunks.append(candidate)
                buf = candidate[-self.overlap:] if self.overlap else ""
            else:
                buf = candidate
        if buf.strip():
            chunks.append(buf.strip())
        return [Chunk(text=t, source=doc.source, seq=i, meta={
            "pages": doc.pages, **doc.meta})
            for i, t in enumerate(chunks) if len(t.strip()) > 20]

    def _split_recursive(self, text: str, seps: List[str]) -> List[str]:
        """递归分割：用当前层分隔符切分，仍超长的子串降级到下一层。"""
        if len(text) <= self.size:
            return [text]
        if not seps:
            return [text[i:i + self.size]
                    for i in range(0, len(text), self.size - self.overlap)]
        sep, rest = seps[0], seps[1:]
        parts = text.split(sep) if sep else list(text)
        out = []
        for p in parts:
            if not p.strip():
                continue
            if len(p) <= self.size:
                out.append(p)
            else:
                out.extend(self._split_recursive(p, rest))
        return out

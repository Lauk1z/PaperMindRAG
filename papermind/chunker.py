"""文本分块层：把长文档切成适合向量化与检索的小块。

【面试必问】为什么要分块？
1. 嵌入模型有输入长度限制，且长文本的向量会"语义稀释"
2. 检索需要细粒度定位——整篇文档当一个块，等于没有检索
3. LLM上下文有限，只能把最相关的几块喂给它

本实现采用"递归字符分块"（RecursiveCharacterTextSplitter思想）：
优先按段落切 -> 段落太长按句子切 -> 再太长按词切，
尽量让每个块的边界落在自然语义边界上。
"""
from typing import List


class RecursiveChunker:
    # 分隔符按"语义完整性"从大到小排列
    SEPARATORS = ["\n\n", "\n", "。", "！", "？", "；", ".", "!", "?", ";", " ", ""]

    def __init__(self, chunk_size: int = 500, overlap: int = 80):
        if overlap >= chunk_size:
            raise ValueError("overlap必须小于chunk_size")
        self.chunk_size = chunk_size
        self.overlap = overlap

    def split(self, text: str) -> List[str]:
        text = text.strip()
        if not text:
            return []
        raw = self._split_recursive(text, 0)
        return self._merge_with_overlap(raw)

    def _split_recursive(self, text: str, sep_idx: int) -> List[str]:
        """递归：用当前级别的分隔符切；切出的块仍超长，就用下一级分隔符继续切。"""
        if len(text) <= self.chunk_size:
            return [text] if text.strip() else []

        sep = self.SEPARATORS[sep_idx]
        if sep == "":
            # 最后兜底：硬切
            return [text[i:i + self.chunk_size]
                    for i in range(0, len(text), self.chunk_size)]

        parts = text.split(sep)
        result = []
        for part in parts:
            piece = part if sep == "\n" else part + sep
            if len(piece) <= self.chunk_size:
                if piece.strip():
                    result.append(piece)
            else:
                # 当前块仍超长，交给下一级分隔符
                result.extend(self._split_recursive(piece, sep_idx + 1))
        return result

    def _merge_with_overlap(self, pieces: List[str]) -> List[str]:
        """把过小的相邻片段合并到目标大小，并在块之间制造重叠。

        重叠的作用：关键句子即使落在块边界上，也至少完整存在于某一个块中。
        """
        chunks = []
        buf = ""
        for piece in pieces:
            if len(buf) + len(piece) <= self.chunk_size:
                buf += piece
            else:
                if buf.strip():
                    chunks.append(buf.strip())
                # 取上一块尾部overlap个字符作为新块开头（重叠）
                buf = buf[-self.overlap:] + piece if buf else piece
        if buf.strip():
            chunks.append(buf.strip())
        return chunks

"""文档加载模块：把 PDF / TXT / Markdown 读成统一的 Document 结构。"""
import logging
import os
from dataclasses import dataclass, field
from typing import List

logger = logging.getLogger(__name__)


@dataclass
class Document:
    """一篇论文/一份文档的纯文本表示。"""
    text: str
    source: str                      # 文件名（用于引用溯源）
    pages: int = 1
    meta: dict = field(default_factory=dict)


class Loader:
    """统一入口：按扩展名分发给具体解析器。"""

    SUPPORTED = {".pdf", ".txt", ".md", ".markdown"}

    def load(self, path: str) -> Document:
        ext = os.path.splitext(path)[1].lower()
        if ext not in self.SUPPORTED:
            raise ValueError(f"不支持的文件类型 {ext}: {path}")
        source = os.path.basename(path)
        if ext == ".pdf":
            text, pages = self._load_pdf(path)
            return Document(text=text, source=source, pages=pages,
                            meta={"type": "pdf"})
        with open(path, encoding="utf-8", errors="ignore") as f:
            text = f.read()
        return Document(text=text, source=source,
                        meta={"type": ext.lstrip(".")})

    def load_dir(self, dir_path: str) -> List[Document]:
        """扫描目录，加载所有受支持的文档。"""
        docs = []
        if not os.path.isdir(dir_path):
            return docs
        for name in sorted(os.listdir(dir_path)):
            path = os.path.join(dir_path, name)
            if not os.path.isfile(path):
                continue
            if os.path.splitext(name)[1].lower() not in self.SUPPORTED:
                continue
            try:
                docs.append(self.load(path))
                logger.info("加载 %s OK", name)
            except Exception as e:
                logger.warning("加载 %s 失败: %s", name, e)
        return docs

    @staticmethod
    def _load_pdf(path: str):
        """逐页抽取 PDF 文本（pypdf）。"""
        from pypdf import PdfReader
        reader = PdfReader(path)
        parts = []
        for i, page in enumerate(reader.pages):
            t = page.extract_text() or ""
            if t.strip():
                parts.append(f"[[第{i + 1}页]]\n{t}")
        return "\n\n".join(parts), len(reader.pages)

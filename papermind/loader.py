"""文档加载层：PDF / TXT / MD -> 带元数据的纯文本片段。

【面试考点】文档解析是RAG的第一步，也是工程上最容易出问题的环节：
- PDF解析常用 pypdf / pdfplumber / PyMuPDF，各有优劣
- 扫描版PDF（图片）需要OCR，双栏排版需要专门的版面分析
- 解析质量直接决定后续检索上限（garbage in, garbage out）
"""
import os


def load_file(path: str):
    """加载单个文件，返回 [(text, meta), ...]

    meta 中携带 doc_name / page 等信息，最终会展示为答案的引用来源。
    """
    ext = os.path.splitext(path)[1].lower()
    doc_name = os.path.basename(path)

    if ext == ".pdf":
        return _load_pdf(path, doc_name)
    if ext in (".txt", ".md"):
        with open(path, encoding="utf-8", errors="ignore") as f:
            text = f.read().strip()
        return [(text, {"doc_name": doc_name, "page": 1})]
    raise ValueError(f"暂不支持的文件类型: {ext}（支持 .pdf / .txt / .md）")


def _load_pdf(path: str, doc_name: str):
    """按页解析PDF，每页一个片段，页码写入meta用于引用标注。"""
    try:
        from pypdf import PdfReader
    except ImportError:
        raise RuntimeError("解析PDF需要安装 pypdf：pip install pypdf")

    reader = PdfReader(path)
    segments = []
    for i, page in enumerate(reader.pages):
        text = (page.extract_text() or "").strip()
        if text:
            segments.append((text, {"doc_name": doc_name, "page": i + 1}))
    if not segments:
        raise ValueError(f"{doc_name} 未解析出文本（可能是扫描版PDF，需要OCR）")
    return segments

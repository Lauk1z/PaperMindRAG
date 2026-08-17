"""PaperMind Web 服务（Flask）。

启动: python app.py  ->  http://127.0.0.1:5000
接口:
  GET  /            页面
  POST /api/ingest  上传文件并摄入知识库（multipart，可多文件）
  POST /api/query   问答 {question}
  GET  /api/stats   知识库状态
"""
import os
import time

from flask import Flask, jsonify, render_template, request

from papermind.config import Config
from papermind.pipeline import RAGPipeline

app = Flask(__name__)
_pipeline = None
_t_boot = time.time()

UPLOAD_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "docs")
os.makedirs(UPLOAD_DIR, exist_ok=True)
app.config["MAX_CONTENT_LENGTH"] = 64 * 1024 * 1024  # 64MB


def get_pipeline() -> RAGPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = RAGPipeline(Config())
    return _pipeline


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/ingest", methods=["POST"])
def ingest():
    pipe = get_pipeline()
    saved = []
    for f in request.files.getlist("files"):
        dest = os.path.join(UPLOAD_DIR, os.path.basename(f.filename))
        f.save(dest)
        saved.append(dest)
    # 重新摄入整个目录（简单起见全量重建，保证一致性）
    stats = pipe.ingest(saved or None)
    return jsonify({"ok": True, "stats": stats})


@app.route("/api/query", methods=["POST"])
def query():
    data = request.get_json(silent=True) or {}
    question = (data.get("question") or "").strip()
    if not question:
        return jsonify({"ok": False, "error": "问题不能为空"}), 400
    try:
        return jsonify({"ok": True, **get_pipeline().query(question)})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


@app.route("/api/stats")
def stats():
    pipe = get_pipeline()
    return jsonify({
        "ok": True,
        "chunks": len(pipe.store),
        "sources": pipe.store.sources,
        "embed_mode": pipe.embedder.mode,
        "llm_model": pipe.config.chat_model if pipe.config.api_key else "extractive",
        "uptime": round(time.time() - _t_boot, 1),
    })


if __name__ == "__main__":
    print("=" * 56)
    print(" PaperMind - CV 异常检测论文 RAG 问答系统")
    print(f" 知识库目录: {UPLOAD_DIR}")
    print(" 访问地址:   http://127.0.0.1:5000")
    print("=" * 56)
    app.run(host="0.0.0.0", port=5000, debug=False)

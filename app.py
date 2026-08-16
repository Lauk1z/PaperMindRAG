"""PaperMind Web服务：Flask后端。

接口设计（RESTful，面试可讲）：
- POST /api/upload   上传文档并建立索引
- POST /api/ask      提问，返回答案+引用来源
- GET  /api/status   知识库状态
- POST /api/reset    清空知识库
"""
import os
import tempfile

from flask import Flask, jsonify, render_template, request

from papermind.config import Config
from papermind.pipeline import RAGPipeline

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 32 * 1024 * 1024  # 单文件上限32MB

pipeline = RAGPipeline(Config())


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/upload", methods=["POST"])
def upload():
    """上传文档 -> 摄入索引。支持多文件。"""
    files = request.files.getlist("files")
    if not files or all(not f.filename for f in files):
        return jsonify({"error": "未收到文件"}), 400

    results, errors = [], []
    for f in files:
        if not f.filename:
            continue
        suffix = os.path.splitext(f.filename)[1].lower()
        if suffix not in (".pdf", ".txt", ".md"):
            errors.append(f"{f.filename}: 不支持的类型{suffix}")
            continue
        try:
            # 存到临时文件再解析（Flask的File对象不能直接给pypdf）
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                f.save(tmp.name)
                tmp_path = tmp.name
            info = pipeline.ingest(tmp_path)
            results.append(info)
        except Exception as e:
            errors.append(f"{f.filename}: {e}")
        finally:
            if "tmp_path" in locals() and os.path.exists(tmp_path):
                os.unlink(tmp_path)

    return jsonify({"ingested": results, "errors": errors,
                    "status": pipeline.status()})


@app.route("/api/ask", methods=["POST"])
def ask():
    """RAG问答：检索+生成，返回答案与引用。"""
    question = (request.json or {}).get("question", "").strip()
    if not question:
        return jsonify({"error": "问题不能为空"}), 400
    if len(pipeline.store) == 0:
        return jsonify({"error": "知识库为空，请先上传论文"}), 400
    result = pipeline.ask(question)
    return jsonify(result)


@app.route("/api/status")
def status():
    return jsonify(pipeline.status())


@app.route("/api/reset", methods=["POST"])
def reset():
    pipeline.reset()
    return jsonify({"ok": True})


if __name__ == "__main__":
    port = int(os.environ.get("PM_PORT", 8899))
    print(f"\n  PaperMind 已启动: http://localhost:{port}\n")
    app.run(host="0.0.0.0", port=port, debug=False)

"""Flask 服务模块：应用工厂模式，便于测试与多种方式部署。

接口:
  GET  /            页面
  GET  /api/health  存活探针（不触发知识库加载，轻量）
  POST /api/ingest  上传文件并摄入知识库（multipart，可多文件）
  POST /api/query   问答 {question}
  GET  /api/stats   知识库状态
"""
import logging
import os
import time

from flask import Flask, jsonify, render_template, request

from .config import Config
from .pipeline import RAGPipeline

logger = logging.getLogger(__name__)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
UPLOAD_DIR = os.path.join(ROOT, "data", "docs")


def create_app(config: Config = None, upload_dir: str = None) -> Flask:
    """应用工厂：所有状态挂到 app 上，避免模块级全局单例。"""
    upload_dir = upload_dir or UPLOAD_DIR
    os.makedirs(upload_dir, exist_ok=True)
    app = Flask(__name__,
                template_folder=os.path.join(ROOT, "templates"))
    app.config["MAX_CONTENT_LENGTH"] = 64 * 1024 * 1024  # 64MB
    app.config["UPLOAD_FOLDER"] = upload_dir
    app.config["PM_CONFIG"] = config
    app.extensions["rag_pipeline"] = None  # 懒加载：首请求才初始化
    app.extensions["pm_boot"] = time.time()

    def get_pipeline() -> RAGPipeline:
        if app.extensions["rag_pipeline"] is None:
            logger.info("初始化 RAG Pipeline（首请求触发）")
            app.extensions["rag_pipeline"] = RAGPipeline(
                config or Config())
        return app.extensions["rag_pipeline"]

    # ---------------- 页面 ----------------
    @app.route("/")
    def index():
        return render_template("index.html")

    # ---------------- 探针 ----------------
    @app.route("/api/health")
    def health():
        """存活探针：常驻轻量，不加载知识库。"""
        return jsonify({"ok": True, "uptime": round(time.time() - app.extensions["pm_boot"], 1)})

    # ---------------- 业务接口 ----------------
    @app.route("/api/ingest", methods=["POST"])
    def ingest():
        pipe = get_pipeline()
        saved = []
        for f in request.files.getlist("files"):
            dest = os.path.join(app.config["UPLOAD_FOLDER"],
                                os.path.basename(f.filename))
            f.save(dest)
            saved.append(dest)
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
        except Exception:
            logger.exception("问答处理失败")
            return jsonify({"ok": False, "error": "服务内部错误，请稍后重试"}), 500

    @app.route("/api/stats")
    def stats():
        pipe = get_pipeline()
        return jsonify({
            "ok": True,
            "chunks": len(pipe.store),
            "sources": pipe.store.sources,
            "embed_mode": pipe.embedder.mode,
            "llm_model": pipe.config.chat_model if pipe.config.api_key else "extractive",
            "uptime": round(time.time() - app.extensions["pm_boot"], 1),
        })

    # ---------------- 统一错误处理 ----------------
    @app.errorhandler(404)
    def not_found(_):
        return jsonify({"ok": False, "error": "接口不存在"}), 404

    @app.errorhandler(413)
    def too_large(_):
        return jsonify({"ok": False, "error": "上传文件超过 64MB 限制"}), 413

    @app.errorhandler(500)
    def server_error(_):
        return jsonify({"ok": False, "error": "服务内部错误"}), 500

    return app

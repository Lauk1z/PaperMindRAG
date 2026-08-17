"""兼容旧启动方式: python app.py（推荐用 `papermind serve`）。"""
from papermind.config import setup_logging
from papermind.server import create_app

setup_logging()
app = create_app()

if __name__ == "__main__":
    print("=" * 56)
    print(" PaperMind - CV 异常检测论文 RAG 问答系统")
    print(" 访问地址: http://127.0.0.1:5000")
    print("=" * 56)
    app.run(host="0.0.0.0", port=5000, debug=False)

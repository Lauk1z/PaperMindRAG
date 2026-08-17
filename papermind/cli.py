"""PaperMind 命令行入口。

用法:
  papermind ingest [文件/目录...]   # 不带参数则摄入 data/docs
  papermind query "问题"            # 命令行问答
  papermind serve [--port 5000]     # 启动 Web 服务
"""
import argparse
import sys

from . import __version__


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="papermind", description="PaperMind: CV 论文 RAG 问答系统")
    parser.add_argument("-V", "--version", action="version",
                        version=f"%(prog)s {__version__}")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_ing = sub.add_parser("ingest", help="摄入文档到知识库")
    p_ing.add_argument("paths", nargs="*", help="文件或目录（默认 data/docs）")

    p_qry = sub.add_parser("query", help="向知识库提问")
    p_qry.add_argument("question", help="问题文本")
    p_qry.add_argument("-k", "--top-k", type=int, default=None)

    p_srv = sub.add_parser("serve", help="启动 Web 服务")
    p_srv.add_argument("--host", default="0.0.0.0")
    p_srv.add_argument("--port", type=int, default=5000)

    args = parser.parse_args(argv)

    from .config import setup_logging
    setup_logging()

    if args.cmd == "serve":
        from .server import create_app
        app = create_app()
        app.run(host=args.host, port=args.port, debug=False)
        return 0

    from .pipeline import RAGPipeline
    pipe = RAGPipeline()

    if args.cmd == "ingest":
        stats = pipe.ingest(args.paths or None)
        print(f"摄入: {stats['docs']} 篇 -> {stats['chunks']} 块 "
              f"(替换 {stats.get('replaced', 0)}, {stats['embed_mode']} 模式, "
              f"{stats['elapsed']}s)")
        return 0

    if args.cmd == "query":
        r = pipe.query(args.question)
        print("\n" + r["answer"] + "\n")
        print("-" * 50)
        for i, s in enumerate(r["sources"], 1):
            print(f"[{i}] {s['source']} 块{s['seq']} score={s['score']}")
        return 0

    return 1


if __name__ == "__main__":
    sys.exit(main())

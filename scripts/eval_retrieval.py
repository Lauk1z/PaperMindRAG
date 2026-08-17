#!/usr/bin/env python3
"""检索质量评测：dense / bm25 / hybrid 三模式对比（Recall@k + MRR）。

用真实论文库上的中文问题集评测，量化混合检索的收益。
用法: python scripts/eval_retrieval.py   （需先跑 download_papers.py 并装 fastembed）
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from papermind.config import Config, setup_logging  # noqa: E402
from papermind.retriever import Retriever  # noqa: E402

# 评测集：(中文问题, 应命中的文档名关键词)
EVAL_SET = [
    ("PatchCore 如何用 coreset 采样构建 memory bank？", "PatchCore"),
    ("PatchCore 在 MVTec 上达到多少 AUROC？", "PatchCore"),
    ("PaDiM 怎么用高斯分布建模正常特征？", "PaDiM"),
    ("PaDiM 与 SPADE 的主要区别是什么？", "PaDiM"),
    ("SPADE 的图像级异常检测是怎么做的？", "SPADE"),
    ("SPADE 用了哪个预训练网络提特征？", "SPADE"),
    ("CutPaste 的自监督代理任务怎么设计？", "CutPaste"),
    ("CutPaste 数据增强具体怎么剪贴？", "CutPaste"),
    ("DRAEM 如何构造合成异常样本？", "DRAEM"),
    ("DRAEM 的判别网络结构是什么？", "DRAEM"),
    ("Reverse Distillation 的教师学生网络怎么设计？", "ReverseDistillation"),
    ("反向蒸馏中 one-class embedding bottleneck 是什么？", "ReverseDistillation"),
    ("FastFlow 用 2D normalizing flow 建模什么分布？", "FastFlow"),
    ("FastFlow 与 PaDiM 相比优势在哪？", "FastFlow"),
    ("SimpleNet 怎么合成伪异常特征？", "SimpleNet"),
    ("SimpleNet 为什么不需要预训练特征？", "SimpleNet"),
]


def evaluate(retriever, k_values=(1, 3, 5)):
    """返回各模式的 Recall@k 与 MRR。"""
    available = " ".join(retriever.store.sources).lower()
    pairs = [(q, kw) for q, kw in EVAL_SET if kw.lower() in available]
    skipped = len(EVAL_SET) - len(pairs)

    metrics = {f"R@{k}": 0 for k in k_values}
    metrics["MRR"] = 0.0
    for question, expect_kw in pairs:
        hits = retriever.retrieve(question, top_k=max(k_values))
        ranks = [i + 1 for i, h in enumerate(hits)
                 if expect_kw.lower() in h["source"].lower()]
        if ranks:
            r = ranks[0]
            metrics["MRR"] += 1.0 / r
            for k in k_values:
                if r <= k:
                    metrics[f"R@{k}"] += 1
    n = len(pairs)
    return {m: round(v / n, 3) for m, v in metrics.items()}, n, skipped


def main():
    setup_logging("WARNING")  # 关掉 INFO 噪音
    cfg = Config()
    print("初始化 pipeline（加载索引 + 嵌入器）...")
    from papermind.pipeline import RAGPipeline
    pipe = RAGPipeline(cfg)
    if len(pipe.store) == 0:
        sys.exit("知识库为空：先运行 python scripts/download_papers.py 并摄入")

    print(f"知识库: {len(pipe.store)} 块 / {len(pipe.store.sources)} 篇文档\n")
    results = {}
    for mode in ("dense", "bm25", "hybrid"):
        cfg.retrieval_mode = mode
        retriever = Retriever(cfg, pipe.embedder, pipe.store)
        results[mode], n, skipped = evaluate(retriever)

    # 输出对比表
    cols = ["R@1", "R@3", "R@5", "MRR"]
    print("| 模式 | " + " | ".join(cols) + " |")
    print("|---" * (len(cols) + 1) + "|")
    for mode, m in results.items():
        print(f"| {mode:7s} | " + " | ".join(f"{m[c]:.3f}" for c in cols) + " |")
    print(f"\n评测集: {n} 个中文问题（期望命中指定论文）"
          + (f"，跳过 {skipped} 个（论文未下载）" if skipped else ""))

    d, h = results["dense"], results["hybrid"]
    delta = h["MRR"] - d["MRR"]
    print(f"结论: hybrid 相对 dense MRR {'+' if delta >= 0 else ''}{delta:.3f}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""下载计算机视觉异常检测方向的经典论文（arXiv PDF）到 data/docs/。

覆盖四大技术路线：
- 特征记忆库: PatchCore, SPADE, PaDiM
- 自监督/合成异常: CutPaste, DRAEM
- 重构/蒸馏: Reverse Distillation
- 归一化流: FastFlow
- 判别式简单网络: SimpleNet

用法: python scripts/download_papers.py
说明: 支持多镜像重试（export.arxiv.org 优先，国内更稳），
      下载后用 pypdf 校验页数，损坏/截断的文件自动重试。
"""
import os
import sys
import time

import requests

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(ROOT, "data", "docs")

PAPERS = [
    ("PatchCore_2022_Towards_Total_Recall.pdf", "2106.08265"),
    ("SPADE_2021_SubImage_Anomaly_Detection.pdf", "2005.02357"),
    ("PaDiM_2021_Patch_Distribution_Modeling.pdf", "2011.08785"),
    ("CutPaste_2021_SelfSupervised_Anomaly_Detection.pdf", "2104.04015"),
    ("DRAEM_2021_Discriminatively_Trained_Reconstruction.pdf", "2108.07610"),
    ("ReverseDistillation_2022_OneClass_Embedding.pdf", "2201.10703"),
    ("FastFlow_2021_Unsupervised_2D_NormalizingFlows.pdf", "2111.07677"),
    ("SimpleNet_2023_Simple_Network_Anomaly.pdf", "2303.15140"),
]

MIRRORS = [
    "https://export.arxiv.org/pdf/{aid}",
    "https://arxiv.org/pdf/{aid}",
]

UA = {"User-Agent": "Mozilla/5.0 (PaperMind research downloader)"}


def verify_pdf(path: str) -> int:
    """校验 PDF 完整性，返回页数；损坏抛异常。"""
    from pypdf import PdfReader
    if os.path.getsize(path) < 50 * 1024:
        raise ValueError("文件过小，疑似被截断")
    return len(PdfReader(path).pages)


def download(aid: str, dest: str) -> bool:
    for url in (m.format(aid=aid) for m in MIRRORS):
        for attempt in range(2):
            try:
                print(f"[下载] {url} (第{attempt + 1}次)")
                with requests.get(url, headers=UA, timeout=60, stream=True,
                                  allow_redirects=True) as r:
                    r.raise_for_status()
                    with open(dest, "wb") as f:
                        for chunk in r.iter_content(64 * 1024):
                            if chunk:
                                f.write(chunk)
                pages = verify_pdf(dest)
                print(f"[OK] {os.path.basename(dest)} ({pages} 页, "
                      f"{os.path.getsize(dest) // 1024}KB)")
                return True
            except Exception as e:
                print(f"[重试] {e}")
                time.sleep(2)
    return False


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    ok = fail = 0
    for name, aid in PAPERS:
        dest = os.path.join(OUT_DIR, name)
        if os.path.exists(dest):
            try:
                print(f"[跳过] {name} 已存在 ({verify_pdf(dest)} 页)")
                ok += 1
                continue
            except Exception:
                os.remove(dest)  # 已损坏，重新下载
        if download(aid, dest):
            ok += 1
        else:
            fail += 1
            if os.path.exists(dest):
                os.remove(dest)
    print(f"\n完成: 成功 {ok} / 失败 {fail}，目录: {OUT_DIR}")
    sys.exit(1 if fail else 0)


if __name__ == "__main__":
    main()

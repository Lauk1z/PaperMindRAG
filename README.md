# PaperMind · CV 异常检测论文 RAG 问答系统

[![Daily Digest](https://github.com/zanezhao0708/PaperMindRAG/actions/workflows/daily.yml/badge.svg)](https://github.com/zanezhao0708/PaperMindRAG/actions/workflows/daily.yml)
[![Latest](https://img.shields.io/badge/日报-每日自动更新-blue)](./digest/README.md)

## 🔔 每日论文日报（自动更新，每天可用）

**每天北京时间 09:00**，GitHub Actions 自动完成：抓取 arXiv 最新 CV 异常检测论文 → DeepSeek 中文解读（标题翻译/一句话总结/方法亮点/★推荐评级）→ 自动生成日报提交到仓库。无需任何人工操作，Star 后每天来仓库看 [digest/](./digest/README.md) 就能跟踪领域最新进展。

**这一步同时也是 RAG 知识库的自动供给**——新论文持续入库，问答系统随之"越用越懂"。

---

面向**计算机视觉异常检测（Industrial Anomaly Detection）**论文库的检索增强生成（RAG）问答系统：上传论文 PDF，用中文提问，系统检索最相关的文献片段并让 LLM 生成**带 [1][2] 引用标注**的回答。

## 架构

```
                     ┌──────────── 摄入链路 ────────────┐
 PDF/TXT/MD ──> Loader ──> Chunker ──> Embedder ──> VectorStore(持久化)
              (pypdf抽取)  (递归分割    (三级降级)     (NumPy余弦)
                           800/120)

                     ┌──────────── 问答链路 ────────────┐
 中文问题 ──> Embedder ──> Retriever ──> Generator ──> 带引用的回答
              (查询嵌入)   (top-k+阈值过滤)  (DeepSeek LLM)
```

### 嵌入三级降级（系统在无网/无 Key 环境也能跑）
1. **API 嵌入**：OpenAI 兼容 `/embeddings`（需配置 `PM_EMBED_API_KEY`）
2. **本地语义嵌入**：fastembed + ONNX Runtime，默认多语言模型
   `paraphrase-multilingual-MiniLM-L12-v2`，支持**中文问句 ↔ 英文论文**跨语言检索
3. **哈希兜底**：词/字符 n-gram 哈希投影，纯离线（仅词面匹配，无语义）

## 快速开始

```bash
pip install -r requirements.txt

# 1) 下载 CV 异常检测经典论文（PatchCore/PaDiM/SPADE/CutPaste/DRAEM 等 8 篇）
python scripts/download_papers.py

# 2) 启动（自动加载 data/docs 并摄入）
python app.py            # 打开 http://127.0.0.1:5000
```

命令行直接问答：

```bash
python -c "
from papermind.pipeline import RAGPipeline
pipe = RAGPipeline()
r = pipe.query('PatchCore 如何用 memory bank 做异常检测?')
print(r['answer']); print(r['sources'])"
```

## 配置（.env，已 gitignore）

```ini
PM_API_KEY=sk-xxx                # DeepSeek（生成模型）
PM_BASE_URL=https://api.deepseek.com/v1
PM_CHAT_MODEL=deepseek-chat
PM_LOCAL_EMBED_MODEL=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
HF_ENDPOINT=https://hf-mirror.com   # 模型权重国内镜像
```

## 目录结构

```
papermind/
├── papermind/          # 核心包
│   ├── config.py       # 集中配置 + .env 加载
│   ├── loader.py       # PDF/TXT/MD 加载
│   ├── chunker.py      # 递归字符分块（可重叠）
│   ├── embeddings.py   # 三级降级嵌入
│   ├── vectorstore.py  # NumPy 向量库（余弦检索+持久化）
│   ├── retriever.py    # top-k 召回 + 阈值过滤
│   ├── generator.py    # LLM 生成（引用约束/抽取式兜底）
│   └── pipeline.py     # RAG 编排
├── templates/index.html# Web UI
├── scripts/download_papers.py
└── app.py              # Flask 服务
```

## 设计取舍（面试可讲）

| 决策 | 理由 |
|---|---|
| NumPy 暴力余弦而非 FAISS | 论文库量级小（千级块），暴力扫描 <10ms；全流程透明可解释 |
| 检索阈值过滤（0.30） | 宁缺毋滥，低相关片段会诱导 LLM 幻觉 |
| 分块 800 字符 + 120 重叠 | 论文段落级语义完整；重叠避免关键句被截断 |
| 提示词强制引用编号 | 回答可溯源到具体 chunk，可核验、可评估 |
| 多语言 MiniLM 本地嵌入 | 中文提问英文论文；ONNX 推理免 torch，部署轻 |

## 局限与改进方向

- 检索为纯稠密向量，可加 BM25 混合检索（Hybrid Search）提升召回
- 图表内容无法解析（PDF 只抽文本），可引入多模态文档解析
- 全量重建索引，可改增量摄入 + 按 source 去重更新

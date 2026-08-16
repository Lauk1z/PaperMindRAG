# 📚 PaperMind — 学术论文 RAG 智能问答系统

上传论文 PDF，向量化建库，用大模型回答你的问题——**每个答案都带引用来源，可溯源核查**。

一个为**研究生复试**打造的 RAG 全链路实现：代码逐模块拆解、关键位置附面试考点注释、仓库内含《复试 RAG 高频面试题与回答思路》。

## ✨ 特性

- **完整 RAG 链路**：解析 → 递归分块 → 向量化 → 余弦检索 → 带引用生成，每个环节独立模块，可单独讲解与替换
- **引用溯源**：答案标注 `[1][2]` 来源（文档名 + 页码 + 相关度），这是控制幻觉的关键设计
- **零重依赖**：向量检索用 numpy 手写（面试能讲清余弦相似度公式），不依赖 FAISS/LangChain 黑盒
- **多后端**：嵌入与生成均支持任意 OpenAI 兼容接口（OpenAI / DeepSeek / 通义千问 / 本地 Ollama）；无 API key 也能跑通流程（降级为本地嵌入 + 抽取式回答）
- **Web 界面**：拖拽上传、实时问答、知识库管理

## 🚀 快速开始

```bash
pip install -r requirements.txt

# 可选：配置大模型（不配也能跑，仅检索模式）
export PM_API_KEY=sk-xxxx
export PM_BASE_URL=https://api.deepseek.com/v1   # 或其他OpenAI兼容接口
export PM_CHAT_MODEL=deepseek-chat
export PM_EMBED_MODEL=text-embedding-3-small

python app.py
# 打开 http://localhost:8899
```

**使用流程**：拖入论文 PDF → 自动切块建索引 → 提问 → 得到带引用的回答。

## 🏗️ 架构（面试讲解顺序）

```
离线索引阶段                          在线问答阶段
┌────────┐   ┌────────┐   ┌────────┐     ┌────────┐   ┌────────┐   ┌────────┐
│ loader │ → │chunker │ → │embedder│     │embedder│ → │retriever│ → │generator│
│ PDF解析 │   │ 递归分块│   │ 向量化  │     │问题向量化│  │ Top-K检索│   │带引用生成 │
└────────┘   └────────┘   └───┬────┘     └────────┘   └────────┘   └────────┘
                              ↓               ↑
                        ┌──────────────────────────┐
                        │      vectorstore          │
                        │  向量存储 + 余弦相似度检索   │
                        └──────────────────────────┘
```

| 模块 | 文件 | 面试考点 |
|------|------|---------|
| 文档解析 | [loader.py](papermind/loader.py) | PDF解析方案对比、扫描版OCR、版面分析 |
| 递归分块 | [chunker.py](papermind/chunker.py) | 为什么分块、chunk_size怎么定、overlap作用 |
| 嵌入 | [embeddings.py](papermind/embeddings.py) | Embedding原理、向量vs关键词检索、混合检索 |
| 向量库 | [vectorstore.py](papermind/vectorstore.py) | 余弦相似度公式、HNSW/IVF索引、向量库选型 |
| 检索 | [retriever.py](papermind/retriever.py) | 查询改写、多路召回、重排序、HyDE |
| 生成 | [generator.py](papermind/generator.py) | Prompt工程、幻觉控制、引用标注 |
| 管线 | [pipeline.py](papermind/pipeline.py) | RAG全流程串讲 |

## 🎓 复试准备

**必读**：[INTERVIEW.md](INTERVIEW.md) — 整理了 20+ 道 RAG 高频面试题与回答思路，覆盖原理、工程、调优、前沿方向，并标注了本项目中对应的代码位置（面试时可以说"这个我在我的项目里是这样实现的…"）。

## ⚙️ 可调参数

全部集中在 [config.py](papermind/config.py)：`chunk_size`（分块大小）、`chunk_overlap`（重叠）、`top_k`（召回数）、`sim_threshold`（相关度阈值）——改参数观察效果变化，就是最好的调参练习。

## 📁 目录结构

```
papermind/
├── app.py                 # Flask Web服务
├── templates/index.html   # 聊天界面
├── papermind/
│   ├── config.py          # 全局配置（含参数原理注释）
│   ├── loader.py          # 文档解析
│   ├── chunker.py         # 递归分块
│   ├── embeddings.py      # 嵌入（API/本地兜底）
│   ├── vectorstore.py     # 向量存储与检索
│   ├── retriever.py       # 检索策略
│   ├── generator.py       # 带引用生成
│   └── pipeline.py        # RAG管线
├── INTERVIEW.md           # 复试面试题与回答思路
└── requirements.txt
```

## License

MIT

"""生成模块：把检索到的文献片段 + 用户问题交给 LLM，生成带引用的回答。

设计要点（RAG 减少幻觉的关键约束）：
1. System Prompt 明确要求"仅基于给定文献片段回答"；
2. 片段带编号，要求回答中用 [1][2] 标注引用；
3. 检索为空或片段不含答案时，明确说"文献中未提及"，禁止自由发挥。
"""
from typing import List

from .config import Config

SYSTEM_PROMPT = (
    "你是论文阅读助手。请仅基于下面给出的文献片段回答问题，"
    "并在对应结论后用 [编号] 标注引用来源（如 [1][2]）。"
    "如果片段不足以回答，请直接说明\"提供的文献片段中未涉及该问题\"，"
    "不要编造。回答使用中文，专业术语可保留英文。"
)


class Generator:
    def __init__(self, config: Config):
        self.config = config

    # ---------------- 对外主入口 ----------------
    def generate(self, question: str, contexts: List[dict]) -> dict:
        if not contexts:
            return {"answer": "未在知识库中检索到与问题相关的片段，"
                              "请先上传文档或换个问法。", "model": "rule"}
        if not self.config.api_key:
            return self._extractive(question, contexts)

        context_text = "\n\n".join(
            f"[{i + 1}] 来源: {c['source']} (块{c['seq']})\n{c['text']}"
            for i, c in enumerate(contexts))
        user_prompt = f"文献片段：\n{context_text}\n\n问题：{question}"
        answer = self._chat(
            [{"role": "system", "content": SYSTEM_PROMPT},
             {"role": "user", "content": user_prompt}])
        return {"answer": answer, "model": self.config.chat_model}

    # ---------------- LLM 调用（OpenAI 兼容协议） ----------------
    def _chat(self, messages: List[dict]) -> str:
        import requests
        resp = requests.post(
            f"{self.config.base_url.rstrip('/')}/chat/completions",
            headers={"Authorization": f"Bearer {self.config.api_key}"},
            json={"model": self.config.chat_model,
                  "messages": messages,
                  "temperature": self.config.temperature},
            timeout=self.config.timeout)
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"].strip()

    # ---------------- 无 API Key 时的抽取式兜底 ----------------
    @staticmethod
    def _extractive(question: str, contexts: List[dict]) -> dict:
        """没有 LLM 时，直接把最相关的片段作为回答（纯检索模式）。"""
        lines = ["（未配置 LLM API Key，以下为最相关的原文片段）"]
        for i, c in enumerate(contexts[:3]):
            snippet = c["text"][:300].replace("\n", " ")
            lines.append(f"[{i + 1}] {c['source']}: {snippet}...")
        return {"answer": "\n\n".join(lines), "model": "extractive"}

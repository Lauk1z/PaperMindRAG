"""生成层：把检索到的证据 + 用户问题交给LLM，生成带引用的回答。

【面试必问】RAG如何控制幻觉？
1. Prompt中明确要求"只依据提供的资料回答，资料不足就说不知道"
2. 要求标注引用来源[1][2]，让答案可溯源、可核查
3. 检索端设相似度阈值，宁缺毋滥（retriever.py中实现）
4. 答案后处理：检查引用编号是否真实存在

【面试考点】Prompt工程要点：
- 系统提示定义角色与约束
- 上下文按相关性排序注入
- 输出格式约束（引用标注规范）
"""
import json
import urllib.request
from typing import List, Tuple

SYSTEM_PROMPT = """你是一个严谨的学术论文助手。请严格依据下方【参考资料】回答用户问题：
1. 只使用资料中的信息，禁止编造资料外的内容
2. 每个关键论断后用[编号]标注来源，编号对应资料序号
3. 资料不足以回答时，明确说"根据现有资料无法回答"，并说明缺少什么信息
4. 回答使用中文，条理清晰"""


class Generator:
    def __init__(self, config):
        self.config = config

    def build_prompt(self, question: str,
                     contexts: List[Tuple[str, dict, float]]) -> str:
        """组装最终prompt：系统约束 + 编号上下文 + 问题。"""
        parts = []
        for i, (text, meta, score) in enumerate(contexts, 1):
            src = f"{meta.get('doc_name', '?')} 第{meta.get('page', '?')}页"
            parts.append(f"[{i}] (来源: {src}, 相关度{score:.2f})\n{text}")
        ctx_block = "\n\n".join(parts)
        return (f"{SYSTEM_PROMPT}\n\n【参考资料】\n{ctx_block}\n\n"
                f"【用户问题】\n{question}")

    def generate(self, question: str,
                 contexts: List[Tuple[str, dict, float]]) -> str:
        """调用LLM生成回答。无API key时返回基于证据的抽取式兜底答案。"""
        if not contexts:
            return ("根据现有资料无法回答这个问题。"
                    "（检索未找到足够相关的内容，请确认知识库中是否有相关文献，"
                    "或尝试换一种问法）")

        prompt = self.build_prompt(question, contexts)
        if self.config.api_key:
            try:
                return self._call_llm(prompt)
            except Exception as e:
                return (f"【LLM调用失败: {e}，以下为检索到的原始证据】\n\n"
                        + self._fallback_answer(contexts))
        return self._fallback_answer(contexts)

    def _call_llm(self, prompt: str) -> str:
        """调用OpenAI兼容的 /chat/completions 接口。"""
        req = urllib.request.Request(
            f"{self.config.base_url.rstrip('/')}/chat/completions",
            data=json.dumps({
                "model": self.config.chat_model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.3,  # 低温度 -> 更忠实于资料，减少发散
            }).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.config.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=120) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        return data["choices"][0]["message"]["content"].strip()

    @staticmethod
    def _fallback_answer(contexts) -> str:
        """无LLM时的兜底：直接展示最相关的证据块（抽取式回答）。"""
        lines = ["（未配置LLM API key，以下为检索到的最相关原文片段）\n"]
        for i, (text, meta, score) in enumerate(contexts, 1):
            src = f"{meta.get('doc_name', '?')} 第{meta.get('page', '?')}页"
            snippet = text[:300] + ("..." if len(text) > 300 else "")
            lines.append(f"[{i}] 来源: {src} (相关度{score:.2f})\n{snippet}\n")
        return "\n".join(lines)

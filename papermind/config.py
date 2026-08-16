"""全局配置：所有可调参数集中在这里，便于理解与调优。

【面试考点】RAG系统的效果由一串超参数共同决定：
- chunk_size / overlap 决定"知识切得多碎"
- top_k / threshold 决定"取多少证据、要不要宁缺毋滥"
- 嵌入模型决定"语义理解能力上限"
调参思路是面试高频问题，改这个文件就是最好的练习。
"""
import os
from dataclasses import dataclass, field


def _load_dotenv():
    """加载项目根目录的 .env 文件到环境变量（不覆盖已存在的变量）。

    手写极简解析，避免引入 python-dotenv 依赖。
    .env 已被 .gitignore 排除，API key 不会进入版本库。
    """
    env_path = os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), ".env")
    if not os.path.exists(env_path):
        return
    with open(env_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key, value = key.strip(), value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value


_load_dotenv()


@dataclass
class Config:
    # ---------- 分块参数 ----------
    # chunk_size: 每个文本块的目标长度（字符数）。
    #   太大 -> 检索不精准、塞爆LLM上下文；太小 -> 语义不完整。
    #   经验值300~800，论文类长文本取500左右。
    chunk_size: int = 500
    # overlap: 相邻块的重叠字符数，防止一句话被切断导致检索漏掉关键信息。
    chunk_overlap: int = 80

    # ---------- 检索参数 ----------
    # top_k: 每次问答从向量库取回最相关的k个文本块作为证据。
    top_k: int = 4
    # sim_threshold: 余弦相似度下限，低于它的块视为无关，宁缺毋滥。
    #   这是控制"幻觉"的第一道闸：检索不到相关证据就不强行回答。
    sim_threshold: float = 0.05

    # ---------- 模型后端 ----------
    # 支持任何OpenAI兼容接口：OpenAI / DeepSeek / 通义千问 / 本地Ollama
    # 例(DeepSeek): PM_BASE_URL=https://api.deepseek.com/v1 PM_CHAT_MODEL=deepseek-chat
    api_key: str = field(default_factory=lambda: (
        os.environ.get("PM_API_KEY") or os.environ.get("OPENAI_API_KEY", "")
    ))
    base_url: str = field(default_factory=lambda: os.environ.get(
        "PM_BASE_URL", "https://api.openai.com/v1"))
    embed_model: str = field(default_factory=lambda: os.environ.get(
        "PM_EMBED_MODEL", "text-embedding-3-small"))
    chat_model: str = field(default_factory=lambda: os.environ.get(
        "PM_CHAT_MODEL", "gpt-4o-mini"))

    # ---------- 存储路径 ----------
    data_dir: str = "data"

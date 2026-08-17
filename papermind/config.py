"""全局配置模块。

所有可调参数集中于此；敏感信息（API Key）从环境变量 / .env 读取，
避免硬编码进代码仓库。
"""
import os
from dataclasses import dataclass


def _load_dotenv():
    """加载项目根目录的 .env 文件到环境变量（不覆盖已存在的变量）。"""
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
    """RAG 全流程配置。"""

    # ---------- 生成模型（OpenAI 兼容协议，默认 DeepSeek） ----------
    api_key: str = os.environ.get("PM_API_KEY", "")
    base_url: str = os.environ.get("PM_BASE_URL", "https://api.deepseek.com/v1")
    chat_model: str = os.environ.get("PM_CHAT_MODEL", "deepseek-chat")

    # ---------- 嵌入（可选 API；默认走本地语义模型） ----------
    # OpenAI 兼容 /embeddings 端点（DeepSeek 暂不提供嵌入 API，留空即跳过）
    embed_api_key: str = os.environ.get("PM_EMBED_API_KEY", "")
    embed_base_url: str = os.environ.get("PM_EMBED_BASE_URL", "")
    embed_api_model: str = os.environ.get("PM_EMBED_API_MODEL", "text-embedding-3-small")
    # 本地语义模型：多语言，支持 中文问句 <-> 英文论文 跨语言检索
    local_embed_model: str = os.environ.get(
        "PM_LOCAL_EMBED_MODEL",
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    # 哈希兜底嵌入的维度
    hash_embed_dim: int = int(os.environ.get("PM_HASH_EMBED_DIM", "384"))

    # ---------- 分块 ----------
    chunk_size: int = int(os.environ.get("PM_CHUNK_SIZE", "800"))     # 每块目标字符数
    chunk_overlap: int = int(os.environ.get("PM_CHUNK_OVERLAP", "120"))  # 相邻块重叠

    # ---------- 检索 ----------
    top_k: int = int(os.environ.get("PM_TOP_K", "5"))
    score_threshold: float = float(os.environ.get("PM_SCORE_THRESHOLD", "0.30"))

    # ---------- 生成 ----------
    temperature: float = 0.3
    timeout: int = 60          # LLM 请求超时（秒）

    # ---------- 路径（相对项目根目录） ----------
    data_dir: str = os.environ.get("PM_DATA_DIR", "data/docs")
    index_dir: str = os.environ.get("PM_INDEX_DIR", "data/index")

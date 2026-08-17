"""端到端集成测试：全离线（哈希嵌入 + 抽取式生成），无需网络/模型/Key。

覆盖：摄入->检索->生成主链路、索引持久化、增量更新不重复入库。
"""
import pytest

from papermind.config import Config
from papermind.pipeline import RAGPipeline

ANOMALY_TEXT = (
    "PatchCore uses a memory bank of patch features sampled from a "
    "pretrained network for anomaly detection. The coreset subsampling "
    "reduces the memory bank size while keeping detection accuracy. "
) * 8

COOKING_TEXT = (
    "Boil the pasta for ten minutes. Prepare the tomato sauce with garlic "
    "and olive oil. Serve the dish with grated parmesan cheese. "
) * 8


@pytest.fixture()
def offline_config(tmp_path):
    """临时目录 + 禁用外部依赖的配置。"""
    (tmp_path / "docs").mkdir()
    (tmp_path / "index").mkdir()
    return Config(data_dir=str(tmp_path / "docs"),
                  index_dir=str(tmp_path / "index"),
                  api_key="", score_threshold=0.05)


def make_pipeline(cfg):
    pipe = RAGPipeline(cfg)
    pipe.embedder._api_failed = True   # 强制跳过 API
    pipe.embedder._local_failed = True  # 强制跳过本地模型（CI 无需下载）
    return pipe


def _write(doc_dir, name, text):
    (doc_dir / name).write_text(text, encoding="utf-8")


def test_full_pipeline_offline(offline_config, tmp_path):
    docs = tmp_path / "docs"
    _write(docs, "patchcore.txt", ANOMALY_TEXT)
    _write(docs, "cooking.txt", COOKING_TEXT)

    pipe = make_pipeline(offline_config)
    assert len(pipe.store) > 0
    assert pipe.embedder.mode == "hash"

    result = pipe.query("anomaly detection memory bank")
    assert result["answer"]
    assert result["sources"], "应检索到来源"
    # 相关问题应命中 anomaly 文档而非 cooking
    top = result["sources"][0]
    assert top["source"] == "patchcore.txt", f"检索错文档: {top}"


def test_index_persisted_and_reloadable(offline_config, tmp_path):
    docs = tmp_path / "docs"
    _write(docs, "patchcore.txt", ANOMALY_TEXT)
    pipe = make_pipeline(offline_config)
    n = len(pipe.store)

    # 重新构造 pipeline 应直接加载索引（不再重复摄入）
    pipe2 = make_pipeline(offline_config)
    assert len(pipe2.store) == n


def test_incremental_ingest_no_duplicates(offline_config, tmp_path):
    docs = tmp_path / "docs"
    _write(docs, "patchcore.txt", ANOMALY_TEXT)
    pipe = make_pipeline(offline_config)
    n1 = len(pipe.store)

    # 同一文件再摄入：替换旧块，总量不变
    stats = pipe.ingest()
    assert stats["replaced"] > 0
    assert len(pipe.store) == n1


def test_query_empty_store(offline_config):
    pipe = make_pipeline(offline_config)
    result = pipe.query("anything")
    assert "未在知识库" in result["answer"] or result["sources"] == []

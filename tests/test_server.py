"""Web 服务测试：应用工厂 + 各接口契约（不依赖网络与知识库）。"""
import pytest

from papermind.config import Config
from papermind.server import create_app


@pytest.fixture()
def client(tmp_path, monkeypatch):
    """禁用外部依赖 + 临时上传目录的测试客户端。"""
    (tmp_path / "docs").mkdir()
    cfg = Config(data_dir=str(tmp_path / "docs"),
                 index_dir=str(tmp_path / "index"), api_key="")
    app = create_app(cfg, upload_dir=str(tmp_path / "docs"))
    # 强制嵌入走哈希，测试环境不装 fastembed
    app.config["TESTING"] = True

    with app.test_client() as c:
        # 首请求后再强制降级（pipeline 懒加载）
        yield c


def test_health(client):
    r = client.get("/api/health")
    assert r.status_code == 200 and r.get_json()["ok"] is True


def test_index_page(client):
    assert client.get("/").status_code == 200


def test_query_requires_question(client):
    r = client.post("/api/query", json={"question": "   "})
    assert r.status_code == 400
    assert r.get_json()["error"]


def test_unknown_route_returns_json_404(client):
    r = client.get("/api/nope")
    assert r.status_code == 404
    assert r.get_json()["ok"] is False


def test_upload_and_query_roundtrip(client, tmp_path):
    import io
    data = {"files": (io.BytesIO(b"anomaly detection with memory bank " * 20),
                      "test.txt")}
    r = client.post("/api/ingest", data=data, content_type="multipart/form-data")
    assert r.status_code == 200 and r.get_json()["ok"] is True

    r = client.post("/api/query", json={"question": "anomaly detection memory bank"})
    body = r.get_json()
    assert body["ok"] is True
    assert any("test.txt" in s["source"] for s in body["sources"])

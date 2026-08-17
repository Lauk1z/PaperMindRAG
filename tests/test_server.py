"""Web 服务测试：应用工厂 + 各接口契约（不依赖网络与知识库）。"""
import sqlite3

import pytest

from papermind.config import Config
from papermind.server import create_app


def _test_app(tmp_path):
    (tmp_path / "docs").mkdir()
    cfg = Config(data_dir=str(tmp_path / "docs"),
                 index_dir=str(tmp_path / "index"), api_key="")
    app = create_app(cfg, upload_dir=str(tmp_path / "docs"),
                     env_path=str(tmp_path / ".env"),
                     auth_db_path=str(tmp_path / "users.db"))
    app.config["TESTING"] = True
    return app


def _csrf(client):
    with client.session_transaction() as session:
        return session["csrf_token"]


@pytest.fixture()
def anonymous_client(tmp_path):
    """未登录测试客户端。"""
    app = _test_app(tmp_path)
    with app.test_client() as client:
        yield client


@pytest.fixture()
def client(tmp_path):
    """已登录客户端；业务接口测试不依赖外部 OAuth。"""
    app = _test_app(tmp_path)

    with app.test_client() as test_client:
        test_client.get("/login")
        response = test_client.post(
            "/auth/register",
            json={"email": "tester@example.com", "password": "safe-pass-123"},
            headers={"X-CSRF-Token": _csrf(test_client)},
        )
        assert response.status_code == 200
        yield test_client


def test_health(client):
    r = client.get("/api/health")
    assert r.status_code == 200 and r.get_json()["ok"] is True


def test_index_page(client):
    r = client.get("/")
    assert r.status_code == 200
    html = r.get_data(as_text=True)
    assert 'data-testid="papermind-workbench"' in html
    assert "论文知识库" in html
    assert "/api/ingest" in html and "/api/query" in html
    assert 'id="themeToggle"' in html and "/api/config" in html
    assert 'class="brand-symbol"' in html
    assert 'class="upload-icon-symbol"' in html
    assert 'id="accountSlot"' in html and 'id="loginLink"' in html
    assert 'id="directFileNotice"' in html
    assert "{% if current_user %}" not in html


def test_auth_gate_and_login_page(anonymous_client):
    r = anonymous_client.get("/")
    assert r.status_code == 302 and r.headers["Location"].endswith("/login")
    assert anonymous_client.get("/api/stats").status_code == 401

    r = anonymous_client.get("/login")
    html = r.get_data(as_text=True)
    assert r.status_code == 200
    assert 'data-testid="auth-page"' in html
    assert "GitHub 登录" in html
    assert "Microsoft 登录" in html
    assert "Google 登录" in html


def test_email_register_login_and_logout(anonymous_client):
    anonymous_client.get("/login")
    payload = {
        "display_name": "Paper Reader",
        "email": "reader@example.com",
        "password": "a-secure-password",
    }
    r = anonymous_client.post(
        "/auth/register", json=payload,
        headers={"X-CSRF-Token": _csrf(anonymous_client)},
    )
    assert r.status_code == 200
    me = anonymous_client.get("/auth/me").get_json()
    assert me["user"]["email"] == payload["email"]
    assert me["csrf_token"]
    assert me["can_configure"] is True

    db_path = anonymous_client.application.config["PM_AUTH_DB_PATH"]
    with sqlite3.connect(db_path) as conn:
        password_hash = conn.execute(
            "SELECT password_hash FROM users WHERE email = ?", (payload["email"],)
        ).fetchone()[0]
    assert payload["password"] not in password_hash

    r = anonymous_client.post(
        "/auth/logout", headers={"X-CSRF-Token": _csrf(anonymous_client)}
    )
    assert r.status_code == 200
    assert anonymous_client.get("/auth/me").status_code == 401

    anonymous_client.get("/login")
    r = anonymous_client.post(
        "/auth/login",
        json={"email": payload["email"], "password": "wrong-password"},
        headers={"X-CSRF-Token": _csrf(anonymous_client)},
    )
    assert r.status_code == 401
    r = anonymous_client.post(
        "/auth/login",
        json={"email": payload["email"], "password": payload["password"]},
        headers={"X-CSRF-Token": _csrf(anonymous_client)},
    )
    assert r.status_code == 200


def test_registration_validation_and_csrf(anonymous_client):
    anonymous_client.get("/login")
    payload = {"email": "invalid", "password": "short"}
    assert anonymous_client.post("/auth/register", json=payload).status_code == 403
    r = anonymous_client.post(
        "/auth/register", json=payload,
        headers={"X-CSRF-Token": _csrf(anonymous_client)},
    )
    assert r.status_code == 400


def test_unconfigured_oauth_provider_returns_to_login(anonymous_client):
    r = anonymous_client.get("/auth/oauth/github")
    assert r.status_code == 302
    assert "/login?error=" in r.headers["Location"]


def test_github_oauth_start_uses_authorization_flow(tmp_path, monkeypatch):
    monkeypatch.setenv("PM_GITHUB_CLIENT_ID", "github-client-id")
    monkeypatch.setenv("PM_GITHUB_CLIENT_SECRET", "github-client-secret")
    app = _test_app(tmp_path)
    with app.test_client() as test_client:
        r = test_client.get("/auth/oauth/github")
        assert r.status_code == 302
        assert r.headers["Location"].startswith("https://github.com/login/oauth/authorize?")
        assert "client_id=github-client-id" in r.headers["Location"]
        assert "code_challenge_method=S256" in r.headers["Location"]
        assert "state=" in r.headers["Location"]


def test_oauth_callback_uses_forwarded_https_on_render(tmp_path, monkeypatch):
    monkeypatch.setenv("PM_TRUST_PROXY", "1")
    monkeypatch.setenv("PM_GITHUB_CLIENT_ID", "github-client-id")
    monkeypatch.setenv("PM_GITHUB_CLIENT_SECRET", "github-client-secret")
    app = _test_app(tmp_path)
    with app.test_client() as test_client:
        r = test_client.get(
            "/auth/oauth/github",
            headers={
                "X-Forwarded-Proto": "https",
                "X-Forwarded-Host": "papermindrag-lauk1z.onrender.com",
            },
        )
    assert r.status_code == 302
    assert "redirect_uri=https%3A%2F%2Fpapermindrag-lauk1z.onrender.com" in (
        r.headers["Location"]
    )


def test_stats_exposes_ui_capabilities(client):
    body = client.get("/api/stats").get_json()
    assert body["ok"] is True
    assert body["supported_extensions"] == [".pdf", ".txt", ".md"]
    assert body["max_upload_mb"] == 64
    assert body["retrieval_mode"] in {"hybrid", "dense", "bm25"}


def test_upload_requires_supported_file(client):
    import io

    r = client.post("/api/ingest", data={}, content_type="multipart/form-data")
    assert r.status_code == 400

    data = {"files": (io.BytesIO(b"not a paper"), "notes.exe")}
    r = client.post("/api/ingest", data=data, content_type="multipart/form-data")
    assert r.status_code == 400
    assert r.get_json()["files"] == ["notes.exe"]


def test_remote_config_requires_admin(client, monkeypatch):
    r = client.get("/api/config", environ_base={"REMOTE_ADDR": "192.168.1.8"})
    assert r.status_code == 403

    monkeypatch.setenv("PM_ADMIN_EMAIL", "tester@example.com")
    r = client.get("/api/config", environ_base={"REMOTE_ADDR": "192.168.1.8"})
    assert r.status_code == 200


def test_config_update_requires_csrf(client):
    r = client.put("/api/config", json={"chat_model": "example-chat"})
    assert r.status_code == 403


def test_config_saves_without_returning_secrets(client):
    env_path = client.application.config["PM_ENV_PATH"]
    with open(env_path, "w", encoding="utf-8") as env_file:
        env_file.write("# keep this setting\nPM_LOG_LEVEL=DEBUG")

    payload = {
        "api_key": "chat-secret-value",
        "base_url": "https://llm.example.test/v1",
        "chat_model": "example-chat",
        "embed_api_key": "embed-secret-value",
        "embed_base_url": "https://embed.example.test/v1",
        "embed_api_model": "example-embed",
    }
    r = client.put(
        "/api/config", json=payload,
        headers={"X-CSRF-Token": _csrf(client)},
    )
    assert r.status_code == 200
    assert r.get_json()["api_key_configured"] is True

    saved = open(env_path, encoding="utf-8").read()
    assert "# keep this setting\nPM_LOG_LEVEL=DEBUG\n" in saved
    assert "PM_API_KEY=chat-secret-value" in saved
    assert "PM_EMBED_API_KEY=embed-secret-value" in saved

    body = client.get("/api/config").get_json()
    assert body["base_url"] == "https://llm.example.test/v1"
    assert body["chat_model"] == "example-chat"
    assert "chat-secret-value" not in str(body)
    assert "embed-secret-value" not in str(body)
    assert client.application.config["PM_CONFIG"].api_key == "chat-secret-value"
    assert client.application.extensions["rag_pipeline"] is None


def test_config_validates_url_and_can_clear_key(client):
    headers = {"X-CSRF-Token": _csrf(client)}
    r = client.put(
        "/api/config", json={"base_url": "file:///tmp/model"}, headers=headers
    )
    assert r.status_code == 400

    client.put("/api/config", json={"api_key": "temporary-secret"}, headers=headers)
    r = client.put("/api/config", json={"clear_api_key": True}, headers=headers)
    assert r.status_code == 200
    assert r.get_json()["api_key_configured"] is False
    saved = open(client.application.config["PM_ENV_PATH"], encoding="utf-8").read()
    assert "PM_API_KEY" not in saved


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

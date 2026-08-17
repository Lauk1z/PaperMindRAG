import os
import time

from flask import Flask
import requests

from papermind.desktop import LocalServer, configure_desktop_environment


def test_configure_desktop_environment_uses_persistent_paths(tmp_path, monkeypatch):
    for key in (
        "PM_DATA_ROOT",
        "PM_DATA_DIR",
        "PM_INDEX_DIR",
        "PM_ENV_PATH",
        "PM_COOKIE_SECURE",
        "PM_AUTH_REQUIRED",
        "XDG_CACHE_HOME",
    ):
        monkeypatch.delenv(key, raising=False)

    paths = configure_desktop_environment(tmp_path / "PaperMind")

    assert paths.documents.is_dir()
    assert paths.index.is_dir()
    assert paths.webview.is_dir()
    assert os.environ["PM_DATA_ROOT"] == str(paths.data)
    assert os.environ["PM_DATA_DIR"] == str(paths.documents)
    assert os.environ["PM_INDEX_DIR"] == str(paths.index)
    assert os.environ["PM_ENV_PATH"] == str(paths.env_file)
    assert os.environ["PM_COOKIE_SECURE"] == "0"
    assert os.environ["PM_AUTH_REQUIRED"] == "1"


def test_local_server_uses_loopback_and_stops():
    app = Flask(__name__)

    @app.get("/health")
    def health():
        return {"ok": True}

    server = LocalServer(app)
    assert server.url.startswith("http://127.0.0.1:")

    server.start()
    try:
        response = requests.get(f"{server.url}health", timeout=3)
        assert response.json() == {"ok": True}
    finally:
        server.stop()

    for _ in range(20):
        if not server.running:
            break
        time.sleep(0.01)
    assert not server.running

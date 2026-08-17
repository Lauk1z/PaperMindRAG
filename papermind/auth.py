"""本地账户与第三方 OAuth 登录。"""
from datetime import timedelta
from ipaddress import ip_address
import os
import re
import secrets
import sqlite3

from authlib.integrations.flask_client import OAuth
from flask import (Blueprint, current_app, g, jsonify, redirect, render_template,
                   request, session, url_for)
from werkzeug.security import check_password_hash, generate_password_hash


auth_bp = Blueprint("auth", __name__)
EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


def _connect():
    conn = sqlite3.connect(current_app.config["PM_AUTH_DB_PATH"])
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def _user_dict(row):
    if row is None:
        return None
    return {
        "id": row["id"],
        "email": row["email"],
        "display_name": row["display_name"],
        "avatar_url": row["avatar_url"] or "",
    }


def _load_user(user_id):
    with _connect() as conn:
        row = conn.execute(
            "SELECT id, email, display_name, avatar_url FROM users WHERE id = ?",
            (user_id,),
        ).fetchone()
    return _user_dict(row)


def ensure_csrf_token():
    token = session.get("csrf_token")
    if not token:
        token = secrets.token_urlsafe(32)
        session["csrf_token"] = token
    return token


def valid_csrf():
    supplied = request.headers.get("X-CSRF-Token", "")
    expected = session.get("csrf_token", "")
    return bool(supplied and expected and secrets.compare_digest(supplied, expected))


def _is_loopback(remote_addr):
    try:
        addr = ip_address((remote_addr or "").split("%", 1)[0])
    except ValueError:
        return False
    return addr.is_loopback or bool(
        addr.version == 6 and addr.ipv4_mapped and addr.ipv4_mapped.is_loopback
    )


def can_manage_config():
    """本机用户可配置；公网部署仅允许指定管理员邮箱。"""
    if _is_loopback(request.remote_addr):
        return True
    admin_email = os.environ.get("PM_ADMIN_EMAIL", "").strip().lower()
    return bool(g.user and admin_email and g.user["email"].lower() == admin_email)


def _sign_in(user_id):
    session.clear()
    session["user_id"] = user_id
    session.permanent = True
    ensure_csrf_token()


def _provider_status():
    return current_app.extensions["pm_oauth_providers"]


def _oauth_profile(provider, remote, token):
    if provider == "github":
        response = remote.get("user", token=token)
        response.raise_for_status()
        profile = response.json()
        email = profile.get("email")
        if not email:
            response = remote.get("user/emails", token=token)
            response.raise_for_status()
            emails = response.json()
            verified = [item for item in emails if item.get("verified")]
            preferred = next((item for item in verified if item.get("primary")), None)
            email = (preferred or (verified[0] if verified else {})).get("email")
        return {
            "provider_id": str(profile.get("id") or ""),
            "email": email,
            "display_name": profile.get("name") or profile.get("login"),
            "avatar_url": profile.get("avatar_url") or "",
        }

    profile = token.get("userinfo") or {}
    if provider == "google" and profile.get("email_verified") is not True:
        raise ValueError("Google 账户邮箱尚未验证")
    email = profile.get("email") or profile.get("preferred_username")
    return {
        "provider_id": str(profile.get("oid") or profile.get("sub") or ""),
        "email": email,
        "display_name": profile.get("name") or email,
        "avatar_url": profile.get("picture") or "",
    }


def _oauth_user(provider, profile):
    provider_id = profile["provider_id"]
    email = (profile.get("email") or "").strip().lower()
    if not provider_id or not EMAIL_RE.fullmatch(email):
        raise ValueError("第三方账户没有提供可用邮箱")

    with _connect() as conn:
        row = conn.execute(
            """SELECT users.id, users.email, users.display_name, users.avatar_url
               FROM user_identities
               JOIN users ON users.id = user_identities.user_id
               WHERE provider = ? AND provider_user_id = ?""",
            (provider, provider_id),
        ).fetchone()
        if row:
            return _user_dict(row)

        existing = conn.execute(
            "SELECT id FROM users WHERE email = ? COLLATE NOCASE", (email,)
        ).fetchone()
        if existing:
            raise ValueError("该邮箱已注册，请先使用邮箱密码登录")

        cursor = conn.execute(
            """INSERT INTO users (email, password_hash, display_name, avatar_url)
               VALUES (?, NULL, ?, ?)""",
            (email, (profile.get("display_name") or email).strip()[:80],
             profile.get("avatar_url") or ""),
        )
        user_id = cursor.lastrowid
        conn.execute(
            """INSERT INTO user_identities (provider, provider_user_id, user_id)
               VALUES (?, ?, ?)""",
            (provider, provider_id, user_id),
        )
    return _load_user(user_id)


@auth_bp.route("/login")
def login_page():
    if g.user and current_app.config["PM_AUTH_REQUIRED"]:
        return redirect(url_for("index"))
    return render_template(
        "login.html",
        csrf_token=ensure_csrf_token(),
        providers=_provider_status(),
        oauth_error=request.args.get("error", ""),
    )


@auth_bp.route("/auth/register", methods=["POST"])
def register():
    if not valid_csrf():
        return jsonify({"ok": False, "error": "页面已过期，请刷新后重试"}), 403
    data = request.get_json(silent=True) or {}
    email = str(data.get("email") or "").strip().lower()
    password = str(data.get("password") or "")
    display_name = str(data.get("display_name") or "").strip()
    if not EMAIL_RE.fullmatch(email):
        return jsonify({"ok": False, "error": "请输入有效邮箱地址"}), 400
    if not 8 <= len(password) <= 128:
        return jsonify({"ok": False, "error": "密码长度需要为 8–128 个字符"}), 400
    display_name = (display_name or email.split("@", 1)[0])[:80]

    try:
        with _connect() as conn:
            cursor = conn.execute(
                """INSERT INTO users (email, password_hash, display_name)
                   VALUES (?, ?, ?)""",
                (email, generate_password_hash(password), display_name),
            )
            user_id = cursor.lastrowid
    except sqlite3.IntegrityError:
        return jsonify({"ok": False, "error": "该邮箱已注册"}), 409

    _sign_in(user_id)
    return jsonify({"ok": True, "user": _load_user(user_id)})


@auth_bp.route("/auth/login", methods=["POST"])
def login():
    if not valid_csrf():
        return jsonify({"ok": False, "error": "页面已过期，请刷新后重试"}), 403
    data = request.get_json(silent=True) or {}
    email = str(data.get("email") or "").strip().lower()
    password = str(data.get("password") or "")
    with _connect() as conn:
        row = conn.execute(
            """SELECT id, email, password_hash, display_name, avatar_url
               FROM users WHERE email = ? COLLATE NOCASE""",
            (email,),
        ).fetchone()
    if not row or not row["password_hash"] or not check_password_hash(
            row["password_hash"], password):
        return jsonify({"ok": False, "error": "邮箱或密码错误"}), 401
    _sign_in(row["id"])
    return jsonify({"ok": True, "user": _user_dict(row)})


@auth_bp.route("/auth/logout", methods=["POST"])
def logout():
    if not valid_csrf():
        return jsonify({"ok": False, "error": "页面已过期，请刷新后重试"}), 403
    session.clear()
    return jsonify({"ok": True})


@auth_bp.route("/auth/me")
def me():
    if not g.user:
        return jsonify({"ok": False, "error": "尚未登录"}), 401
    return jsonify({
        "ok": True,
        "user": g.user,
        "csrf_token": ensure_csrf_token(),
        "can_configure": can_manage_config(),
    })


@auth_bp.route("/auth/oauth/<provider>")
def oauth_start(provider):
    if provider not in _provider_status():
        return redirect(url_for("auth.login_page", error="不支持该登录方式"))
    if not _provider_status()[provider]:
        return redirect(url_for("auth.login_page", error=f"{provider.title()} 登录尚未配置"))
    remote = current_app.extensions["pm_oauth"].create_client(provider)
    callback = url_for("auth.oauth_callback", provider=provider, _external=True)
    return remote.authorize_redirect(callback)


@auth_bp.route("/auth/oauth/<provider>/callback")
def oauth_callback(provider):
    if provider not in _provider_status() or not _provider_status()[provider]:
        return redirect(url_for("auth.login_page", error="登录方式未配置"))
    try:
        remote = current_app.extensions["pm_oauth"].create_client(provider)
        token = remote.authorize_access_token()
        user = _oauth_user(provider, _oauth_profile(provider, remote, token))
        _sign_in(user["id"])
        return redirect(url_for("index"))
    except Exception as error:
        current_app.logger.warning("%s OAuth 登录失败: %s", provider, error)
        message = str(error) if isinstance(error, ValueError) else "第三方登录失败，请重试"
        return redirect(url_for("auth.login_page", error=message))


def _session_secret(db_path):
    configured = os.environ.get("PM_SECRET_KEY", "")
    if configured:
        if len(configured) < 32:
            raise ValueError("PM_SECRET_KEY 至少需要 32 个字符")
        return configured
    secret_path = os.path.join(os.path.dirname(os.path.abspath(db_path)), ".session_secret")
    os.makedirs(os.path.dirname(secret_path), exist_ok=True)
    try:
        with open(secret_path, encoding="utf-8") as secret_file:
            saved = secret_file.read().strip()
        if saved:
            return saved
    except FileNotFoundError:
        pass
    secret = secrets.token_urlsafe(48)
    try:
        with open(secret_path, "x", encoding="utf-8") as secret_file:
            secret_file.write(secret)
    except FileExistsError:
        with open(secret_path, encoding="utf-8") as secret_file:
            return secret_file.read().strip()
    return secret


def _init_db(path):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with sqlite3.connect(path) as conn:
        conn.executescript(
            """
            PRAGMA foreign_keys = ON;
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                email TEXT NOT NULL UNIQUE COLLATE NOCASE,
                password_hash TEXT,
                display_name TEXT NOT NULL,
                avatar_url TEXT,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );
            CREATE TABLE IF NOT EXISTS user_identities (
                provider TEXT NOT NULL,
                provider_user_id TEXT NOT NULL,
                user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                PRIMARY KEY (provider, provider_user_id)
            );
            """
        )


def init_auth(app, db_path, required=True):
    """初始化认证数据库、会话和 OAuth 客户端。"""
    app.config.update(
        SECRET_KEY=_session_secret(db_path),
        PERMANENT_SESSION_LIFETIME=timedelta(days=7),
        SESSION_COOKIE_HTTPONLY=True,
        SESSION_COOKIE_SAMESITE="Lax",
        SESSION_COOKIE_SECURE=os.environ.get("PM_COOKIE_SECURE") == "1",
        PM_AUTH_DB_PATH=db_path,
        PM_AUTH_REQUIRED=required,
    )
    _init_db(db_path)

    oauth = OAuth(app)
    provider_env = {
        "github": ("PM_GITHUB_CLIENT_ID", "PM_GITHUB_CLIENT_SECRET"),
        "google": ("PM_GOOGLE_CLIENT_ID", "PM_GOOGLE_CLIENT_SECRET"),
        "microsoft": ("PM_MICROSOFT_CLIENT_ID", "PM_MICROSOFT_CLIENT_SECRET"),
    }
    providers = {
        name: bool(os.environ.get(client_id) and os.environ.get(client_secret))
        for name, (client_id, client_secret) in provider_env.items()
    }
    app.extensions["pm_oauth_providers"] = providers

    oauth.register(
        "github",
        client_id=os.environ.get("PM_GITHUB_CLIENT_ID"),
        client_secret=os.environ.get("PM_GITHUB_CLIENT_SECRET"),
        access_token_url="https://github.com/login/oauth/access_token",
        authorize_url="https://github.com/login/oauth/authorize",
        api_base_url="https://api.github.com/",
        client_kwargs={
            "scope": "read:user user:email",
            "code_challenge_method": "S256",
        },
    )
    oauth.register(
        "google",
        client_id=os.environ.get("PM_GOOGLE_CLIENT_ID"),
        client_secret=os.environ.get("PM_GOOGLE_CLIENT_SECRET"),
        server_metadata_url="https://accounts.google.com/.well-known/openid-configuration",
        client_kwargs={
            "scope": "openid profile email",
            "code_challenge_method": "S256",
        },
    )
    tenant = os.environ.get("PM_MICROSOFT_TENANT", "common")
    oauth.register(
        "microsoft",
        client_id=os.environ.get("PM_MICROSOFT_CLIENT_ID"),
        client_secret=os.environ.get("PM_MICROSOFT_CLIENT_SECRET"),
        server_metadata_url=(
            f"https://login.microsoftonline.com/{tenant}/v2.0/"
            ".well-known/openid-configuration"
        ),
        client_kwargs={
            "scope": "openid profile email User.Read",
            "code_challenge_method": "S256",
        },
    )
    app.extensions["pm_oauth"] = oauth
    app.register_blueprint(auth_bp)

    @app.before_request
    def load_authenticated_user():
        g.user = _load_user(session.get("user_id")) if session.get("user_id") else None
        if session.get("user_id") and not g.user:
            session.clear()

    @app.before_request
    def require_authentication():
        if not app.config["PM_AUTH_REQUIRED"] or g.user:
            return None
        endpoint = request.endpoint or ""
        if endpoint == "health" or endpoint == "static" or endpoint.startswith("auth."):
            return None
        if request.path.startswith("/api/"):
            return jsonify({"ok": False, "error": "请先登录"}), 401
        return redirect(url_for("auth.login_page"))

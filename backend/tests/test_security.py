"""The gateway refuses anything that is not our own page talking to us."""
import pytest
from fastapi.testclient import TestClient

from app import config, security
from app.main import app

BASE = f"http://127.0.0.1:{config.PORT}"


@pytest.fixture(autouse=True)
def fresh_limiter():
    security.limiter._hits.clear()


@pytest.fixture
def anon():
    return TestClient(app, base_url=BASE)


@pytest.fixture
def client():
    c = TestClient(app, base_url=BASE)
    token = c.get("/api/session").json()["csrf_token"]
    c.headers[security.CSRF_HEADER] = token
    return c


def test_api_needs_a_session(anon):
    assert anon.get("/api/gallery").status_code == 401
    assert anon.post("/api/gallery", json={"name": "x"}).status_code == 401
    assert anon.delete("/api/gallery").status_code == 401
    assert anon.get("/api/people/1/thumb").status_code == 401


def test_health_and_session_are_public(anon):
    assert anon.get("/api/health").status_code == 200
    res = anon.get("/api/session")
    assert res.status_code == 200
    cookie = res.headers["set-cookie"].lower()
    assert "httponly" in cookie and "samesite=strict" in cookie


def test_session_allows_reads(client):
    assert client.get("/api/gallery").status_code == 200


def test_writes_need_the_csrf_token(client):
    del client.headers[security.CSRF_HEADER]
    assert client.post("/api/gallery", json={"name": "x"}).status_code == 403
    client.headers[security.CSRF_HEADER] = "wrong"
    assert client.post("/api/gallery", json={"name": "x"}).status_code == 403


def test_wrong_cookie_is_refused(anon):
    anon.cookies.set(security.SESSION_COOKIE, "guess")
    assert anon.get("/api/gallery").status_code == 401


def test_unknown_host_is_refused():
    # DNS rebinding: the attacker's domain resolves to 127.0.0.1.
    evil = TestClient(app, base_url="http://evil.example:8000")
    assert evil.get("/api/health").status_code == 400
    assert evil.get("/").status_code == 400


def test_cross_site_requests_are_refused(client):
    assert client.get("/api/gallery", headers={"Origin": "http://evil.example"}).status_code == 403
    assert client.get("/api/gallery", headers={"Sec-Fetch-Site": "cross-site"}).status_code == 403
    assert client.get("/api/session", headers={"Sec-Fetch-Site": "cross-site"}).status_code == 403
    assert client.get("/api/gallery", headers={"Origin": BASE, "Sec-Fetch-Site": "same-origin"}).status_code == 200


def test_no_cors_headers_are_ever_sent(client):
    res = client.get("/api/gallery", headers={"Origin": BASE})
    assert "access-control-allow-origin" not in res.headers


def test_security_headers_present(client):
    res = client.get("/api/gallery")
    assert res.headers["x-content-type-options"] == "nosniff"
    assert res.headers["x-frame-options"] == "DENY"
    assert "frame-ancestors 'none'" in res.headers["content-security-policy"]
    assert res.headers["cache-control"] == "no-store"
    denied = TestClient(app, base_url=BASE).get("/api/gallery")
    assert denied.headers["x-content-type-options"] == "nosniff"


def test_docs_are_disabled(client):
    for path in ("/docs", "/redoc", "/openapi.json"):
        res = client.get(path)
        assert "swagger" not in res.text.lower() and "openapi" not in res.text.lower()


def test_oversized_request_is_refused_early(client):
    res = client.post("/api/identify", headers={"Content-Length": str(config.MAX_REQUEST_BYTES + 1)}, content=b"")
    assert res.status_code == 413


def test_bad_uploads_are_refused(client):
    gid = client.post("/api/gallery", json={"name": "Team"}).json()["id"]
    url = f"/api/gallery/{gid}/people"
    exe = client.post(url, files={"file": ("a.exe", b"MZ")}, data={"filename": "a.exe"})
    assert exe.status_code == 415
    empty = client.post(url, files={"file": ("a.jpg", b"")}, data={"filename": "a.jpg"})
    assert empty.status_code == 400
    long_name = "a" * 300 + ".jpg"
    assert client.post(url, files={"file": ("a.jpg", b"x")}, data={"filename": long_name}).status_code == 400
    assert client.post("/api/gallery/999999/people", files={"file": ("a.jpg", b"x")},
                       data={"filename": "a.jpg"}).status_code == 404


def test_path_traversal_in_filename_is_neutralised(client):
    gid = client.post("/api/gallery", json={"name": "Team"}).json()["id"]
    res = client.post(f"/api/gallery/{gid}/people", files={"file": ("x.jpg", b"not an image")},
                      data={"filename": "../../etc/Mallory.jpg"})
    assert res.status_code == 200
    assert res.json()["file_name"] == "Mallory.jpg" and res.json()["status"] == "failed"


def test_unsupported_methods_are_refused(client):
    assert client.put("/api/gallery").status_code == 405
    assert client.patch("/api/gallery").status_code == 405


def test_rate_limit(client, monkeypatch):
    monkeypatch.setattr(config, "RATE_SESSION_PER_MIN", 3)
    # The fixture already used one of the three allowed requests.
    codes = [client.get("/api/session").status_code for _ in range(4)]
    assert codes == [200, 200, 429, 429]


def test_static_path_traversal(client):
    res = client.get("/..%2f..%2fbackend/app/security.py")
    assert "SESSION_TOKEN" not in res.text

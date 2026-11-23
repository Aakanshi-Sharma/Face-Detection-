"""Gateway security for the local API.

The server only listens on 127.0.0.1, but a web page open in the same browser
can still try to call it (cross-site requests, DNS rebinding). Every request
therefore passes these checks before it reaches an endpoint:

  1. Host header      must be 127.0.0.1 / localhost on our port (anti DNS-rebinding)
  2. Origin / Fetch   cross-site requests to /api are refused; there is no CORS
  3. Session cookie   HttpOnly, SameSite=Strict, random per launch
  4. CSRF token       custom header required on every state-changing request
  5. Rate limits      per endpoint class, in memory
  6. Size limits      request bodies are capped before they are read
  7. Response headers CSP, no framing, no sniffing, no caching of API data
"""
from __future__ import annotations

import hmac
import secrets
import threading
import time
from collections import deque

from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.types import ASGIApp

from . import config

SESSION_COOKIE = "faceid_session"
CSRF_HEADER = "x-csrf-token"

# New secrets on every launch: old cookies stop working when the app restarts.
SESSION_TOKEN = secrets.token_urlsafe(32)
CSRF_TOKEN = secrets.token_urlsafe(32)

# Endpoints reachable without a session (they reveal nothing about the data).
PUBLIC_API = {"/api/session", "/api/health"}

SECURITY_HEADERS = {
    "Content-Security-Policy": (
        "default-src 'self'; "
        "script-src 'self'; "
        "style-src 'self'; style-src-attr 'unsafe-inline'; "
        "img-src 'self' data: blob:; "
        "connect-src 'self'; "
        "font-src 'self'; "
        "object-src 'none'; base-uri 'none'; form-action 'self'; frame-ancestors 'none'"
    ),
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "Referrer-Policy": "no-referrer",
    "Cross-Origin-Opener-Policy": "same-origin",
    "Cross-Origin-Resource-Policy": "same-origin",
    "Permissions-Policy": "camera=(), microphone=(), geolocation=(), payment=(), usb=()",
}


def allowed_hosts() -> set[str]:
    hosts = {f"127.0.0.1:{config.PORT}", f"localhost:{config.PORT}"}
    if config.DEV_MODE:  # Vite dev server proxying to the API
        hosts |= {"127.0.0.1:5173", "localhost:5173"}
    return hosts


def allowed_origins() -> set[str]:
    return {f"http://{h}" for h in allowed_hosts()}


class RateLimiter:
    """Sliding-window limiter. Small and in memory: this is a one-user app."""

    def __init__(self):
        self._lock = threading.Lock()
        self._hits: dict[str, deque[float]] = {}

    def allow(self, key: str, limit: int, window: float = 60.0) -> bool:
        now = time.monotonic()
        with self._lock:
            hits = self._hits.setdefault(key, deque())
            while hits and now - hits[0] > window:
                hits.popleft()
            if len(hits) >= limit:
                return False
            hits.append(now)
            return True


limiter = RateLimiter()


def _rate_rule(method: str, path: str) -> tuple[str, int]:
    if path == "/api/identify":
        return "identify", config.RATE_IDENTIFY_PER_MIN
    if path == "/api/session":
        return "session", config.RATE_SESSION_PER_MIN
    if method == "POST" and path.endswith("/people"):
        return "enrol", config.RATE_ENROL_PER_MIN
    return "general", config.RATE_GENERAL_PER_MIN


def _deny(status: int, detail: str) -> JSONResponse:
    return JSONResponse({"detail": detail}, status_code=status)


def _same_site(request: Request) -> bool:
    """True unless the browser tells us the request came from another site."""
    fetch_site = request.headers.get("sec-fetch-site")
    if fetch_site and fetch_site not in ("same-origin", "none"):
        return False
    origin = request.headers.get("origin")
    if origin and origin not in allowed_origins():
        return False
    return True


def check_request(request: Request) -> JSONResponse | None:
    """Return an error response if the request must be refused, else None."""
    path, method = request.url.path, request.method

    if request.headers.get("host", "").lower() not in allowed_hosts():
        return _deny(400, "Unrecognised host.")

    if method not in ("GET", "HEAD", "POST", "DELETE"):
        return _deny(405, "Method not allowed.")

    length = request.headers.get("content-length")
    if length and (not length.isdigit() or int(length) > config.MAX_REQUEST_BYTES):
        return _deny(413, "That file is too large.")

    if not path.startswith("/api/"):
        return None  # static UI files

    if not _same_site(request):
        return _deny(403, "Cross-site requests are not allowed.")

    key, limit = _rate_rule(method, path)
    if not limiter.allow(key, limit):
        return _deny(429, "Too many requests. Wait a moment and try again.")

    if path in PUBLIC_API:
        return None

    cookie = request.cookies.get(SESSION_COOKIE, "")
    if not hmac.compare_digest(cookie, SESSION_TOKEN):
        return _deny(401, "Session expired. Reload the page.")

    if method not in ("GET", "HEAD"):
        token = request.headers.get(CSRF_HEADER, "")
        if not hmac.compare_digest(token, CSRF_TOKEN):
            return _deny(403, "Missing or invalid request token. Reload the page.")
    return None


def apply_headers(request: Request, response: Response) -> None:
    for name, value in SECURITY_HEADERS.items():
        response.headers.setdefault(name, value)
    if request.url.path.startswith("/api/"):
        # Faces and names must never sit in a shared or disk cache.
        response.headers["Cache-Control"] = "no-store"


def start_session(response: Response) -> dict:
    """Give the page its session cookie and CSRF token."""
    response.set_cookie(
        SESSION_COOKIE,
        SESSION_TOKEN,
        httponly=True,       # not readable by scripts
        samesite="strict",   # never sent on requests started by other sites
        path="/api",
        # No `secure` flag: the app is served over plain HTTP on loopback only.
    )
    return {"csrf_token": CSRF_TOKEN}


class SecurityMiddleware:
    """Pure ASGI middleware so refused requests never have their body read."""

    def __init__(self, app: ASGIApp):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        request = Request(scope)
        denied = check_request(request)
        if denied is not None:
            apply_headers(request, denied)
            await denied(scope, receive, send)
            return

        async def send_with_headers(message):
            if message["type"] == "http.response.start":
                headers = {k.decode("latin-1").lower() for k, _ in message.get("headers", [])}
                extra = dict(SECURITY_HEADERS)
                if request.url.path.startswith("/api/"):
                    extra["Cache-Control"] = "no-store"
                    message["headers"] = [
                        (k, v) for k, v in message.get("headers", []) if k.lower() != b"cache-control"
                    ]
                    headers.discard("cache-control")
                message.setdefault("headers", [])
                for name, value in extra.items():
                    if name.lower() not in headers:
                        message["headers"].append((name.encode("latin-1"), value.encode("latin-1")))
            await send(message)

        await self.app(scope, receive, send_with_headers)

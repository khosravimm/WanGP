"""Safe loopback-only CPU launcher for WanGP v19.

MVP integrates one *already-installed* CPU backend: OpenMOSS ONNX TTS.
No torch, CUDA, Gradio or MMGP imports and no model weight downloads.
"""
from __future__ import annotations

import argparse
import ipaddress
import json
import logging
import os
import re
import threading
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

from . import VERSION

MAX_REQUEST_BYTES = 21 * 1024 * 1024  # reference audio max 20 MiB (+ multipart metadata)
MAX_BACKEND_RESPONSE_BYTES = 48 * 1024 * 1024
DEFAULT_BACKEND = "http://127.0.0.1:18083"
INDEX_PATH = Path(__file__).with_name("index.html")
_LOG = logging.getLogger(__name__)


def check_loopback_host(host: str) -> bool:
    name = (host or "").strip().lower()
    if name == "localhost":
        return True
    try:
        return ipaddress.ip_address(name).is_loopback
    except ValueError:
        return False


def validate_backend_url(value: str) -> str:
    parsed = urlsplit(value)
    if parsed.scheme != "http" or not parsed.hostname or not check_loopback_host(parsed.hostname):
        raise ValueError("CPU audio backend must be a local HTTP loopback URL.")
    if parsed.username or parsed.password or parsed.path not in ("", "/") or parsed.query or parsed.fragment:
        raise ValueError("CPU audio backend URL must be origin-only with no credentials, path or query.")
    if parsed.port is None:
        raise ValueError("CPU audio backend URL must specify a port.")
    return value.rstrip("/")


def call_backend(backend: str, path: str, *, method: str = "GET",
                 data: bytes | None = None, content_type: str | None = None,
                 timeout: float = 120) -> tuple[int, bytes, str]:
    """Fixed-origin HTTP transport. Rejects backend redirects (avoid SSRF)."""
    if path not in ("/health", "/api/generate", "/model-catalog"):
        raise ValueError("Backend path not allowed.")
    headers = {}
    if content_type:
        headers["Content-Type"] = content_type
    request = urllib.request.Request(backend + path, data=data, headers=headers, method=method)

    class NoRedirect(urllib.request.HTTPRedirectHandler):
        def redirect_request(self, req, fp, code, msg, headers, newurl):
            return None

    opener = urllib.request.build_opener(NoRedirect)
    try:
        with opener.open(request, timeout=timeout) as response:
            payload = response.read(MAX_BACKEND_RESPONSE_BYTES + 1)
            if len(payload) > MAX_BACKEND_RESPONSE_BYTES:
                raise ValueError("Backend response exceeded the configured limit.")
            return response.status, payload, response.headers.get("Content-Type", "application/json")
    except urllib.error.HTTPError as exc:
        payload = exc.read(min(MAX_BACKEND_RESPONSE_BYTES, 8192))
        return exc.code, payload, exc.headers.get("Content-Type", "application/json")


class CpuHttpServer(ThreadingHTTPServer):
    allow_reuse_address = True
    daemon_threads = True

    def __init__(self, server_address, handler_class, backend: str):
        super().__init__(server_address, handler_class)
        self.backend = validate_backend_url(backend)
        self.generation_slot = threading.BoundedSemaphore(value=1)


class Handler(BaseHTTPRequestHandler):
    server: CpuHttpServer
    server_version = "WanGP-CPU/" + VERSION

    def log_message(self, fmt, *args):
        _LOG.info("%s - %s", self.address_string(), fmt % args)

    def _headers(self, status: int, length: int, content_type: str):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(length))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("X-Frame-Options", "DENY")
        self.send_header("Referrer-Policy", "no-referrer")
        if self.path == "/model-catalog":
            # This installed static catalog is an upstream-generated HTML page
            # containing its own inline JS/CSS. Scope the CSP exception only
            # to this one explicit route (other routes remain strict).
            policy = ("default-src 'none'; script-src 'unsafe-inline'; "
                      "style-src 'unsafe-inline'; img-src https: data:; "
                      "connect-src 'none'; object-src 'none'; frame-ancestors 'none'; "
                      "base-uri 'none'; form-action 'none'")
        else:
            policy = ("default-src 'self'; script-src 'self'; style-src 'self'; "
                      "connect-src 'self'; img-src 'self' data:; media-src 'self' blob:; "
                      "object-src 'none'; frame-ancestors 'none'; base-uri 'none'")
        self.send_header("Content-Security-Policy", policy)
        self.end_headers()

    def _send(self, status: int, payload: bytes, content_type: str = "application/json; charset=utf-8"):
        self._headers(status, len(payload), content_type)
        self.wfile.write(payload)

    def _json(self, status: int, value: dict):
        self._send(status, json.dumps(value, ensure_ascii=False).encode("utf-8"))

    def _safe_host(self) -> bool:
        host_header = self.headers.get("Host", "")
        try:
            parsed_host = urlsplit("http://" + host_header).hostname
        except ValueError:
            parsed_host = None
        # Protect an unauthenticated loopback app from DNS rebinding.
        if not parsed_host or not check_loopback_host(parsed_host):
            self._json(403, {"error": "Host header not permitted."})
            return False
        return True

    def do_GET(self):
        if not self._safe_host():
            return
        if self.path in ("/", "/index.html"):
            return self._send(200, INDEX_PATH.read_bytes(), "text/html; charset=utf-8")
        if self.path == "/styles.css":
            return self._send(200, INDEX_PATH.with_name("styles.css").read_bytes(), "text/css; charset=utf-8")
        if self.path == "/app.js":
            return self._send(200, INDEX_PATH.with_name("app.js").read_bytes(), "application/javascript; charset=utf-8")
        if self.path == "/health":
            return self._json(200, {
                "status": "ok", "version": VERSION, "runtime": "cpu-only",
                "gpu_required": False, "audio": "moss-nano-onnx",
                "video": "not-implemented", "backend": "loopback",
            })
        if self.path in ("/api/backend-health", "/model-catalog"):
            target = "/health" if self.path.endswith("backend-health") else "/model-catalog"
            try:
                status, payload, ctype = call_backend(
                    self.server.backend, target, timeout=5)
            except (OSError, ValueError, urllib.error.URLError) as exc:
                _LOG.warning("CPU backend GET unavailable: %s", type(exc).__name__)
                return self._json(503, {"error": "Local MOSS ONNX backend unavailable."})
            return self._send(status, payload, ctype)
        return self._json(404, {"error": "Unknown CPU-mode route."})

    def do_POST(self):
        if not self._safe_host():
            return
        if self.path != "/api/generate":
            return self._json(404, {"error": "Unknown CPU-mode route."})
        # Deny cross-origin browser submissions; loopback access is local-only.
        origin = self.headers.get("Origin", "")
        if origin:
            parsed_origin = urlsplit(origin)
            server_port = self.server.server_address[1]
            if not (parsed_origin.scheme == "http" and check_loopback_host(parsed_origin.hostname or "")
                    and parsed_origin.port == server_port):
                return self._json(403, {"error": "Cross-origin request denied."})
        mime = self.headers.get("Content-Type", "")
        if not re.match(r"^multipart/form-data;\s*boundary=", mime, flags=re.I):
            return self._json(415, {"error": "multipart/form-data is required."})
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            return self._json(400, {"error": "Invalid request length."})
        if not 0 < length <= MAX_REQUEST_BYTES:
            return self._json(413, {"error": "Audio request exceeds 21 MiB."})
        if not self.server.generation_slot.acquire(blocking=False):
            return self._json(429, {"error": "CPU is already generating audio; retry later."})
        try:
            data = self.rfile.read(length)
            if len(data) != length:
                return self._json(400, {"error": "Incomplete request body."})
            try:
                status, payload, ctype = call_backend(
                    self.server.backend, "/api/generate", method="POST",
                    data=data, content_type=mime, timeout=240)
            except (OSError, ValueError, urllib.error.URLError) as exc:
                _LOG.warning("CPU TTS unavailable: %s", type(exc).__name__)
                return self._json(503, {"error": "Local MOSS ONNX inference unavailable."})
            return self._send(status, payload, ctype)
        finally:
            self.server.generation_slot.release()


def create_server(host: str = "127.0.0.1", port: int = 18084,
                  backend: str = DEFAULT_BACKEND) -> CpuHttpServer:
    if not check_loopback_host(host):
        raise ValueError("CPU mode must bind to loopback only; use an authenticated proxy for external access.")
    if not 0 <= port <= 65535:
        raise ValueError("Invalid port.")
    return CpuHttpServer((host, port), Handler, backend)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="WanGP experimental CPU-only audio frontend (no CUDA/MMGP imports)")
    parser.add_argument("--cpu-only", action="store_true",
                        help="Enable this minimal CPU-only mode when launched from wgp.py.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18084)
    parser.add_argument("--moss-url", default=os.environ.get("WANGP_CPU_MOSS_URL", DEFAULT_BACKEND))
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    httpd = create_server(host=args.host, port=args.port, backend=args.moss_url)
    print(f"WanGP {VERSION} CPU-only audio UI at http://{args.host}:{httpd.server_port}/",
          flush=True)
    try:
        httpd.serve_forever(poll_interval=0.2)
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()


if __name__ == "__main__":
    main()

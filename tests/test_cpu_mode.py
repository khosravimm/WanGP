"""Stdlib-only CPU v19 regression suite; no torch or GPU required."""
from __future__ import annotations

import base64
import io
import json
import subprocess
import sys
import threading
import unittest
import urllib.error
import urllib.request
import wave
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from cpu_mode import VERSION
from cpu_mode.server import (
    Handler, MAX_REQUEST_BYTES, call_backend, create_server,
    validate_backend_url, check_loopback_host,
)


def make_wav():
    stream = io.BytesIO()
    with wave.open(stream, "wb") as out:
        out.setnchannels(1)
        out.setsampwidth(2)
        out.setframerate(48000)
        out.writeframes(b"\x7f\x00" * 4800)
    return stream.getvalue()


class FakeOnnxHandler(BaseHTTPRequestHandler):
    def log_message(self, *_args):
        pass

    def send_data(self, status, data, content_type="application/json"):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path == "/health":
            return self.send_data(200, b'{"status":"ok","device":"cpu"}')
        if self.path == "/model-catalog":
            return self.send_data(200, b"MOCK CATALOG", "text/html")
        if self.path == "/redirect":
            self.send_response(302)
            self.send_header("Location", "https://example.com/")
            self.end_headers()
            return
        return self.send_data(404, b"{}")

    def do_POST(self):
        if self.path != "/api/generate":
            return self.send_data(404, b"{}")
        size = int(self.headers.get("Content-Length", "0"))
        data = self.rfile.read(size)
        if b'name="text"' not in data:
            return self.send_data(422, b'{"error":"text missing"}')
        value = {
            "audio_base64": base64.b64encode(make_wav()).decode("ascii"),
            "sample_rate": 48000,
            "run_status": "mock CPU model",
        }
        return self.send_data(200, json.dumps(value).encode())


class HttpIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.backend = ThreadingHTTPServer(("127.0.0.1", 0), FakeOnnxHandler)
        cls.backend_thread = threading.Thread(target=cls.backend.serve_forever, daemon=True)
        cls.backend_thread.start()
        cls.front = create_server(
            "127.0.0.1", 0, "http://127.0.0.1:" + str(cls.backend.server_port)
        )
        cls.thread = threading.Thread(target=cls.front.serve_forever, daemon=True)
        cls.thread.start()
        cls.base = "http://127.0.0.1:" + str(cls.front.server_port)

    @classmethod
    def tearDownClass(cls):
        cls.front.shutdown()
        cls.front.server_close()
        cls.backend.shutdown()
        cls.backend.server_close()

    def request(self, path, body=None, headers=None):
        req = urllib.request.Request(self.base + path, data=body, headers=headers or {})
        try:
            with urllib.request.urlopen(req, timeout=10) as r:
                return r.status, r.read(), r.headers
        except urllib.error.HTTPError as err:
            try:
                return err.code, err.read(), err.headers
            finally:
                err.close()

    def test_health_gpu_free_contract(self):
        status, payload, _ = self.request("/health")
        data = json.loads(payload)
        self.assertEqual(status, 200)
        self.assertEqual(data["version"], VERSION)
        self.assertIs(data["gpu_required"], False)
        self.assertEqual(data["video"], "not-implemented")

    def test_ui_assets(self):
        for path, type_ in [("/", "text/html"), ("/app.js", "javascript"),
                            ("/styles.css", "text/css")]:
            code, data, headers = self.request(path)
            self.assertEqual(code, 200)
            self.assertIn(type_, headers["Content-Type"])
            self.assertTrue(len(data) > 100)
            self.assertEqual(headers["X-Content-Type-Options"], "nosniff")

    def test_audio_backend_health(self):
        status, data, _ = self.request("/api/backend-health")
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(data)["device"], "cpu")

    def test_catalog_proxy(self):
        status, data, headers = self.request("/model-catalog")
        self.assertEqual((status, data), (200, b"MOCK CATALOG"))
        # Catalog is a static previously-generated HTML file with inline JS.
        self.assertIn("script-src 'unsafe-inline'", headers["Content-Security-Policy"])

    def test_main_ui_strict_csp(self):
        status, _, headers = self.request("/")
        self.assertEqual(status, 200)
        self.assertIn("script-src 'self'", headers["Content-Security-Policy"])
        self.assertNotIn("unsafe-inline", headers["Content-Security-Policy"])

    def test_generate_real_http_contract(self):
        boundary = "WanGpCpuBoundary"
        body = (f"--{boundary}\r\nContent-Disposition: form-data; name=\"text\"\r\n\r\n"
                "CPU integration test\r\n"
                f"--{boundary}\r\nContent-Disposition: form-data; name=\"demo_id\"\r\n\r\n"
                "demo-10\r\n"
                f"--{boundary}--\r\n").encode()
        status, data, _ = self.request("/api/generate", body, {
            "Content-Type": "multipart/form-data; boundary=" + boundary,
        })
        self.assertEqual(status, 200)
        audio = base64.b64decode(json.loads(data)["audio_base64"])
        with wave.open(io.BytesIO(audio)) as w:
            self.assertEqual(w.getframerate(), 48000)
            self.assertEqual(w.getnframes(), 4800)

    def test_reject_invalid_content_type(self):
        status, _, _ = self.request("/api/generate", b"{}", {"Content-Type": "application/json"})
        self.assertEqual(status, 415)

    def test_reject_large_upload_without_reading(self):
        status, _, _ = self.request("/api/generate", b"one", {
            "Content-Type": "multipart/form-data; boundary=ABC",
            "Content-Length": str(MAX_REQUEST_BYTES + 1),
        })
        self.assertEqual(status, 413)

    def test_reject_unknown_route(self):
        self.assertEqual(self.request("/api/nonexistent")[0], 404)

    def test_refuse_dns_rebinding_host_header(self):
        req = urllib.request.Request(self.base + "/", headers={"Host": "evil.example"})
        with self.assertRaises(urllib.error.HTTPError) as cm:
            urllib.request.urlopen(req, timeout=10)
        self.assertEqual(cm.exception.code, 403)
        cm.exception.close()

    def test_refuse_cross_origin_post(self):
        self.assertEqual(self.request(
            "/api/generate", b"hello", {
                "Content-Type": "multipart/form-data; boundary=TEST",
                "Origin": "http://attacker.example",
            }
        )[0], 403)


class SecurityUnit(unittest.TestCase):
    def test_backend_url_only_loopback(self):
        self.assertEqual(validate_backend_url("http://127.0.0.1:18083"), "http://127.0.0.1:18083")
        for url in ("https://127.0.0.1:18083", "http://192.168.1.1:18083",
                    "http://foo.com:18083", "http://127.0.0.1:18083/admin",
                    "http://user:pass@localhost:18083", "http://127.0.0.1:18083/?x"):
            with self.subTest(url=url), self.assertRaises(ValueError):
                validate_backend_url(url)

    def test_server_refuses_network_listen(self):
        for addr in ("0.0.0.0", "192.168.1.20", "::"):
            with self.subTest(addr=addr), self.assertRaises(ValueError):
                create_server(addr, 0)

    def test_no_cuda_imported_through_cpu_server(self):
        code = ("import cpu_mode.server,sys;"
                "assert 'torch' not in sys.modules;"
                "assert 'mmgp' not in sys.modules;"
                "assert 'triton' not in sys.modules;"
                "print('NO_GPU_IMPORTS')")
        result = subprocess.run([sys.executable, "-c", code], cwd=Path(__file__).resolve().parents[1],
                                capture_output=True, text=True, timeout=10, check=True)
        self.assertIn("NO_GPU_IMPORTS", result.stdout)

    def test_cpu_cli_help_does_not_require_cuda(self):
        project = Path(__file__).resolve().parents[1]
        result = subprocess.run([sys.executable, "wgp.py", "--cpu-only", "--help"],
                                cwd=project, capture_output=True, text=True, timeout=10, check=True)
        self.assertIn("CPU-only", result.stdout)


if __name__ == "__main__":
    unittest.main()

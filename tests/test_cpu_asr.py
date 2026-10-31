"""Security and API regression for the optional whisper.cpp CPU ASR backend."""
import io
import json
import threading
import unittest
import urllib.error
import urllib.request
import wave
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from cpu_mode.asr import (AudioValidationError, form_for_whisper,
                          normalize_whisper_result, parse_transcription_multipart,
                          validate_pcm_wav)
from cpu_mode.server import create_server, check_loopback_host


def pcm_wav(rate=16000, channels=1, duration=0.4):
    samples = int(rate * duration)
    out = io.BytesIO()
    with wave.open(out, "wb") as wav:
        wav.setnchannels(channels)
        wav.setsampwidth(2)
        wav.setframerate(rate)
        wav.writeframes(b"\x01\x00" * samples * channels)
    return out.getvalue()


class MockWhisper(BaseHTTPRequestHandler):
    def log_message(self, *args): pass

    def do_GET(self):
        payload = b"Ready"
        self.send_response(200)
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_POST(self):
        if self.path != "/inference":
            self.send_error(404)
            return
        length = int(self.headers.get("Content-Length", 0))
        request = self.rfile.read(length)
        if b'audio.wav' not in request or b'verbose_json' not in request:
            self.send_error(400)
            return
        payload = json.dumps({
            "language": "persian",
            "detected_language": "persian",
            "text": "سلام جهان",
            "segments": [{"text": "سلام جهان", "start": 0.0, "end": 0.4}],
        }, ensure_ascii=False).encode("utf8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


class ASRUnitTests(unittest.TestCase):
    def test_valid_mono_16k(self):
        self.assertAlmostEqual(validate_pcm_wav(pcm_wav()), 0.4)

    def test_reject_wrong_rates_and_channels(self):
        for x in (pcm_wav(rate=44100), pcm_wav(channels=2), b"hello",
                  pcm_wav()[:-250], pcm_wav(duration=180.1)):
            with self.subTest(length=len(x)), self.assertRaises(AudioValidationError):
                validate_pcm_wav(x)

    def test_multipart_roundtrip(self):
        audio = pcm_wav()
        request, content_type = form_for_whisper(audio, "fa")
        # Producer includes more fields than the deliberately narrow frontend
        # parser. Build the same two-field request sent by the browser.
        boundary = "WanGPTestBoundary"
        raw = (f"--{boundary}\r\nContent-Disposition: form-data; name=\"language\"\r\n\r\n"
               "fa\r\n"
               f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"sample.wav\"\r\n"
               "Content-Type: audio/wav\r\n\r\n").encode() + audio + f"\r\n--{boundary}--\r\n".encode()
        got, language = parse_transcription_multipart(
            "multipart/form-data; boundary=" + boundary, raw)
        self.assertEqual((got, language), (audio, "fa"))
        self.assertIn(b"response_format", request)
        self.assertTrue(content_type.startswith("multipart/form-data"))

    def test_reject_unsupported_language(self):
        with self.assertRaises(AudioValidationError):
            parse_transcription_multipart("multipart/form-data; boundary=abc",
                                          b"--abc\r\nContent-Disposition: form-data; name=\"language\"\r\n\r\nxx\r\n--abc--\r\n")

    def test_normalize_real_whisper_schema(self):
        value = normalize_whisper_result(json.dumps({
            "text": "سلام جهان", "segments": [{"start": 0.0, "end": 1.0, "text": "سلام جهان"}],
            "detected_language": "persian", "language": "persian",
        }, ensure_ascii=False).encode(), "fa", 1.2)
        self.assertEqual(value["text"], "سلام جهان")
        self.assertEqual(len(value["segments"]), 1)
        self.assertEqual(value["language_detected"], "persian")
        self.assertIs(value["speaker_diarization"], False)


class ASRHttpTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.backend = ThreadingHTTPServer(("127.0.0.1", 0), MockWhisper)
        cls.worker = threading.Thread(target=cls.backend.serve_forever, daemon=True)
        cls.worker.start()
        base = "http://127.0.0.1:" + str(cls.backend.server_port)
        cls.front = create_server(port=0, backend=base, asr_backend=base)
        cls.front_thread = threading.Thread(target=cls.front.serve_forever, daemon=True)
        cls.front_thread.start()
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
            with urllib.request.urlopen(req, timeout=6) as resp:
                return resp.status, json.loads(resp.read())
        except urllib.error.HTTPError as exc:
            try:
                return exc.code, json.loads(exc.read())
            finally:
                exc.close()

    def post_audio(self, audio, language="fa", extra_headers=None):
        boundary = "WhisperFormTest"
        body = (f"--{boundary}\r\nContent-Disposition: form-data; name=\"language\"\r\n\r\n"
                f"{language}\r\n--{boundary}\r\n"
                "Content-Disposition: form-data; name=\"file\"; filename=\"valid.wav\"\r\n"
                "Content-Type: audio/wav\r\n\r\n").encode() + audio + f"\r\n--{boundary}--\r\n".encode()
        headers = {"Content-Type": f"multipart/form-data; boundary={boundary}"}
        headers.update(extra_headers or {})
        return self.request("/api/transcribe", body, headers)

    def test_asr_health(self):
        code, body = self.request("/api/asr-health")
        self.assertEqual(code, 200)
        self.assertEqual(body["engine"], "whisper.cpp-cpu")
        self.assertFalse(body["speaker_diarization"])

    def test_asr_transcribe_mock(self):
        code, result = self.post_audio(pcm_wav())
        self.assertEqual(code, 200)
        self.assertEqual(result["text"], "سلام جهان")
        self.assertEqual(len(result["segments"]), 1)

    def test_asr_invalid_wav(self):
        code, body = self.post_audio(b"<html>invalid</html>")
        self.assertEqual(code, 422)
        self.assertIn("error", body)

    def test_asr_reject_cross_origin(self):
        code, _ = self.post_audio(pcm_wav(), extra_headers={"Origin": "https://evil.example"})
        self.assertEqual(code, 403)

    def test_asr_reject_invalid_language(self):
        code, _ = self.post_audio(pcm_wav(), language="nonsense")
        self.assertEqual(code, 422)

    def test_asr_refuse_network_server(self):
        with self.assertRaises(ValueError):
            create_server("0.0.0.0", 0, asr_backend="http://127.0.0.1:18085")


if __name__ == "__main__":
    unittest.main()

"""Pure-Python HTTP boundary for a separately managed whisper.cpp CPU server.

Accepts 16 kHz, mono, 16-bit PCM WAV (preprocessed client-side); does not
implement speaker diarization. Uploaded bytes are held in memory only.
"""
from __future__ import annotations

import io
import json
import re
import wave
from email.parser import BytesParser
from email.policy import default

SUPPORTED_LANGUAGES = frozenset(("auto", "fa", "en", "ar", "de", "fr", "tr", "ru", "zh", "es", "it", "ja"))
MAX_AUDIO_SECONDS = 180
MAX_WAV_BYTES = 20 * 1024 * 1024


class AudioValidationError(ValueError):
    pass


def parse_transcription_multipart(content_type: str, data: bytes) -> tuple[bytes, str]:
    """Bounded multipart extraction, ignoring caller-supplied file names."""
    match = re.search(r"boundary=(?:\"([^\"]+)\"|([^;\s]+))", content_type, re.I)
    if not match:
        raise AudioValidationError("Invalid multipart boundary.")
    boundary = match.group(1) or match.group(2)
    if not 1 <= len(boundary) <= 70 or not boundary.isascii():
        raise AudioValidationError("Invalid multipart boundary.")
    envelope = ("MIME-Version: 1.0\r\nContent-Type: multipart/form-data; boundary="
                + boundary + "\r\n\r\n").encode()
    message = BytesParser(policy=default).parsebytes(envelope + data)
    if not message.is_multipart():
        raise AudioValidationError("Malformed multipart request.")
    audio = None
    language = "auto"
    parts = message.iter_parts()
    count = 0
    for part in parts:
        count += 1
        if count > 5:
            raise AudioValidationError("Too many form fields.")
        field = part.get_param("name", header="content-disposition")
        payload = part.get_payload(decode=True)
        if payload is None:
            raise AudioValidationError("Invalid form field.")
        if field == "file":
            if audio is not None or len(payload) > MAX_WAV_BYTES:
                raise AudioValidationError("Invalid audio size or duplicate upload.")
            audio = payload
        elif field == "language":
            if len(payload) > 20:
                raise AudioValidationError("Invalid language.")
            language = payload.decode("ascii", errors="strict").lower()
        else:
            raise AudioValidationError("Unexpected form field.")
    if audio is None:
        raise AudioValidationError("Missing WAV audio file.")
    if language not in SUPPORTED_LANGUAGES:
        raise AudioValidationError("Unsupported transcription language.")
    validate_pcm_wav(audio)
    return audio, language


def validate_pcm_wav(audio: bytes) -> float:
    if not 44 <= len(audio) <= MAX_WAV_BYTES:
        raise AudioValidationError("Invalid WAV size.")
    if not (audio.startswith(b"RIFF") and audio[8:12] == b"WAVE"):
        raise AudioValidationError("Audio must be a PCM WAV file.")
    try:
        with wave.open(io.BytesIO(audio), "rb") as wav:
            if (wav.getcomptype() != "NONE" or wav.getnchannels() != 1 or
                    wav.getsampwidth() != 2 or wav.getframerate() != 16000):
                raise AudioValidationError("Audio must be mono 16 kHz 16-bit PCM WAV.")
            seconds = wav.getnframes() / wav.getframerate()
            if not 0 < seconds <= MAX_AUDIO_SECONDS:
                raise AudioValidationError("WAV duration must be between 0 and 180 seconds.")
            # Fail on a forged length declaring samples that aren't in the request.
            if len(wav.readframes(wav.getnframes())) != wav.getnframes() * 2:
                raise AudioValidationError("Incomplete WAV samples.")
            return seconds
    except (EOFError, wave.Error, OSError) as exc:
        raise AudioValidationError("Invalid or corrupted WAV file.") from exc


def form_for_whisper(audio: bytes, language: str) -> tuple[bytes, str]:
    boundary = "WanGPCpuASROnlyPrevalidatedWAV"
    body = bytearray()
    for name, value in (("language", language), ("response_format", "verbose_json"),
                        ("temperature", "0.0"), ("translate", "false")):
        body.extend(f"--{boundary}\r\nContent-Disposition: form-data; name=\"{name}\"\r\n\r\n{value}\r\n".encode())
    body.extend(f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"audio.wav\"\r\nContent-Type: audio/wav\r\n\r\n".encode())
    body.extend(audio)
    body.extend(f"\r\n--{boundary}--\r\n".encode())
    return bytes(body), f"multipart/form-data; boundary={boundary}"


def normalize_whisper_result(payload: bytes, language: str, seconds: float) -> dict:
    try:
        data = json.loads(payload)
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise AudioValidationError("Invalid response from whisper.cpp.") from exc
    if not isinstance(data, dict):
        raise AudioValidationError("Unexpected response format.")
    # whisper.cpp 1.9.x verbose_json has {text,segments,detected_language}.
    # Older whisper.cpp builds may use a transcription array.
    transcription = data.get("segments")
    if transcription is None:
        transcription = data.get("transcription") or []
    if isinstance(transcription, list):
        lines = [seg.get("text", "") for seg in transcription if isinstance(seg, dict)]
        text = data.get("text") or " ".join(lines)
        segments = transcription
    else:
        text = data.get("text", "")
        segments = []
    if not isinstance(text, str):
        raise AudioValidationError("Invalid transcript.")
    return {
        "text": text.strip(),
        "language_requested": language,
        "language_detected": data.get("detected_language") or data.get("language"),
        "language_used": data.get("language"),
        "duration_seconds": round(seconds, 3),
        "segments": segments,
        "engine": "whisper.cpp-cpu",
        "speaker_diarization": False,
    }

# WanGP CPU Edition changelog

## 19.0-cpu.2 — experimental CPU ASR
- Add independent whisper.cpp CPU transcription backend and in-browser
  audio transcoding to mono PCM16 WAV at 16 kHz.
- Add file and language inputs, transcript preview and TXT export.
- Add /api/asr-health, /api/transcribe with WAV and multipart validation,
  loopback-only fixed backend URL and request size/duration limits.
- Explicitly do not claim speaker diarization or guaranteed Persian accuracy.
- Add separate Whisper service template, automated tests and E2 test report.

## 19.0-cpu.1 — experimental (2026-10-31 UTC)
- Base: upstream WanGP 19.0 / ee207709.
- Add isolated CLI flag `--cpu-only` before CUDA/Torch imports.
- Add dependency-free Python HTTP UI and ONNX TTS local transport.
- Enable text-to-speech and optional reference-voice cloning.
- Restrict service to loopback with upload/response caps, basic origin/host
  validation and single-job backpressure.
- Preserve upstream GPU mode unchanged.
- Include optional MOSS-TTS-Nano Gradio app plugin.
- Add CPU-only CI and Python tests, live Sal-AI E2 evidence.
- Explicitly defer video pipeline, main GPU gallery and MMGP.

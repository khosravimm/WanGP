# WanGP CPU Edition changelog

## 19.0-cpu.3 — actual WanGP v19 original interface (experimental)
- Add opt-in `--native-ui-cpu`, loading the genuine upstream Gradio
  application rather than rendering a separate frontend.
- Safely skip import-time GPU capability queries and restrict CUDA-only
  model handler imports in this opt-in mode.
- Fail closed when original GPU model generation is attempted; expose
  MOSS Nano CPU via WanGP's genuine plugin tab.
- Correct unused plugin `api_session` callback signature and return
  a private Gradio-compatible WAV copy while keeping the archive.
- Add independent loopback-only native UI service on port 18088.
- Add native UI smoke tests and deployment documentation.

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

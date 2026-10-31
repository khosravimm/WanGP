# WanGP Community CPU Edition — v19.0-cpu.1

**Status: experimental audio MVP (E2 on Sal-AI). This does not imply video
generation or MMGP/Deepy GPU workloads run on CPU.**

This edition is maintained on the fork
[github.com/khosravimm/WanGP](https://github.com/khosravimm/WanGP).
Upstream base: WanGP 19.0, commit `ee2077090e57da6498123a9106f6321ebfc7576e`.
License: WanGP Community License 2.0; see `LICENSE.txt`.

## Architecture and objectives

The normal GPU workflow stays unchanged. Invoking `python wgp.py --cpu-only`
dispatches **before** importing `torch`, `mmgp`, `triton`, `gradio` or
the main GPU pipeline. It launches a minimal, standard-library HTTP server:

```
Browser :18084 (WanGP CPU UI and HTTP API)
    |
    | fixed local-origin HTTP, one generation at a time
    v
127.0.0.1:18083 (OpenMOSS-TTS-Nano ONNX CPU service)
    |
    v
ONNXRuntime / local 100M model + audio tokenizer -> 48kHz WAV
```

The audio engine is the separately-installed, independently licensed
`OpenMOSS-Team/MOSS-TTS-Nano-100M-ONNX` and
`OpenMOSS-Team/MOSS-Audio-Tokenizer-Nano-ONNX`, not the GPU-oriented
WanGP MOSS-TTS v1.5 handler. The user must install and test that CPU
engine separately; these model weights are **not** bundled in the repo.

A second, optional Gradio plugin exists at `plugins/moss_tts_cpu`.
It uses the same separate ONNX venv and does not by itself fix the
GPU-first main application.

## Start

Requires Python **3.11+** and **no additional pip packages** for this CPU
HTTP frontend. Requires an already-running compatible MOSS ONNX
backend on local 127.0.0.1:18083.

```bash
cd /opt/Progs/WanGP-CPU-v19
python3 wgp.py --cpu-only --host 127.0.0.1 --port 18084 \
    --moss-url http://127.0.0.1:18083
```

On the **same host**, open `http://127.0.0.1:18084/`.
The app exposes `/health`, `/api/backend-health`, `/model-catalog`
and `/api/generate`. The latter accepts multipart form data following
the upstream MOSS ONNX `POST /api/generate` contract; it returns
base64-encoded WAV JSON. Selecting a built-in voice uses upstream
`demo_id`; voice cloning uploads a reference audio file.

To access remotely, use authenticated HTTPS reverse proxy or a
user-authorized SSH local port forward. The app **refuses non-loopback
binding**. Never open the raw 18084/18083 ports on a public interface:
this experimental mode has no built-in user authentication.

## Operational limits / security gates

- No GPU required to launch frontend; no inference without ONNX backend.
- 21 MiB maximum upload request, upstream 20 MiB audio file cap.
- 48 MiB response cap; fixed loopback-only backend origin and path allowlist.
- No arbitrary URL proxying, external HTTP redirect following, nor shell invocation.
- Content Security Policy (CSP), Host-header and Origin checks.
- At most one simultaneous synthesis, subsequent requests receive HTTP 429.
- No model weights downloaded automatically.
- The backend owns its generated audio files; the frontend sends audio as
  JSON in memory and does not persist voice uploads.
- The UI exposes a catalog for offline search of OpenMOSS repos if installed.
- CPU inference performance and voice quality vary; no MOS/ASR validation.
- **Not available:** text-to-video, image-to-video, automatic video synthesis,
  GPU models, main WanGP Gallery, full MMGP-compatible workflow,
  Deepy Prime/Showrunner.

## Real Sal-AI proof (E2)

Environment: 192 GiB RAM, no physical NVIDIA GPU; original WanGP at
`/opt/Progs/Wan2GP` remains v18.84 and untouched. Backend:
`moss-tts-nano-cpu.service` (systemd user), `127.0.0.1:18083`.

Observed first end-to-end test:

- Local app started through `python3 wgp.py --cpu-only --port 18084`.
- `/health`, `/api/backend-health`, `/model-catalog`, CSS, JavaScript
  and UI each returned HTTP 200.
- `POST /api/generate`, Persian, `demo_id=demo-21`, HTTP 200;
  WAV **48 kHz stereo**, duration **6.8 seconds**, wall-clock
  processing **7.84 seconds**. WAV header and nonzero frames verified.
- Output path on Sal-AI:
  `/opt/models/MOSS-TTS-Nano-Models/outputs/wangp_v19_cpu_e2_persian.wav`.
- **Human evaluation** of pronunciation, timbre and similarity was not done.

## Testing

```bash
python3 -m unittest discover -s tests -p test_cpu_mode.py -v
python3 wgp.py --cpu-only --help
```

CI uses Python 3.11 and 3.12 on Ubuntu without installing PyTorch.
The synthetic HTTP TTS mock only validates contracts (E1).
The actual Sal-AI test validates local integration and audio file
structure (E2), not subjective speech quality.

## Rollback and upstream rebase

- Stop `wgp-cpu.service` (if installed) or end the `--cpu-only` process.
- Original `python wgp.py` GPU mode is unaffected by the early-dispatch
  branch and can be restored from the upstream tag/commit as needed.
- The independent MOSS audio service can continue running even when
  the WanGP CPU UI is stopped.
- Keep `main` tracking upstream; propose changes via
  `feature/cpu-only-v19` and code review before merge.
- Changes affecting backend API/validation must include tests and review
  of input validation, privilege boundaries and proxy exposure.

## Future (not yet implemented)

1. Speech quality evaluation for Persian and different reference voices.
2. Standalone transcription/diarization plugin using a CPU-validated model.
3. Queue, task cancellation and gallery using the WanGP API contracts.
4. Model discovery and selection with explicit per-backend CPU capability.
5. Video generation only after a reproducible CPU model, memory, latency
   and quality measurement demonstrates feasibility.

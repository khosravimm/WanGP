# WanGP Community CPU Edition — v19.0-cpu.2

**Status: experimental speech/audio MVP (E2 on Sal-AI). This does not imply video
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
    |                         |
    | TTS                     | ASR, 16kHz mono PCM WAV
    v                         v
127.0.0.1:18083              127.0.0.1:18085
MOSS-TTS-Nano ONNX CPU        whisper.cpp CPU
    |                         |
48kHz stereo WAV             transcript JSON, timestamps
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
and `/api/generate`, `/api/asr-health`, `/api/transcribe`.
The generation endpoint accepts multipart form data following
the upstream MOSS ONNX `POST /api/generate` contract; it returns
base64-encoded WAV JSON. Selecting a built-in voice uses upstream
`demo_id`; voice cloning uploads a reference audio file.

To access remotely, use authenticated HTTPS reverse proxy or a
user-authorized SSH local port forward. The app **refuses non-loopback
binding**. Never open the raw 18084/18083 ports on a public interface:
this experimental mode has no built-in user authentication.

## Operational limits / security gates

- No GPU required to launch frontend; TTS inference needs ONNX backend;
  ASR inference independently needs whisper.cpp backend.
- ASR browser decodes media locally and resamples to 16kHz mono WAV via
  Web Audio API; this browser-side conversion has not been GUI end-to-end
  automated; HTTP ASR route is E2-verified.
- ASR upload must be PCM 16-bit mono 16kHz WAV, <=180 seconds, <=20MiB;
  allowed language codes are explicitly enumerated.
- Optional whisper.cpp server requires a separately obtained multilingual
  GGML model; never use an English-only `.en` checkpoint for Persian.
- **No speaker diarization** by the base Whisper ASR model; do not infer
  speaker labels from segments or word timestamps.
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
- **Not available:** speaker diarization, text-to-video, image-to-video, automatic video synthesis,
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

## ASR addition (v19.0-cpu.2)

**Selected for immediate CPU deployment:** ggml-org/whisper.cpp, via a
preinstalled native CPU binary, with multilingual Whisper Base model for
the initial proof. This avoids another Python/Torch runtime. An optional
Small multilingual checkpoint may improve accuracy at higher resource
cost but must be evaluated on labeled Persian speech, not assumed better.

The reference GGML models are published at
https://huggingface.co/ggerganov/whisper.cpp .
On Sal-AI, the Base binary is at
`/opt/models/WanGP-CPU-ASR/weights/ggml-base.bin` (SHA-1
`465707469ff3a37a2b9b8d8f89f2f99de7299dac`).

Existing prebuilt native executable (owned by a different application,
**not copied or modified**):
`/opt/Progs/Unsloth/whisper.cpp/build/bin/whisper-server`.

Run standalone without GPU:

```bash
whisper-server --no-gpu   --model /opt/models/WanGP-CPU-ASR/weights/ggml-base.bin   --host 127.0.0.1 --port 18085 --threads 8 --language auto
```

Replace `whisper-server` above with its actual executable path on your
host. Example systemd service: `cpu_mode/service/wangp-cpu-whisper-asr.service.example`.
Backend is loopback-only and is **not authenticated**; do not expose it
or the CPU web frontend to arbitrary LAN users. The native server also
has its own model loading endpoint: treat it as a trusted local tool,
never a multi-tenant public service.

The browser accepts audio files supported by Web Audio API up to
20 MiB/3 minutes. It locally decodes and resamples to 16kHz mono PCM16,
then sends the preprocessed WAV to the trusted WanGP CPU frontend.
The frontend independently verifies the WAV header, format, bytes,
duration and multipart content before forwarding on a fixed local
`/inference` route to whisper.cpp.

**Initial E2 quality caveat:** On a 6.8-second synthetic Persian voice
sample from MOSS Nano, Whisper Base returned only `سالوم` for a much
longer target sentence. Thus the pipeline runs but Persian accuracy
on this sample is poor. Do not claim usable Persian WER until a
human-labeled multi-speaker evaluation exists. No diarization is
implemented; Whisper's segments are timestamp chunks, not speaker IDs.

### ASR testing

```bash
python -m unittest discover -s tests -p 'test_cpu_*.py' -v
curl -f http://127.0.0.1:18084/api/asr-health
curl -f http://127.0.0.1:18084/api/transcribe   -F 'language=fa' -F 'file=@/path/mono-16kHz-PCM16.wav'
```

`docs/CPU_ONLY.md` describes the intentionally limited feature scope.
Only the CPU mode is supported without GPU; full native WanGP remains
GPU-first. Licenses for WanGP, whisper.cpp and each model are independent.


### Verified ASR integration (Sal-AI; E2; 2026-10-31 UTC)

- Whisper Base multilingual 142 MiB, SHA-1 verified against official table.
- Native `whisper-server --no-gpu` running via user service
  `wangp-cpu-whisper-asr.service` on 127.0.0.1:18085.
- `POST http://127.0.0.1:18084/api/transcribe` with 16 kHz mono PCM
  and `language=fa`: HTTP **200**, wall time **0.883s** for a
  **6.8s** generated Persian voice sample; one timestamp segment.
- Result text was only `سالوم`, while the original text was longer.
  This is **a failed informal accuracy check**, not proof of Persian
  speech-to-text suitability. The detected language was English even
  with requested Persian decoding; the API reports both separately.
- Malformed audio is rejected by WanGP endpoint: HTTP **422**.
- The original speech generation API still produced a 48kHz stereo WAV
  through the same frontend: HTTP **200**, **4.88s** audio and **5.25s**
  request time.
- 26 Python/HTTP tests passed locally before this documentation update.
- The *browser AudioContext conversion path* has static JS validation,
  but still needs GUI functional testing on an actual browser.
- Real speaker diarization remains **not implemented**.

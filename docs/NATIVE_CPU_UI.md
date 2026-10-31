# Original WanGP Gradio UI on CPU-only hardware (v19.0-cpu.3 experimental)

## What this is

An **opt-in launch mode for the actual upstream WanGP v19 UI**.
It is not the separate small HTTP frontend shipped as `--cpu-only`
in v19.0-cpu.2. The original Gradio UI, app bar, gallery layout, model
configuration surfaces and plugin system are loaded from unmodified
upstream UI modules.

**Status: E2 locally on Sal-AI for UI launch, real original Gradio
plugin callback to ONNX speech inference, returned WAV file.**
Full GPU-oriented generation and model loading are **not CPU-capable**.

Source branch: `feature/native-cpu-ui-v19` in the
[khosravimm/WanGP](https://github.com/khosravimm/WanGP) fork.

## Start (Sal-AI)

This mode reuses the **already installed WanGP Python 3.11/Gradio/Torch
environment** in `/opt/Progs/Wan2GP/env_uv`. It does not install
new packages or modify that environment.

```bash
cd /opt/Progs/WanGP-Native-CPU
/opt/Progs/Wan2GP/env_uv/bin/python -u wgp.py \
    --native-ui-cpu --offline \
    --server-name 127.0.0.1 --server-port 18088 \
    --config /opt/models/WanGP-Native-CPU/config
```

Open **http://127.0.0.1:18088/** from the same machine.
The `moss_tts_cpu` plugin must be enabled in the isolated config file
and appears in WanGP's original app bar as **MOSS TTS CPU**. Its
`Generate speech on CPU` button uses the isolated MOSS-TTS-Nano ONNX
inference executable, rather than native GPU model handlers.

In the isolated config, select `moss_tts_v1_5` as the **UI placeholder
only**, not as a runnable model. Its native generation button returns
an explicit unsupported-GPU error. **Actual CPU synthesis runs from the
MOSS TTS CPU app.**

## Implementation, boundaries and limitations

1. `--native-ui-cpu` is removed from the argument list before the
   upstream option parser is invoked; sets `WANGP_NATIVE_UI_CPU=1`.
2. `shared/attention.py` and early `wgp.py` GPU-capability checks
   are guarded only in this explicit launch mode, leaving the original
   GPU path unchanged.
3. Importing all GPU model handlers causes CUDA initialization during
   import even without running a model. This CPU mode maps only the
   upstream `moss_tts_v1_5` handler to build its native audio form.
4. The native generator is **fail-closed**: attempting standard GPU
   inference displays a Gradio error, before attempting model loading.
5. The original WanGP plugin callback contract does not require
   `api_session`; removing an unused optional `api_session`
   parameter prevents accidentally invoking the WebUI-session wrapper.
6. WanGP/Gradio only accepts generated files from approved paths;
   the CPU audio plugin makes a secure mode-0600 temporary copy in
   the system temp directory for the native Gradio audio component,
   retaining the original WAV in the local archive.
7. The original UI still displays GPU-oriented configuration controls,
   gallery sections, Showrunner/Media Flow/Deepy modules and other
   surfaces that are **not validated on CPU**. Their presence is not
   evidence of functional GPU model inference.

## Real local evidence

- Genuine WanGP 19 UI: GET `/` HTTP 200 (~11.2 MB rendered HTML),
  GET `/config` HTTP 200 (~4.4 MB), GET `/gradio_api/info` HTTP 200.
- `/config` includes ~3,147 original Gradio components and 115 tab
  definitions; the `MOSS TTS CPU` tab is discovered and enabled.
- Real POST `/gradio_api/call/generate_speech` on native UI followed
  by event polling returns `event: complete` and a Gradio-served
  playable WAV file path after **12.52 s** CPU inference. This is
  original Gradio callback end-to-end, not the custom mini UI.
- Local-only port 18088; the existing MOSS ONNX service remains on
  18083; existing compact UI on 18084 and ASR on 18085 remain
  independent and unmodified.
- **Not yet human-reviewed:** GUI visual interaction on real browser,
  audio quality, gallery workflow, native audio generation placement.

## Security and deployment

- This mode has the original WanGP app's capabilities, including
  management interfaces; **never expose it to an untrusted network**.
- Listen on `127.0.0.1` only. No authentication is enabled by default.
- If remote access is needed, use an authenticated TLS reverse proxy or
  an SSH local forwarding channel restricted to an authorized operator.
- `--offline` and `HF_HUB_OFFLINE=1` block model downloads. Additional
  upstream tools may have other network functions; review separately.
- This uses an upstream-copyright UI under WanGP Community License 2.0.
  Keep vendor notices and any third-party model licenses.
- No changes to the original installed
  `/opt/Progs/Wan2GP` working tree (v18.84) or its tracked files.

## Unit/CI and rollback

```bash
python -m unittest discover -s tests -p 'test_cpu_*.py' -v
```

Additional static tests assert that CPU opt-in, fail-closed
GPU generation and plugin callback signatures are present.
E2 requires a running server and separately installed MOSS Nano backend.

Stop only `wangp-native-ui-cpu.service` to roll back.
The original WanGP and the independent 19.0-cpu.2 compact UI stay
unchanged. The feature branch must be reviewed before merging.

# MOSS TTS CPU plugin (experimental)

Adds a Gradio tab to WanGP, calling the separately-installed, PyTorch-free
`/opt/Progs/MOSS-TTS-Nano-CPU` via subprocess. It does not require GPU resources.
It does **not** repair WanGP's GPU-dependent startup path; production use needs a
working WanGP host or its independent ONNX UI. This local plugin is experimental.

Security: local fixed command, no shell expansion, 1200-char text limit,
20 MiB reference input limit, 240-second job timeout, single inference lock,
offline weights, generated WAV validation, no external listener.

Standalone callback smoke test:
`env_uv/bin/python -c 'from plugins.moss_tts_cpu.plugin import generate_speech; print(generate_speech("Hello from plugin CPU.", "Ava", None, 60))'`

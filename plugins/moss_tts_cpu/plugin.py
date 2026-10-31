"""Standalone CPU speech panel for WanGP, without changes to the core engine.

The inference dependency runs in a separate venv, never inside WanGP's CUDA env.
The plugin may be enabled once WanGP itself can launch; it does NOT bypass
WanGP/MMGP's other GPU requirements.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from threading import Lock

from shared.utils.plugins import WAN2GPPlugin

PROJECT = Path("/opt/Progs/MOSS-TTS-Nano-CPU")
PYTHON = PROJECT / ".venv/bin/python"
BACKEND = PROJECT / "infer_onnx.py"
MODELS = PROJECT / "models"
OUTPUTS = Path("/opt/models/MOSS-TTS-Nano-Models/outputs")
_INFERENCE_LOCK = Lock()
_SUPPORTED_VOICES = ("Ava", "Bella", "Adam", "Nathan", "Junhao", "Trump")


def generate_speech(
    text: str,
    voice: str = "Ava",
    reference_audio: str | None = None,
    max_new_frames: int = 120,
) -> str:
    """Return a verified WAV path, suitable for a gr.Audio component."""
    text = (text or "").strip()
    if not text or len(text) > 1200:
        raise ValueError("Text must have 1–1200 characters.")
    if voice not in _SUPPORTED_VOICES:
        raise ValueError("Unknown built-in voice.")
    frames = int(max_new_frames)
    if not 15 <= frames <= 250:
        raise ValueError("Maximum frames must be 15–250.")
    reference = Path(reference_audio).resolve() if reference_audio else None
    if reference and (not reference.is_file() or reference.stat().st_size > 20 * 1024 * 1024):
        raise ValueError("Reference audio is missing or exceeds the 20 MiB limit.")
    for required in (PYTHON, BACKEND, MODELS / "MOSS-TTS-Nano-100M-ONNX/moss_tts_global_shared.data"):
        if not required.exists():
            raise RuntimeError(f"CPU backend missing: {required}")
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=OUTPUTS, suffix=".wav", prefix="moss_wangp_", delete=False
    ) as file:
        path = Path(file.name)
    command = [
        str(PYTHON), str(BACKEND),
        "--model-dir", str(MODELS),
        "--execution-provider", "cpu",
        "--cpu-threads", "8",
        "--text", text,
        "--max-new-frames", str(frames),
        "--sample-mode", "greedy",
        "--do-sample", "0",
        "--disable-wetext-processing",
        "--output-audio-path", str(path),
    ]
    if reference:
        command.extend(["--prompt-audio-path", str(reference)])
    else:
        command.extend(["--voice", voice])
    env = os.environ.copy()
    env.update(
        HF_HUB_OFFLINE="1",
        OPENBLAS_NUM_THREADS="8",
        OMP_NUM_THREADS="8",
        CUDA_VISIBLE_DEVICES="",
    )
    try:
        with _INFERENCE_LOCK:  # avoid concurrent multi-GB model loads
            subprocess.run(
                command, cwd=str(PROJECT), env=env,
                check=True, capture_output=True, text=True, timeout=240,
            )
        import wave
        with wave.open(str(path), "rb") as wav:
            if wav.getnframes() < 1200 or wav.getframerate() != 48000:
                raise RuntimeError("WAV output validation failed.")
        # Gradio only accepts generated paths from the app directory or /tmp.
        # Preserve the original archival WAV in the configured output folder
        # while staging a private 0600 copy for the browser to play.
        with tempfile.NamedTemporaryFile(
            dir=tempfile.gettempdir(), prefix="wangp_moss_audio_", suffix=".wav",
            delete=False
        ) as staged:
            browser_path = Path(staged.name)
            try:
                with path.open("rb") as original:
                    shutil.copyfileobj(original, staged)
            except BaseException:
                browser_path.unlink(missing_ok=True)
                raise
        return str(browser_path)
    except (subprocess.SubprocessError, OSError, RuntimeError):
        path.unlink(missing_ok=True)
        raise


class MossTTSCpuPlugin(WAN2GPPlugin):
    def __init__(self):
        super().__init__()
        self.name = "MOSS TTS (CPU)"
        self.version = "0.1.0"
        self.description = "OpenMOSS ONNX speech synthesis and voice cloning on CPU."

    def setup_ui(self):
        self.add_tab(
            tab_id="moss_tts_cpu",
            label="MOSS TTS CPU",
            component_constructor=self.create_ui,
            position=3,
        )

    def create_ui(self):
        import gradio as gr

        with gr.Column():
            gr.Markdown("### MOSS-TTS Nano — CPU Only\nRuns in a separate ONNX runtime with no GPU dependency.")
            prompt = gr.Textbox(label="Speech text / متن گفتار", lines=5, max_lines=12, value="سلام، به آزمایش تولید گفتار فارسی خوش آمدید.")
            voice = gr.Dropdown(choices=list(_SUPPORTED_VOICES), value="Ava", label="Preset voice (ignored when reference audio is provided)")
            reference = gr.Audio(label="Optional voice reference / صدای مرجع", type="filepath", sources=["upload"])
            frames = gr.Slider(minimum=15, maximum=250, step=5, value=120, label="Maximum generated frames")
            generate = gr.Button("Generate speech on CPU", variant="primary")
            output = gr.Audio(label="Generated speech", type="filepath")
            generate.click(
                fn=generate_speech,
                inputs=[prompt, voice, reference, frames],
                outputs=[output],
                show_progress="full",
            )

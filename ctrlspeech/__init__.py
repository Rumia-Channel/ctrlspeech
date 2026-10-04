"""CtrlSpeech — controllable expressive TTS with coarse-to-fine latent control."""

# Request ONNX Runtime telemetry opt-out before constructing speech sessions.
# This runtime API is applied after native import; it is not a network sandbox.
import onnxruntime as _onnxruntime

_onnxruntime.disable_telemetry_events()
del _onnxruntime

from .assets import MODELS, Assets, ModelSpec, download_assets
from .pipeline import (
    FPS,
    HOP_LENGTH,
    LOUDNESS_BINS,
    PITCH_BINS,
    SAMPLE_RATE,
    Baseline,
    CtrlSpeech,
    Generation,
    shift_loudness_db,
    shift_pitch_semitones,
)

__version__ = "0.1.0"

__all__ = [
    "Assets",
    "Baseline",
    "CtrlSpeech",
    "FPS",
    "Generation",
    "HOP_LENGTH",
    "LOUDNESS_BINS",
    "MODELS",
    "ModelSpec",
    "PITCH_BINS",
    "SAMPLE_RATE",
    "download_assets",
    "shift_loudness_db",
    "shift_pitch_semitones",
]

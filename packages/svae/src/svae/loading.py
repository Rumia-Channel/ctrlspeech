"""Self-contained SARA pairs and frozen codec APIs used by TTS."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import load_file

from .models.sara import SARA, SARADecoder, SARAEncoder, validate_config
from .constants import DEFAULT_LATENT_DIM

CODEC_FORMAT_VERSION = 2
DEFAULT_CODEC_CONFIG = {
    "sample_rate": 48000, "vae_dim": DEFAULT_LATENT_DIM,
    "encoder_dim": 64, "acoustic_dim": 1024, "encoder_rates": [4, 3, 4, 4, 5],
    "decoder_dim": 1024, "decoder_rates": [5, 4, 4, 3, 4],
    "resblock_kernel_sizes": [3, 7, 11], "resblock_dilations": [1, 3, 5],
    "semantic_model": "facebook/w2v-bert-2.0", "semantic_revision": None,
}


def build_codec(config, *, pretrained=True):
    """Training starts from complete SSL weights; offline loading uses saved architecture."""
    return SARA(copy.deepcopy(dict(config)), pretrained=pretrained)


def read_codec_config(directory):
    with (Path(directory) / "metainfo.json").open(encoding="utf-8") as stream:
        document = json.load(stream)
    if document.get("sara_format_version") != CODEC_FORMAT_VERSION or "SARA" not in document:
        raise ValueError("incompatible codec: export a 48 kHz / 50 Hz SARA pair and regenerate TTS features")
    config = document["SARA"]
    validate_config(config)
    return config


def encoder_state(state):
    return {key: value for key, value in state.items() if not key.startswith("decoder.")}


def trainable_state(state):
    """Frozen SSL weights live once in encoder.safetensors, not in EMA snapshots."""
    return {key: value for key, value in state.items() if not key.startswith("semantic.")}


def _selected_weights(directory, *, use_ema, prefixes=None):
    path = Path(directory) / ("ema.safetensors" if use_ema else "weights.safetensors")
    with safe_open(str(path), framework="pt", device="cpu") as tensors:
        return {key: tensors.get_tensor(key) for key in tensors.keys()
                if prefixes is None or key.startswith(prefixes)}


def build_encoder(directory):
    """Build architecture only; a complete encoder state is required before use."""
    return SARAEncoder(read_codec_config(directory), pretrained=False).eval()


def load_codec(directory, *, use_ema=True):
    codec = build_codec(read_codec_config(directory), pretrained=False)
    with safe_open(str(Path(directory) / "encoder.safetensors"), framework="pt", device="cpu") as tensors:
        state = {key: tensors.get_tensor(key) for key in tensors.keys() if key.startswith("semantic.")}
    state.update(_selected_weights(directory, use_ema=use_ema))
    codec.load_state_dict(state, strict=True, assign=True)
    return codec.eval()


def load_encoder(directory, *, use_ema=True):
    encoder = build_encoder(directory)
    state = load_file(str(Path(directory) / "encoder.safetensors"))
    if not use_ema:
        state.update(encoder_state(_selected_weights(directory, use_ema=False)))
    encoder.load_state_dict(state, strict=True, assign=True)
    return encoder.eval().requires_grad_(False)


def load_decoder(directory, *, use_ema=True):
    """Load just native HiFiGAN: no SSL construction, downloads or SSL tensors."""
    decoder = SARADecoder(read_codec_config(directory))
    decoder.load_state_dict(_selected_weights(directory, use_ema=use_ema, prefixes=("decoder.",)),
                            strict=True, assign=True)
    return decoder.eval().requires_grad_(False)


@torch.no_grad()
def encode_audio(signal, encoder):
    """Sample [B,D,T] latents once; skip the training-only KL computation."""
    mu, logvar = encoder.posterior(signal)
    return encoder.reparameterize(mu, logvar).transpose(1, 2)

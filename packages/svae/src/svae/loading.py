"""Paired codec exports and the frozen encoder/decoder APIs used by TTS."""

from __future__ import annotations

import copy
import json
import math
from pathlib import Path

import torch

from .models.dac.model import DAC

DEFAULT_DECODER_CONFIG = {
    "resblock": "1", "num_mels": 64,
    "upsample_rates": [5, 5, 2, 2, 2, 2],
    "upsample_kernel_sizes": [9, 9, 4, 4, 4, 4],
    "upsample_initial_channel": 1024,
    "resblock_kernel_sizes": [3, 7, 11],
    "resblock_dilation_sizes": [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
    "use_tanh_at_final": False, "use_bias_at_final": False,
    "activation": "snakebeta", "snake_logscale": True,
}
DEFAULT_CODEC_CONFIG = {
    "encoder_dim": 64, "encoder_rates": [4, 4, 5, 5], "latent_dim": None,
    "decoder_dim": 1024, "decoder_rates": [5, 5, 4, 4],
    "vae_dim": 64, "sample_rate": 16000, "distill": True,
    "distill_hidden_dim": 1024, "decoder_type": "bigvgan",
    "pre_vae_block": False, "attn_proj": True, "post_vae_block": False,
}


def build_codec(config):
    """Instantiate an architecture, never infer pretrained weights from metadata."""
    options = copy.deepcopy(dict(config))
    decoder_config = options.pop("decoder_dict", copy.deepcopy(DEFAULT_DECODER_CONFIG))
    encoder_hop = math.prod(options.get("encoder_rates", [2, 4, 8, 8]))
    decoder_rates = (decoder_config["upsample_rates"] if options.get("decoder_type", "dac") == "bigvgan"
                     else options.get("decoder_rates", [8, 8, 4, 2]))
    if math.prod(decoder_rates) != encoder_hop:
        raise ValueError("decoder upsampling factor must match the encoder hop")
    if options.get("decoder_type", "dac") == "bigvgan":
        if decoder_config.get("num_mels") != options.get("vae_dim", 8):
            raise ValueError("BigVGAN num_mels must match the codec vae_dim")
        if decoder_config.get("use_cuda_kernel", False):
            raise ValueError("codec training and exports require the native Torch vocoder path")
    return DAC(**options, decoder_dict=decoder_config)


def read_codec_config(directory):
    with (Path(directory) / "metainfo.json").open(encoding="utf-8") as stream:
        return json.load(stream)["DAC"]


def encoder_state(state):
    """Extract the complete encoder side from a full paired codec state."""
    return {key: value for key, value in state.items()
            if not key.startswith(("projectors.", "decoder.", "decoder_proj."))}


def _weights(directory, use_ema):
    path = Path(directory) / "dac" / ("ema_state_dict.pth" if use_ema else "weights.pth")
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    state = ({key.removeprefix("ema_model."): value for key, value in checkpoint.items()}
             if use_ema else checkpoint["state_dict"])
    return {key: value for key, value in state.items() if key not in {"initted", "step"}}


def build_encoder(directory):
    encoder = build_codec(read_codec_config(directory))
    del encoder.projectors, encoder.decoder, encoder.decoder_proj
    return encoder.eval()


def load_codec(directory, *, use_ema=True):
    codec = build_codec(read_codec_config(directory))
    codec.load_state_dict(_weights(directory, use_ema), strict=True)
    return codec.eval()


def load_encoder(directory, *, use_ema=True):
    encoder = build_encoder(directory)
    encoder.load_state_dict(encoder_state(_weights(directory, use_ema)), strict=True)
    return encoder.eval().requires_grad_(False)


def load_decoder(directory, *, use_ema=True):
    decoder = build_codec(read_codec_config(directory))
    del decoder.projectors, decoder.encoder, decoder.pre_block, decoder.fc_mu, decoder.fc_var
    if decoder.decoder_type == "bigvgan":
        del decoder.decoder_proj
    state = _weights(directory, use_ema)
    excluded = ("projectors.", "encoder.", "pre_block.", "fc_mu.", "fc_var.")
    if decoder.decoder_type == "bigvgan":
        excluded += ("decoder_proj.",)
    decoder.load_state_dict({key: value for key, value in state.items()
                            if not key.startswith(excluded)}, strict=True)
    return decoder.eval().requires_grad_(False)


@torch.no_grad()
def encode_audio(signal, encoder):
    """Sample [B, D, T] latents once; skip the training-only KL computation."""
    audio = encoder.preprocess(signal, encoder.sample_rate)
    mu, logvar = encoder.posterior(audio)
    return encoder.reparameterize(mu, logvar).transpose(1, 2)

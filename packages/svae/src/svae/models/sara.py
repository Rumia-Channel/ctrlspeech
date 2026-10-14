"""SARA-based dual-stream VAE; native Torch, no accelerator-specific extensions.

Dual-stream topology follows arXiv:2606.11611, extended to native 48 kHz by
doubling the first acoustic stride and final decoder stride (960-sample hop).
Latent size is configurable (default 128); widths/presets are local choices.
"""

from __future__ import annotations

import copy
import math

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.parametrizations import weight_norm

from ..constants import AUDIO_SAMPLE_RATE, LATENT_HOP_LENGTH
from .semantic import SemanticEncoder


class Snake(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.log_alpha = nn.Parameter(torch.zeros(1, channels, 1))

    def forward(self, signal):
        alpha = self.log_alpha.exp().clamp_min(1e-4)
        return signal + torch.sin(alpha * signal).square() / alpha


def conv(in_channels, out_channels, kernel, *, dilation=1, stride=1, padding=None):
    if padding is None:
        padding = dilation * (kernel - 1) // 2
    return weight_norm(nn.Conv1d(in_channels, out_channels, kernel, stride=stride,
                               dilation=dilation, padding=padding))


class AcousticResidual(nn.Module):
    def __init__(self, channels, dilation):
        super().__init__()
        self.block = nn.Sequential(Snake(channels), conv(channels, channels, 7, dilation=dilation),
                                   Snake(channels), conv(channels, channels, 1))

    def forward(self, signal):
        return signal + self.block(signal)


class AcousticEncoder(nn.Module):
    def __init__(self, channels, output_dim, rates):
        super().__init__()
        blocks = [conv(1, channels, 7)]
        for rate in rates:
            blocks.extend(AcousticResidual(channels, dilation) for dilation in (1, 3, 9))
            blocks.extend((Snake(channels), conv(channels, channels * 2, 2 * rate,
                                                stride=rate, padding=math.ceil(rate / 2))))
            channels *= 2
        blocks.extend((Snake(channels), conv(channels, output_dim, 3)))
        self.cnn = nn.Sequential(*blocks)
        self.lstm = nn.LSTM(output_dim, output_dim, num_layers=2, batch_first=True,
                            bidirectional=False)

    def forward(self, signal):
        features = self.cnn(signal).transpose(1, 2)
        return self.lstm(features)[0]


class HiFiResidual(nn.Module):
    def __init__(self, channels, kernel, dilations):
        super().__init__()
        self.first = nn.ModuleList(conv(channels, channels, kernel, dilation=d) for d in dilations)
        self.second = nn.ModuleList(conv(channels, channels, kernel) for _ in dilations)

    def forward(self, signal):
        for first, second in zip(self.first, self.second):
            residual = first(F.leaky_relu(signal, .1))
            signal = signal + second(F.leaky_relu(residual, .1))
        return signal


class HiFiGAN(nn.Module):
    def __init__(self, latent_dim, channels, rates, kernels, dilations):
        super().__init__()
        self.pre = conv(latent_dim, channels, 7)
        self.upsamples, self.mrfs = nn.ModuleList(), nn.ModuleList()
        for rate in rates:
            output_channels = channels // 2
            self.upsamples.append(weight_norm(nn.ConvTranspose1d(
                channels, output_channels, 2 * rate, stride=rate,
                padding=math.ceil(rate / 2), output_padding=rate % 2)))
            self.mrfs.append(nn.ModuleList(HiFiResidual(output_channels, kernel, dilations) for kernel in kernels))
            channels = output_channels
        self.post = conv(channels, 1, 7)

    def forward(self, latent):
        signal = self.pre(latent)
        for upsample, residuals in zip(self.upsamples, self.mrfs):
            signal = upsample(F.leaky_relu(signal, .1))
            signal = sum(residual(signal) for residual in residuals) / len(residuals)
        return torch.tanh(self.post(F.leaky_relu(signal, .1)))


def validate_config(config):
    allowed = {"sample_rate", "vae_dim", "encoder_dim", "acoustic_dim", "encoder_rates",
               "decoder_dim", "decoder_rates", "resblock_kernel_sizes", "resblock_dilations",
               "semantic_model", "semantic_revision", "semantic_config", "semantic_extractor_config"}
    if set(config) - allowed:
        raise ValueError("unsupported SARA configuration keys: " + ", ".join(sorted(set(config) - allowed)))
    if config.get("sample_rate") != AUDIO_SAMPLE_RATE:
        raise ValueError("SARA requires 48 kHz audio")
    if tuple(config.get("encoder_rates", ())) != (4, 3, 4, 4, 5):
        raise ValueError("48 kHz SARA acoustic strides must be [4,3,4,4,5] (50 Hz)")
    if math.prod(config.get("decoder_rates", ())) != LATENT_HOP_LENGTH:
        raise ValueError("decoder upsampling factor must match the encoder hop (960)")
    for key in ("encoder_dim", "acoustic_dim", "decoder_dim", "vae_dim"):
        if type(config.get(key)) is not int or config[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    rates = config["decoder_rates"]
    if any(type(rate) is not int or rate < 2 for rate in rates) or config["decoder_dim"] < 2 ** len(rates):
        raise ValueError("decoder rates/channels do not define valid upsampling stages")
    if not config.get("resblock_kernel_sizes") or not config.get("resblock_dilations"):
        raise ValueError("HiFiGAN requires nonempty residual kernels and dilations")
    if any(type(kernel) is not int or kernel < 1 or kernel % 2 == 0 for kernel in config["resblock_kernel_sizes"]):
        raise ValueError("HiFiGAN residual kernels must be positive odd integers")
    if any(type(d) is not int or d < 1 for d in config["resblock_dilations"]):
        raise ValueError("HiFiGAN dilations must be positive integers")


class SARADecoder(nn.Module):
    sample_rate = AUDIO_SAMPLE_RATE
    hop_length = LATENT_HOP_LENGTH

    def __init__(self, config):
        super().__init__()
        validate_config(config)
        self.vae_dim = config["vae_dim"]
        self.decoder = HiFiGAN(self.vae_dim, config["decoder_dim"], config["decoder_rates"],
                               config["resblock_kernel_sizes"], config["resblock_dilations"])

    def decode(self, latent):
        if latent.ndim != 3 or latent.shape[-1] != self.vae_dim or latent.shape[1] < 1:
            raise ValueError(f"decoder expects [batch,frames,{self.vae_dim}] latents")
        return self.decoder(latent.transpose(1, 2))


class SARAEncoder(nn.Module):
    sample_rate = AUDIO_SAMPLE_RATE
    hop_length = LATENT_HOP_LENGTH

    def __init__(self, config, *, pretrained=True):
        super().__init__()
        validate_config(config)
        self.vae_dim = config["vae_dim"]
        self.codec_config = copy.deepcopy(config)
        self.semantic = SemanticEncoder(config, pretrained=pretrained)
        if pretrained:
            self.codec_config.update(self.semantic.architecture_config())
        self.encoder = AcousticEncoder(config["encoder_dim"], config["acoustic_dim"], config["encoder_rates"])
        self.fusion = nn.Linear(config["acoustic_dim"] + self.semantic.hidden_size, self.vae_dim)
        self.fc_mu, self.fc_var = nn.Linear(self.vae_dim, self.vae_dim), nn.Linear(self.vae_dim, self.vae_dim)

    def train(self, mode=True):
        super().train(mode)
        self.semantic.eval()
        return self


    def preprocess(self, audio, sample_rate=None):
        if sample_rate is not None and sample_rate != self.sample_rate:
            raise ValueError("SARA input must be resampled to 48 kHz before encoding")
        if audio.ndim != 3 or audio.shape[1] != 1 or audio.shape[-1] < 1:
            raise ValueError("SARA input must be [batch,1,samples] mono audio")
        padding = (-audio.shape[-1]) % self.hop_length
        return F.pad(audio, (0, padding)) if padding else audio

    def posterior(self, audio):
        audio = self.preprocess(audio, self.sample_rate)
        acoustic = self.encoder(audio)
        semantic = self.semantic(audio, frames=acoustic.shape[1])
        features = self.fusion(torch.cat((acoustic, semantic.to(acoustic.dtype)), dim=-1))
        return self.fc_mu(features), self.fc_var(features).clamp(-12, 12)

    @staticmethod
    def reparameterize(mu, logvar):
        std = torch.exp(.5 * logvar.float())
        return (mu.float() + torch.randn_like(std) * std).to(mu.dtype)

    @staticmethod
    def compute_kl_loss(mu, logvar):
        mu, logvar = mu.float(), logvar.float()
        return .5 * (mu.square() + logvar.expm1() - logvar).sum(-1).mean()


class SARA(SARAEncoder):
    def __init__(self, config, *, pretrained=True):
        super().__init__(config, pretrained=pretrained)
        self.decoder = HiFiGAN(self.vae_dim, config["decoder_dim"], config["decoder_rates"],
                               config["resblock_kernel_sizes"], config["resblock_dilations"])

    def decode(self, latent):
        return SARADecoder.decode(self, latent)

    def forward(self, audio, sample_rate=None):
        length = audio.shape[-1]
        padded = self.preprocess(audio, sample_rate)
        mu, logvar = self.posterior(padded)
        latent = self.reparameterize(mu, logvar)
        return {"audio": self.decode(latent)[..., :length], "z": latent,
                "mu": mu, "log_var": logvar, "vae/kl_loss": self.compute_kl_loss(mu, logvar)}

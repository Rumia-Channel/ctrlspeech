"""Multi-period and multi-band STFT critics without accelerator-specific kernels."""

import math

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.parametrizations import weight_norm


class _Critic(nn.Module):
    def __init__(self, inputs, channels, kernel, stride):
        super().__init__()
        sizes = (inputs, channels, channels * 2, channels * 4)
        self.layers = nn.ModuleList([
            weight_norm(nn.Conv2d(first, last, kernel, stride=stride,
                                 padding=tuple(k // 2 for k in kernel)))
            for first, last in zip(sizes, sizes[1:])
        ])
        self.output = weight_norm(nn.Conv2d(sizes[-1], 1, (3, 1), padding=(1, 0)))

    def forward(self, value):
        features = []
        for layer in self.layers:
            value = F.leaky_relu(layer(value), 0.1)
            features.append(value)
        logits = self.output(value)
        return logits, features


class SpeechDiscriminator(nn.Module):
    def __init__(self, periods=(2, 3, 5, 7, 11), fft_sizes=(2048, 1024, 512),
                 bands=((0, .1), (.1, .25), (.25, .5), (.5, .75), (.75, 1.)), channels=32):
        super().__init__()
        if type(channels) is not int or channels < 1:
            raise ValueError("discriminator channels must be positive")
        if any(type(n) is not int or n < 2 for n in periods):
            raise ValueError("periods must be integers >=2")
        if any(type(n) is not int or n < 16 for n in fft_sizes):
            raise ValueError("STFT sizes must be integers >=16")
        if not periods and not fft_sizes:
            raise ValueError("at least one discriminator is required")
        if fft_sizes and (not bands or any(not (0 <= first < last <= 1) for first, last in bands)):
            raise ValueError("bands must be nonempty intervals within [0,1]")
        self.periods, self.fft_sizes, self.bands = tuple(periods), tuple(fft_sizes), tuple(bands)
        self.period_critics = nn.ModuleList([_Critic(1, channels, (5, 1), (3, 1)) for _ in periods])
        self.spectral_critics = nn.ModuleList([_Critic(2, channels, (3, 3), (1, 2)) for _ in fft_sizes for _ in bands])
        self.band_slices = []
        for index, size in enumerate(self.fft_sizes):
            self.register_buffer(f"window_{index}", torch.hann_window(size))
            bins = size // 2 + 1
            slices = [(math.floor(first * bins), math.ceil(last * bins)) for first, last in self.bands]
            if any(last <= first for first, last in slices):
                raise ValueError("STFT band contains no frequency bins")
            self.band_slices.append(slices)

    def forward(self, audio):
        if audio.ndim != 3 or audio.shape[1] != 1 or not audio.shape[-1]:
            raise ValueError("discriminator input must have shape [batch,1,samples]")
        outputs = []
        for period, critic in zip(self.periods, self.period_critics):
            pad = (-audio.shape[-1]) % period
            value = F.pad(audio, (0, pad), mode="replicate") if pad else audio
            outputs.append(critic(value.reshape(value.shape[0], 1, -1, period)))
        for index, size in enumerate(self.fft_sizes):
            with torch.autocast(device_type=audio.device.type, enabled=False):
                spectrum = torch.stft(audio[:, 0].float(), n_fft=size, hop_length=size // 4,
                                      window=getattr(self, f"window_{index}"), center=True,
                                      pad_mode="constant", return_complex=True)
                value = torch.view_as_real(spectrum).permute(0, 3, 1, 2)
            for band_index, (first, last) in enumerate(self.band_slices[index]):
                critic = self.spectral_critics[index * len(self.bands) + band_index]
                outputs.append(critic(value[:, :, first:last]))
        return outputs

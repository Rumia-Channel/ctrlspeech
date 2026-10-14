"""Differentiable waveform reconstruction and hinge-GAN objectives."""

import torch
import torchaudio
from torch import nn
from torch.nn import functional as F


class SpectralReconstructionLoss(nn.Module):
    def __init__(self, sample_rate=48000, fft_sizes=(4096, 2048, 1024), n_mels=80):
        super().__init__()
        if not fft_sizes or any(type(n) is not int or n < 16 for n in fft_sizes):
            raise ValueError("fft_sizes must contain integers >=16")
        if type(n_mels) is not int or n_mels < 1:
            raise ValueError("n_mels must be positive")
        self.fft_sizes = tuple(fft_sizes)
        for index, size in enumerate(self.fft_sizes):
            self.register_buffer(f"window_{index}", torch.hann_window(size))
            bank = torchaudio.functional.melscale_fbanks(size // 2 + 1, 0, sample_rate / 2,
                                                        n_mels, sample_rate, norm="slaney", mel_scale="slaney")
            self.register_buffer(f"mel_{index}", bank)

    def forward(self, reconstructed, reference):
        if reconstructed.shape != reference.shape or reconstructed.ndim != 3 or reconstructed.shape[1] != 1:
            raise ValueError("reconstruction and reference must have identical [batch,1,samples] shapes")
        with torch.autocast(device_type=reconstructed.device.type, enabled=False):
            fake, real = reconstructed[:, 0].float(), reference[:, 0].detach().float()
            total = fake.new_zeros(())
            for index, size in enumerate(self.fft_sizes):
                options = dict(n_fft=size, hop_length=size // 4, window=getattr(self, f"window_{index}"),
                               center=True, pad_mode="constant", return_complex=True)
                bank = getattr(self, f"mel_{index}")
                actual = torch.stft(fake, **options).abs().transpose(1, 2) @ bank
                expected = torch.stft(real, **options).abs().transpose(1, 2) @ bank
                total = total + F.l1_loss(actual, expected) + F.l1_loss(actual.clamp_min(1e-5).log(), expected.clamp_min(1e-5).log())
            return total / len(self.fft_sizes)


def discriminator_loss(real_outputs, fake_outputs):
    if not real_outputs or len(real_outputs) != len(fake_outputs):
        raise ValueError("real/fake discriminator counts must match and be nonempty")
    return torch.stack([(F.relu(1 - real.float()).mean() + F.relu(1 + fake.float()).mean())
                        for (real, _), (fake, _) in zip(real_outputs, fake_outputs)]).mean()


def generator_adversarial_loss(fake_outputs):
    if not fake_outputs:
        raise ValueError("at least one discriminator is required")
    return torch.stack([-logits.float().mean() for logits, _ in fake_outputs]).mean()


def feature_matching_loss(real_outputs, fake_outputs):
    if not real_outputs or len(real_outputs) != len(fake_outputs):
        raise ValueError("real/fake discriminator counts must match and be nonempty")
    losses = []
    for (_, real), (_, fake) in zip(real_outputs, fake_outputs):
        if not real or len(real) != len(fake):
            raise ValueError("discriminator feature layer counts must match")
        losses.append(torch.stack([F.l1_loss(b.float(), a.detach().float()) for a, b in zip(real, fake)]).mean())
    return torch.stack(losses).mean()

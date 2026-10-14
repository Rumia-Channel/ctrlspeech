import math
from typing import List, Union

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from .base_model import BaseModel, CodecMixin
from .layers import Snake1d, WNConv1d, WNConvTranspose1d
from .bigvgan import BigVGAN
from .regulator import InterpolateRegulator
from .attn_proj import AttnProjection




class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


def init_weights(m):
    if isinstance(m, nn.Conv1d):
        nn.init.trunc_normal_(m.weight, std=0.02)
        nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        nn.init.constant_(m.bias, 0)


class ResidualUnit(nn.Module):
    def __init__(self, dim: int = 16, dilation: int = 1):
        super().__init__()
        pad = ((7 - 1) * dilation) // 2
        self.block = nn.Sequential(
            Snake1d(dim),
            WNConv1d(dim, dim, kernel_size=7, dilation=dilation, padding=pad),
            Snake1d(dim),
            WNConv1d(dim, dim, kernel_size=1),
        )

    def forward(self, x):
        y = self.block(x)
        pad = (x.shape[-1] - y.shape[-1]) // 2
        if pad > 0:
            x = x[..., pad:-pad]
        return x + y


class EncoderBlock(nn.Module):
    def __init__(self, dim: int = 16, stride: int = 1):
        super().__init__()
        self.block = nn.Sequential(
            ResidualUnit(dim // 2, dilation=1),
            ResidualUnit(dim // 2, dilation=3),
            ResidualUnit(dim // 2, dilation=9),
            Snake1d(dim // 2),
            WNConv1d(
                dim // 2,
                dim,
                kernel_size=2 * stride,
                stride=stride,
                padding=math.ceil(stride / 2),
            ),
        )

    def forward(self, x):
        return self.block(x)


class Encoder(nn.Module):
    def __init__(
        self,
        d_model: int = 64,
        strides: list = [2, 4, 8, 8],
        d_latent: int = 64,
    ):
        super().__init__()
        # Create first convolution
        self.block = [WNConv1d(1, d_model, kernel_size=7, padding=3)]

        # Create EncoderBlocks that double channels as they downsample by `stride`
        for stride in strides:
            d_model *= 2
            self.block += [EncoderBlock(d_model, stride=stride)]

        # Create last convolution
        self.block += [
            Snake1d(d_model),
            WNConv1d(d_model, d_latent, kernel_size=3, padding=1),
        ]

        # Wrap black into nn.Sequential
        self.block = nn.Sequential(*self.block)
        self.enc_dim = d_model

    def forward(self, x):
        return self.block(x)


class DecoderBlock(nn.Module):
    def __init__(self, input_dim: int = 16, output_dim: int = 8, stride: int = 1):
        super().__init__()
        self.block = nn.Sequential(
            Snake1d(input_dim),
            WNConvTranspose1d(
                input_dim,
                output_dim,
                kernel_size=2 * stride,
                stride=stride,
                padding=math.ceil(stride / 2),
                output_padding=0 if stride % 2 == 0 else 1,
            ),
            ResidualUnit(output_dim, dilation=1),
            ResidualUnit(output_dim, dilation=3),
            ResidualUnit(output_dim, dilation=9),
        )

    def forward(self, x):
        return self.block(x)


class Decoder(nn.Module):
    def __init__(
        self,
        input_channel,
        channels,
        rates,
        d_out: int = 1,
    ):
        super().__init__()

        # Add first conv layer
        layers = [WNConv1d(input_channel, channels, kernel_size=7, padding=3)]

        # Add upsampling + MRF blocks
        for i, stride in enumerate(rates):
            input_dim = channels // 2**i
            output_dim = channels // 2 ** (i + 1)
            layers += [DecoderBlock(input_dim, output_dim, stride)]

        # Add final conv layer
        layers += [
            Snake1d(output_dim),
            WNConv1d(output_dim, d_out, kernel_size=7, padding=3),
            nn.Tanh(),
        ]

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class ResidualBottleneck(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.LayerNorm(out_dim),  # 替换为 LayerNorm
            nn.GELU(),
            nn.Linear(out_dim, out_dim),
            nn.LayerNorm(out_dim),  # 替换为 LayerNorm
        )
        self.shortcut = nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()

    def forward(self, x):
        return self.block(x) + self.shortcut(x)


class DAC(BaseModel, CodecMixin):
    def __init__(
        self,
        encoder_dim: int = 64,
        encoder_rates: List[int] = [2, 4, 8, 8],
        latent_dim: int = None,
        decoder_dim: int = 1536,
        decoder_rates: List[int] = [8, 8, 4, 2],
        vae_dim: Union[int, list] = 8,
        sample_rate: int = 44100,
        distill: bool = False,
        distill_hidden_dim: int = 1024,
        decoder_type: str = "dac",  # bigvgan | dac
        pre_vae_block: bool = False,
        attn_proj: bool = False,
        post_vae_block: bool = False,
        decoder_dict: dict = {},
        **kwargs,
        # bigvgan_conf: str = "/inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/niuzhikang-240108120093/descript-audio-codec/conf/bigvgan_conf/bigvgan_v2_24khz_100band_256x.json"
    ):
        super().__init__()

        self.encoder_dim = encoder_dim
        self.encoder_rates = encoder_rates
        self.decoder_dim = decoder_dim
        self.decoder_rates = decoder_rates
        self.sample_rate = sample_rate

        if latent_dim is None:
            latent_dim = encoder_dim * (2 ** len(encoder_rates))

        self.latent_dim = latent_dim

        self.hop_length = np.prod(encoder_rates)
        self.sample_rate = sample_rate
        self.encoder = Encoder(encoder_dim, encoder_rates, latent_dim)
        self.vae_dim = vae_dim
        self.attn_proj = attn_proj
        self.pre_vae_block = pre_vae_block
        self.post_vae_block = post_vae_block

        if self.pre_vae_block:
            self.pre_block = self._build_residual_blocks(latent_dim, self.vae_dim)
        elif self.attn_proj:
            # in_dim, out_dim, num_heads, norm_layer=nn.LayerNorm, mlp_ratio=2
            self.pre_block = AttnProjection(latent_dim, self.vae_dim, num_heads=8)
        else:
            self.pre_block = nn.Linear(latent_dim, self.vae_dim)
        self.fc_mu = nn.Linear(self.vae_dim, self.vae_dim)
        self.fc_var = nn.Linear(self.vae_dim, self.vae_dim)

        if self.post_vae_block:
            self.decoder_proj = self._build_residual_blocks(self.vae_dim, latent_dim)
        elif self.attn_proj:
            self.decoder_proj = AttnProjection(self.vae_dim, latent_dim, 8)
        else:
            self.decoder_proj = nn.Linear(self.vae_dim, latent_dim)

        self.decoder_type = decoder_type
        if self.decoder_type == "dac":
            self.decoder = Decoder(
                latent_dim,
                decoder_dim,
                decoder_rates,
            )
            self.apply(init_weights)
        elif self.decoder_type == "bigvgan":
            # self.bigvgan_conf = bigvgan_conf
            # with open(self.bigvgan_conf) as f:
            #     data = f.read()
            # json_config = json.loads(data)
            # h = AttrDict(json_config)
            h = AttrDict(**decoder_dict)
            self.decoder = BigVGAN(h)

        self.distill = distill
        self.distill_hidden_dim = distill_hidden_dim
        proj_dim = self.distill_hidden_dim * 2
        self.projectors = InterpolateRegulator([0, 1], self.vae_dim, proj_dim, self.distill_hidden_dim)

        self.delay = self.get_delay()

    def _build_residual_blocks(self, in_dim, out_dim):
        layers = []
        current_dim = in_dim
        max_depth = 3

        # 统一处理维度变换
        if in_dim != out_dim:
            step_fn = (lambda x: max(x // 2, out_dim)) if in_dim > out_dim else (lambda x: min(x * 2, out_dim))
            for _ in range(max_depth):
                next_dim = step_fn(current_dim)
                layers.append(ResidualBottleneck(current_dim, next_dim))
                current_dim = next_dim
                if current_dim == out_dim:
                    break
            # 强制最终维度对齐
            if current_dim != out_dim:
                layers.append(
                    nn.Sequential(
                        nn.Linear(current_dim, out_dim),
                        # nn.BatchNorm1d(out_dim),
                        nn.GELU(),
                    )
                )
        else:
            # 维度相同则构建恒等残差块
            layers.append(ResidualBottleneck(in_dim, out_dim))

        return nn.Sequential(*layers)

    def preprocess(self, audio_data, sample_rate):
        if sample_rate is None:
            sample_rate = self.sample_rate
        assert sample_rate == self.sample_rate

        length = audio_data.shape[-1]
        right_pad = math.ceil(length / self.hop_length) * self.hop_length - length
        audio_data = nn.functional.pad(audio_data, (0, right_pad))

        return audio_data

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor):
        # Posterior arithmetic stays FP32 under AMP; return the model's dtype.
        std = torch.exp(0.5 * logvar.float())
        return (torch.randn_like(std) * std + mu.float()).to(mu.dtype)

    def compute_kl_loss(self, mu, log_var):
        # exp(12) exceeds FP16's range. expm1 also avoids cancellation near zero.
        mu, log_var = mu.float(), log_var.float()
        return 0.5 * (mu.square() + log_var.expm1() - log_var).sum(dim=-1).mean()

    def posterior(self, audio_data: torch.Tensor):
        z = self.encoder(audio_data).transpose(1, 2)
        z = self.pre_block(z)
        mu = self.fc_mu(z)
        log_var = self.fc_var(z).clamp(min=-12, max=12)
        return mu, log_var

    def encode(self, audio_data: torch.Tensor):
        mu, log_var = self.posterior(audio_data)
        return self.reparameterize(mu, log_var), mu, log_var, self.compute_kl_loss(mu, log_var)

    def decode(self, z: torch.Tensor):
        if self.decoder_type == "dac":
            z = self.decoder_proj(z).transpose(1, 2)
            recon = self.decoder(z)
        elif self.decoder_type == "bigvgan":
            recon = self.decoder(z.transpose(1, 2))
        return recon

    def forward(
        self,
        audio_data: torch.Tensor,  # B, 1, T (duration)
        sample_rate: int = None,
        guidance: torch.Tensor = None,  # B, T, D
    ):
        length = audio_data.shape[-1]
        audio_data = self.preprocess(audio_data, sample_rate)
        z, mu, log_var, kl_loss = self.encode(audio_data)
        proj_loss = z.new_zeros((), dtype=torch.float32)
        if self.distill and (self.training or guidance is not None):
            if guidance is None or len(guidance) != len(z):
                raise ValueError("semantic distillation requires one teacher sequence per waveform")
            if any(g.ndim != 2 or g.shape[0] < 1 or g.shape[1] != self.distill_hidden_dim for g in guidance):
                raise ValueError("teacher features must have shape [frames, distill_hidden_dim]")
            lengths = [g.shape[0] for g in guidance]
            projected, _ = self.projectors(z, [z.shape[1]] * len(z), lengths)
            with torch.autocast(device_type=z.device.type, enabled=False):
                if isinstance(guidance, torch.Tensor):
                    proj_loss = -F.cosine_similarity(projected.float(), guidance.float(), dim=-1).mean()
                else:
                    proj_loss = torch.stack([
                        -F.cosine_similarity(pi[:length].float(), gi.float(), dim=-1).mean()
                        for pi, gi, length in zip(projected, guidance, lengths)
                    ]).mean()

        x = self.decode(z)
        return {
            "audio": x[..., :length],
            "z": z,
            "mu": mu,
            "log_var": log_var,
            "vae/kl_loss": kl_loss,
            "vae/proj_loss": proj_loss,
        }

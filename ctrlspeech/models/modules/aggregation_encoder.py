"""Patch-local acoustic aggregation.

This encoder only summarizes each four-frame SVAE patch before the long-range
LFM2 autoregressive backbone. It deliberately uses a small local Transformer
rather than another language-model implementation; the old version depended on
vendored Qwen3 code even though no long-range language modeling happens here.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class AggregationEncoder(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_hidden_layers: int,
        num_attention_heads: int,
        patch_size: int,
        pool_type: str = "avg",
        dropout: float = 0.0,
        **kwargs,
    ):
        super().__init__()
        if hidden_size % num_attention_heads:
            raise ValueError("hidden_size must be divisible by num_attention_heads")
        if patch_size < 1:
            raise ValueError("patch_size must be positive")
        if pool_type not in {"avg", "cls"}:
            raise ValueError(f"Invalid pool type: {pool_type}")

        self.patch_size = int(patch_size)
        self.pool_type = pool_type

        layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_attention_heads,
            dim_feedforward=intermediate_size,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.model = nn.TransformerEncoder(
            layer,
            num_layers=num_hidden_layers,
            enable_nested_tensor=False,
        )
        self.final_norm = nn.LayerNorm(hidden_size)

        if pool_type == "cls":
            self.summary_token = nn.Parameter(
                torch.randn(1, 1, hidden_size) * 0.02
            )

    def forward(
        self,
        x: torch.Tensor,
        padding_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Aggregate patch_size acoustic frames into one token."""
        if x.ndim != 3 or padding_mask.ndim != 2:
            raise ValueError("expected x=[B,L,D] and padding_mask=[B,L]")
        if x.shape[:2] != padding_mask.shape:
            raise ValueError("x and padding_mask sequence dimensions must match")

        batch, length, dim = x.shape
        if length % self.patch_size:
            raise ValueError(
                f"sequence length {length} is not divisible by "
                f"patch_size={self.patch_size}"
            )

        num_patches = length // self.patch_size
        folded_x = x.reshape(batch * num_patches, self.patch_size, dim)
        folded_mask = padding_mask.reshape(
            batch * num_patches, self.patch_size
        ).bool()

        if self.pool_type == "cls":
            summary = self.summary_token.expand(
                batch * num_patches, -1, -1
            )
            inputs = torch.cat([summary, folded_x], dim=1)
            valid = torch.cat(
                [
                    torch.ones(
                        (batch * num_patches, 1),
                        dtype=torch.bool,
                        device=folded_mask.device,
                    ),
                    folded_mask,
                ],
                dim=1,
            )
        else:
            inputs = folded_x
            valid = folded_mask

        hidden = self.model(inputs, src_key_padding_mask=~valid)
        hidden = self.final_norm(hidden)

        if self.pool_type == "cls":
            summary_tokens = hidden[:, 0]
        else:
            weights = valid.to(hidden.dtype).unsqueeze(-1)
            denom = weights.sum(dim=1).clamp_min(1.0)
            summary_tokens = (hidden * weights).sum(dim=1) / denom

        summary_tokens = summary_tokens.reshape(batch, num_patches, dim)
        summary_padding_mask = folded_mask.any(dim=-1).reshape(
            batch, num_patches
        )
        return summary_tokens, summary_padding_mask

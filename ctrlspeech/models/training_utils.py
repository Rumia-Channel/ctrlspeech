"""Numerically stable objectives and conditioning used by DiTar training."""

import math

import torch


def masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Reduce in float32, excluding padding from both sum and denominator."""
    mask = mask.bool()
    while mask.ndim < values.ndim:
        mask = mask.unsqueeze(-1)
    mask = mask.expand_as(values)
    return values.float().masked_fill(~mask, 0).sum() / mask.sum().clamp_min(1)


def flow_matching_path(clean, noise, time, *, curved=False):
    """Return the interpolant and its exact derivative with respect to time."""
    if curved:
        angle = math.pi * time / 2
        noisy = angle.cos() * noise + angle.sin() * clean
        velocity = (math.pi / 2) * (-angle.sin() * noise + angle.cos() * clean)
        return noisy, velocity
    return (1 - time) * noise + time * clean, clean - noise


def dropout_condition(values, probability, *, training):
    """Drop independent rows without inverted-dropout scaling (CFG training)."""
    if not training or probability == 0:
        return values
    shape = (values.shape[0],) + (1,) * (values.ndim - 1)
    keep = torch.rand(shape, device=values.device) >= probability
    return values * keep.to(values.dtype)


def left_pad_text_prefix(embeds, mask, modality):
    """Compact dual text streams so every sample ends on a valid text token.

    Internal padding between native text and phones would otherwise enter LIV
    convolution history. Right padding would predict the first patch from a
    padded text token in shorter examples.
    """
    mask = mask.bool()
    if not mask.any(dim=1).all():
        raise ValueError("every sample must contain at least one text token")
    packed = torch.zeros_like(embeds)
    packed_mask = torch.zeros_like(mask)
    packed_modality = torch.zeros_like(modality)
    for row in range(embeds.shape[0]):
        count = int(mask[row].sum())
        packed[row, -count:] = embeds[row, mask[row]]
        packed_mask[row, -count:] = True
        packed_modality[row, -count:] = modality[row, mask[row]]
    return packed, packed_mask, packed_modality

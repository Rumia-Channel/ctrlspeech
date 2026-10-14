from typing import Tuple
import torch
import torch.nn as nn
from torch.nn import functional as F


class InterpolateRegulator(nn.Module):
    def __init__(
        self,
        sampling_ratios: Tuple,
        in_channels: int,
        channels: int,
        out_channels: int,
        groups: int = 1,
    ):
        super().__init__()
        self.sampling_ratios = sampling_ratios
        model = nn.ModuleList([])
        if len(sampling_ratios) > 0:
            for i, _ in enumerate(sampling_ratios):
                module = nn.Conv1d(in_channels if i == 0 else channels, channels, 3, 1, 1)
                norm = nn.GroupNorm(groups, channels)
                act = nn.Mish()
                model.extend([module, norm, act])
        model.append(
            nn.Conv1d(channels, out_channels, 1, 1)  # 这个是改channel的维度
        )
        self.model = nn.Sequential(*model)

    def forward(self, x, xlens=None, ylens=None):
        # Resample valid input regions, not batch padding; mask on the target grid.
        if x.ndim != 3 or x.shape[0] < 1 or x.shape[1] < 1:
            raise ValueError("projection input must have nonempty [batch, frames, channels] dimensions")
        batch, width, _ = x.shape
        source = [width] * batch if xlens is None else (xlens.tolist() if isinstance(xlens, torch.Tensor) else list(xlens))
        if ylens is None:
            raise ValueError("target lengths are required")
        target = ylens.tolist() if isinstance(ylens, torch.Tensor) else list(ylens)
        if len(source) != batch or len(target) != batch:
            raise ValueError("projection length counts must match the batch")
        if any(type(n) is not int or not 0 < n <= width for n in source):
            raise ValueError("source lengths must be positive integers within the input")
        if any(type(n) is not int or n < 1 for n in target):
            raise ValueError("target lengths must be positive integers")
        if all(n == source[0] for n in source) and all(n == target[0] for n in target):
            value = F.interpolate(x[:, :source[0]].transpose(1, 2), size=target[0], mode="linear", align_corners=False)
            return self.model(value).transpose(1, 2).contiguous(), ylens
        maximum = max(target)
        outputs = []
        for row, first, last in zip(x, source, target):
            value = F.interpolate(row[:first].transpose(0, 1)[None], size=last, mode="linear", align_corners=False)
            outputs.append(F.pad(self.model(value), (0, maximum - last)))
        return torch.cat(outputs, dim=0).transpose(1, 2).contiguous(), ylens

import torch

from ctrlspeech.models.embeds import DurationConditioner
from ctrlspeech.pipeline import MAX_AR_STEPS, estimate_max_seq_length
from ctrlspeech.retime import (
    MAX_DURATION_FRAMES,
    MAX_TARGET_SECONDS,
    MAX_TIMELINE_FRAMES,
)


def test_duration_conditioner_has_no_192_frame_lookup_limit():
    conditioner = DurationConditioner(dim=32, hidden_dim=16)
    frames = torch.tensor([0.0, 191.0, 6000.0, 12000.0])
    out = conditioner(frames)
    assert out.shape == (4, 32)
    assert torch.isfinite(out).all()


def test_one_minute_generation_limits_are_consistent():
    assert MAX_TARGET_SECONDS == 60
    assert MAX_TIMELINE_FRAMES == 6001
    assert MAX_DURATION_FRAMES == 6000
    assert MAX_AR_STEPS == 600
    assert estimate_max_seq_length(60) == MAX_AR_STEPS
    assert estimate_max_seq_length(120) == MAX_AR_STEPS

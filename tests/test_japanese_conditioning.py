import pytest
import torch

from ctrlspeech.models.embeds import JapaneseLinguisticConditioner


def _features():
    return {
        "accent_pitch": [[1, 2, 0, 1]],
        "phrase_boundary": [[1, 0, 0, 2]],
        "accent_nucleus": [[2, 2, 0, 2]],
        "phrase_mora_count": [[4, 4, 0, 4]],
        "valid": [[True, True, False, True]],
    }


def test_japanese_linguistic_conditioner_shape_and_separator_mask():
    conditioner = JapaneseLinguisticConditioner(dim=32)
    out = conditioner(_features(), batch_size=1, sequence_length=4)
    assert out.shape == (1, 4, 32)
    torch.testing.assert_close(out[:, 2], torch.zeros_like(out[:, 2]))


def test_japanese_linguistic_conditioner_rejects_bad_ids():
    conditioner = JapaneseLinguisticConditioner(dim=16)
    features = _features()
    features["accent_pitch"] = [[1, 3, 0, 1]]
    with pytest.raises(ValueError, match="accent_pitch ids"):
        conditioner(features, batch_size=1, sequence_length=4)


def test_japanese_linguistic_conditioner_requires_aligned_length():
    conditioner = JapaneseLinguisticConditioner(dim=16)
    with pytest.raises(ValueError, match="must have shape"):
        conditioner(_features(), batch_size=1, sequence_length=5)

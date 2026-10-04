import json

import pytest
import torch
from torch import nn

from ctrlspeech.models import vocoder


class TinyDAC(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.projectors = nn.Linear(1, 1)
        self.encoder = nn.Linear(1, 1)
        self.decoder = nn.Linear(1, 1)


@pytest.mark.parametrize('use_ema', [True, False])
def test_decoder_rejects_incomplete_checkpoint(tmp_path, monkeypatch, use_ema):
    monkeypatch.setattr(vocoder, 'DAC', TinyDAC)
    (tmp_path / 'metainfo.json').write_text(json.dumps({'DAC': {}}))
    (tmp_path / 'dac').mkdir()
    name = 'ema_state_dict.pth' if use_ema else 'weights.pth'
    state = {} if use_ema else {'state_dict': {}}
    torch.save(state, tmp_path / 'dac' / name)
    with pytest.raises(RuntimeError, match='incomplete SVAE decoder'):
        vocoder.load_decoder(str(tmp_path), use_ema=use_ema)


@pytest.mark.parametrize('use_ema', [True, False])
def test_decoder_accepts_complete_checkpoint_and_training_metadata(tmp_path, monkeypatch, use_ema):
    monkeypatch.setattr(vocoder, 'DAC', TinyDAC)
    (tmp_path / 'metainfo.json').write_text(json.dumps({'DAC': {}}))
    (tmp_path / 'dac').mkdir()
    reference = TinyDAC()
    state = reference.state_dict()
    if use_ema:
        state = {f'ema_model.{key}': value for key, value in state.items()}
        state.update(initted=torch.tensor(True), step=torch.tensor(5))
        name = 'ema_state_dict.pth'
    else:
        state = {'state_dict': state}
        name = 'weights.pth'
    torch.save(state, tmp_path / 'dac' / name)
    decoder = vocoder.load_decoder(str(tmp_path), use_ema=use_ema)
    torch.testing.assert_close(decoder.decoder.weight, reference.decoder.weight)
    assert not hasattr(decoder, 'encoder')

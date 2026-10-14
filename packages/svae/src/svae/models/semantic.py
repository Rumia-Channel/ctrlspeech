"""Frozen W2v-BERT 2.0 branch used during both training and reference encoding."""

import copy

import torch
import torchaudio
from torch import nn
from torch.nn import functional as F
from transformers import (AutoFeatureExtractor, AutoModel, SeamlessM4TFeatureExtractor,
                          Wav2Vec2BertConfig, Wav2Vec2BertModel)
from ..constants import AUDIO_SAMPLE_RATE


class SemanticEncoder(nn.Module):
    def __init__(self, config, *, pretrained):
        super().__init__()
        self.ready = False
        if pretrained:
            source, revision = config["semantic_model"], config.get("semantic_revision")
            self.extractor = AutoFeatureExtractor.from_pretrained(source, revision=revision)
            self.model, info = AutoModel.from_pretrained(source, revision=revision, output_loading_info=True)
            if info.get("missing_keys") or info.get("mismatched_keys") or info.get("error_msgs"):
                raise ValueError("semantic encoder weights are incomplete; random fallback is forbidden")
            self.ready = True
        else:
            if not config.get("semantic_config") or not config.get("semantic_extractor_config"):
                raise ValueError("offline encoder construction requires the exported SSL architecture")
            self.extractor = SeamlessM4TFeatureExtractor.from_dict(copy.deepcopy(config["semantic_extractor_config"]))
            self.model = Wav2Vec2BertModel(Wav2Vec2BertConfig.from_dict(copy.deepcopy(config["semantic_config"])))
        if not isinstance(self.model, Wav2Vec2BertModel) or not isinstance(self.extractor, SeamlessM4TFeatureExtractor):
            raise ValueError("SARA requires a Wav2Vec2BertModel and SeamlessM4TFeatureExtractor")
        if self.model.config.add_adapter or self.extractor.sampling_rate != 16000 or self.extractor.stride != 2:
            raise ValueError("SARA requires an unadapted 50 Hz W2v-BERT semantic stream")
        if self.model.config.feature_projection_input_dim != self.extractor.num_mel_bins * self.extractor.stride:
            raise ValueError("SSL feature projection and extractor dimensions differ")
        self.hidden_size = self.model.config.hidden_size
        self.resampler = torchaudio.transforms.Resample(AUDIO_SAMPLE_RATE, 16000)
        self.model.eval().requires_grad_(False)

    def architecture_config(self):
        return {"semantic_config": self.model.config.to_dict(),
                "semantic_extractor_config": self.extractor.to_dict()}

    def train(self, mode=True):
        super().train(False)
        return self

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        expected = self.state_dict()
        self.ready = all(prefix + key in state_dict and state_dict[prefix + key].shape == value.shape
                         for key, value in expected.items())
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict,
                                     missing_keys, unexpected_keys, error_msgs)

    @torch.no_grad()
    def forward(self, audio, *, frames):
        if not self.ready:
            raise ValueError("semantic encoder has no complete pretrained weights")
        with torch.autocast(audio.device.type, enabled=False):
            waveform = self.resampler(audio[:, 0].detach().float())
            # The extractor snips a 25ms window and stacks pairs of 10ms frames.
            # One extra 20ms cell exposes the final real waveform cell; crop only
            # the extractor's trailing context, never stretch the semantic clock.
            waveform = F.pad(waveform, (0, 320)).cpu()
            features = self.extractor([row.numpy() for row in waveform], sampling_rate=16000,
                                      return_tensors="pt", padding=False)
            device = next(self.model.parameters()).device
            hidden = self.model(**{key: value.to(device) for key, value in features.items()}).last_hidden_state
        if hidden.ndim != 3 or hidden.shape[-1] != self.hidden_size or hidden.shape[1] != frames:
            raise ValueError(f"SSL/acoustic clocks differ: expected {frames} semantic frames, got {tuple(hidden.shape)}")
        return hidden

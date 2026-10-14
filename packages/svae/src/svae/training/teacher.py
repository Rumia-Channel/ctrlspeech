"""Frozen SSL features; no random model or missing-weight fallback."""

import hashlib
import json
from pathlib import Path

import torch
from transformers import AutoFeatureExtractor, AutoModel

from .data import file_digest


class SemanticTeacher:
    def __init__(self, model_id, *, device, revision=None):
        self.extractor = AutoFeatureExtractor.from_pretrained(model_id, revision=revision)
        if self.extractor.sampling_rate != 16000:
            raise ValueError("semantic teacher must accept 16 kHz waveforms")
        self.model, info = AutoModel.from_pretrained(model_id, revision=revision, output_loading_info=True)
        if info.get("missing_keys") or info.get("mismatched_keys") or info.get("error_msgs"):
            raise ValueError("semantic teacher weights are incomplete; random initialization is not allowed")
        self.model.eval().requires_grad_(False)
        self.hidden_size = self.model.config.hidden_size
        identity = {"model": str(model_id), "revision": revision,
                    "commit": getattr(self.model.config, "_commit_hash", None),
                    "extractor": self.extractor.to_dict()}
        root = Path(model_id)
        if root.is_dir():
            identity["files"] = {str(path.relative_to(root)): file_digest(path)
                                 for path in sorted(root.rglob("*"))
                                 if path.is_file() and path.suffix in {".json", ".safetensors", ".bin"}}
        self.signature = hashlib.sha256(json.dumps(identity, sort_keys=True, default=str).encode()).hexdigest()
        self.device = device
        self.model.to(device)

    @torch.no_grad()
    def __call__(self, audio):
        if audio.device.type != "cpu":
            raise ValueError("teacher preprocessing requires the original CPU waveform batch")
        values = [row[0].numpy() for row in audio]
        features = self.extractor(values, sampling_rate=16000, return_tensors="pt", padding=False)
        with torch.autocast(device_type=self.device.type, enabled=False):
            result = self.model(**{key: value.to(self.device) for key, value in features.items()})
        hidden = result.last_hidden_state
        if hidden.ndim != 3 or hidden.shape[-1] != self.hidden_size:
            raise ValueError("SSL teacher must return [batch,frames,hidden_size] features")
        return hidden

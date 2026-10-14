"""Bounded waveform reads with independent, reproducible crop RNG."""

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio
from torch.utils.data import Dataset


def file_digest(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class WaveDataset(Dataset):
    sample_rate = 16000

    def __init__(self, directory, clip_samples, *, seed=42):
        self.directory = Path(directory).resolve()
        if type(clip_samples) is not int or clip_samples < 400:
            raise ValueError("clip_samples must be an integer >=400")
        self.clip_samples, self.seed, self.epoch = clip_samples, seed, 0
        self.records, self.excluded, self.resamplers = [], [], {}
        paths = sorted(path for path in self.directory.rglob("*") if path.is_file() and path.suffix.lower() == ".wav")
        if not paths:
            raise ValueError(f"no WAV files under {self.directory}")
        for path in paths:
            info = sf.info(path)
            required = math.ceil(clip_samples * info.samplerate / self.sample_rate)
            if info.frames < required:
                self.excluded.append(str(path))
                continue
            self.records.append({"path": path, "rate": info.samplerate, "frames": info.frames,
                                 "required": required, "sha256": file_digest(path)})
        if not self.records:
            raise ValueError(f"no WAV is at least {clip_samples / self.sample_rate:g} seconds under {self.directory}")
        canonical = [{**record, "path": str(record["path"])} for record in self.records]
        self.signature = hashlib.sha256(json.dumps(canonical, sort_keys=True).encode()).hexdigest()
        self.paths = {record["path"] for record in self.records}
        self.digests = {record["sha256"] for record in self.records}

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        generator = torch.Generator().manual_seed(self.seed + self.epoch * len(self) + index)
        start = int(torch.randint(record["frames"] - record["required"] + 1, (), generator=generator))
        # Context prevents resampling a cropped boundary from introducing artificial ringing.
        margin = math.ceil(64 * record["rate"] / self.sample_rate) if record["rate"] != self.sample_rate else 0
        first = max(0, start - margin)
        last = min(record["frames"], start + record["required"] + margin)
        with sf.SoundFile(record["path"]) as stream:
            stream.seek(first)
            audio = stream.read(last - first, dtype="float32", always_2d=True)
        if not np.isfinite(audio).all():
            raise ValueError(f"nonfinite waveform: {record['path']}")
        if np.max(np.abs(audio)) > 1.0001:
            raise ValueError(f"waveform exceeds [-1,1]; fix source levels: {record['path']}")
        value = torch.from_numpy(audio.mean(axis=1))[None]
        if record["rate"] != self.sample_rate:
            transform = self.resamplers.setdefault(record["rate"], None)
            if transform is None:
                transform = torchaudio.transforms.Resample(record["rate"], self.sample_rate)
                self.resamplers[record["rate"]] = transform
            value = transform(value)
        offset = round((start - first) * self.sample_rate / record["rate"])
        value = value[:, offset:offset + self.clip_samples]
        if value.shape[-1] != self.clip_samples:
            raise ValueError(f"resampled waveform is too short: {record['path']}")
        return value


def require_disjoint(train, validation):
    if train.paths & validation.paths or train.digests & validation.digests:
        raise ValueError("training and validation WAV files or contents overlap")

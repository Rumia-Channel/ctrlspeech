"""Validated, cached feature batches for Japanese prompt/target training."""

from __future__ import annotations

import json
from pathlib import Path

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, Sampler

from ..frontend import JAPANESE_PHONE_TO_ID
from ..models.embeds import JapaneseLinguisticConditioner


def validate_example(example, *, patch_size=4, audio_channels=64):
    """Reject misaligned or invalid caches before they reach the model."""
    required = {"vae_features", "speaker_embs", "text_inputs", "linguistic_features", "prompt_frames"}
    missing = required.difference(example)
    if missing:
        raise ValueError("feature cache is missing: " + ", ".join(sorted(missing)))
    latents = example["vae_features"]
    phones = example["text_inputs"]
    speaker = example["speaker_embs"]
    if latents.ndim != 2 or latents.shape[1] != audio_channels or not len(latents):
        raise ValueError(f"vae_features must have shape [T, {audio_channels}], T > 0")
    if not latents.is_floating_point() or not torch.isfinite(latents).all():
        raise ValueError("vae_features must be finite floating point values")
    if speaker.shape != (192,) or not speaker.is_floating_point() or not torch.isfinite(speaker).all():
        raise ValueError("speaker_embs must be a finite floating point vector of length 192")
    if phones.ndim != 1 or len(phones) < 3 or phones.dtype != torch.long:
        raise ValueError("text_inputs must be an unpadded int64 prompt/target phone sequence")
    if (phones <= 0).any() or (phones >= len(JAPANESE_PHONE_TO_ID)).any():
        raise ValueError("text_inputs contains padding or out-of-vocabulary phones")
    if (phones == JAPANESE_PHONE_TO_ID["<unk>"]).any() or (phones == JAPANESE_PHONE_TO_ID["unk"]).any():
        raise ValueError("unknown readings must not enter training caches")
    separators = (phones == JAPANESE_PHONE_TO_ID["|"]).nonzero().flatten()
    if len(separators) != 1 or int(separators[0]) in (0, len(phones) - 1):
        raise ValueError("text_inputs must contain one separator between nonempty prompt and target")
    prompt_frames = example["prompt_frames"]
    if not isinstance(prompt_frames, int) or not 0 < prompt_frames < len(latents):
        raise ValueError("prompt_frames must leave nonempty prompt and target audio")
    if prompt_frames % patch_size:
        raise ValueError("prompt_frames must end on an acoustic patch boundary")

    features = example["linguistic_features"]
    missing = JapaneseLinguisticConditioner.REQUIRED_FIELDS.difference(features)
    if missing:
        raise ValueError("linguistic_features is missing: " + ", ".join(sorted(missing)))
    for name in JapaneseLinguisticConditioner.REQUIRED_FIELDS:
        values = features[name]
        if values.shape != phones.shape or not torch.isfinite(values).all():
            raise ValueError(f"linguistic feature {name} must be finite and phone-aligned")
    for name, limit in (("accent_pitch", 2), ("phrase_boundary", 3)):
        values = features[name]
        if values.dtype != torch.long or (values < 0).any() or (values > limit).any():
            raise ValueError(f"{name} must be int64 IDs in 0..{limit}")
    for name in ("accent_nucleus", "phrase_mora_count"):
        if (features[name] < 0).any():
            raise ValueError(f"{name} must be nonnegative")
    if features["valid"].dtype != torch.bool or features["valid"][separators].any():
        raise ValueError("linguistic valid mask must be bool and exclude the separator")

    if "native_text_inputs" in example:
        ids = example["native_text_inputs"]
        if ids.ndim != 1 or not len(ids) or ids.dtype != torch.long or (ids < 0).any() or (ids >= 65536).any():
            raise ValueError("native_text_inputs must be nonempty native LFM2 int64 IDs")
    controls = {"pitch", "loudness", "duration_segments"}
    present = controls.intersection(example)
    if present and present != controls:
        raise ValueError("pitch, loudness and duration_segments must be provided together")
    if present:
        for name, bins in (("pitch", 128), ("loudness", 64)):
            values = example[name]
            if values.ndim != 1 or not len(values) or values.dtype != torch.long or (values < 0).any() or (values >= bins).any():
                raise ValueError(f"{name} must be an int64 control timeline in 0..{bins - 1}")
        segments = example["duration_segments"]
        if segments.dtype != torch.long or segments.shape != (len(phones), 2):
            raise ValueError("duration_segments must have shape [phone_count, 2] and int64 dtype")
        starts, ends = segments.unbind(dim=1)
        if (starts < 0).any() or (ends < starts).any() or (ends > min(len(example["pitch"]), len(example["loudness"]))).any():
            raise ValueError("duration_segments lie outside the control timelines")
        if segments[separators].any():
            raise ValueError("the separator duration segment must be (0, 0)")
    return example


class CachedSpeechDataset(Dataset):
    """JSONL manifest containing path, frames and speaker_id per cache file."""

    def __init__(self, manifest, *, patch_size=4, audio_channels=64):
        self.manifest = Path(manifest).resolve()
        self.patch_size = patch_size
        self.audio_channels = audio_channels
        self.records = []
        with self.manifest.open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                record = json.loads(line)
                if not isinstance(record.get("frames"), int) or record["frames"] < 1:
                    raise ValueError(f"{self.manifest}:{line_number}: frames must be positive")
                if not isinstance(record.get("speaker_id"), str) or not record["speaker_id"]:
                    raise ValueError(f"{self.manifest}:{line_number}: speaker_id is required")
                path = (self.manifest.parent / record["path"]).resolve()
                if not path.is_file():
                    raise FileNotFoundError(path)
                self.records.append({**record, "path": path})
        if not self.records:
            raise ValueError("training manifest is empty")

    @property
    def lengths(self):
        return [record["frames"] for record in self.records]

    @property
    def speakers(self):
        return {record["speaker_id"] for record in self.records}

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        example = torch.load(record["path"], map_location="cpu", weights_only=True)
        validate_example(example, patch_size=self.patch_size, audio_channels=self.audio_channels)
        if len(example["vae_features"]) != record["frames"]:
            raise ValueError(f"manifest frames disagree with cache: {record['path']}")
        return example


def collate_speech_examples(examples):
    """Keep reference audio as context, supervise only target latent frames."""
    if not examples:
        raise ValueError("examples must not be empty")
    def pad(name):
        return pad_sequence([example[name] for example in examples], batch_first=True)

    lengths = torch.tensor([len(example["vae_features"]) for example in examples])
    latents = pad("vae_features")
    positions = torch.arange(latents.shape[1])[None, :]
    prompt_lengths = torch.tensor([example["prompt_frames"] for example in examples])
    text = pad("text_inputs")
    batch = {
        "vae_features": latents,
        "vae_lengths": lengths,
        "audio_loss_mask": (positions >= prompt_lengths[:, None]) & (positions < lengths[:, None]),
        "speaker_embs": torch.stack([example["speaker_embs"] for example in examples]),
        "text_inputs": text,
        "text_masks": text.ne(0),
        "linguistic_features": {
            name: pad_sequence([example["linguistic_features"][name] for example in examples], batch_first=True)
            for name in JapaneseLinguisticConditioner.REQUIRED_FIELDS
        },
    }
    for names in (("native_text_inputs",), ("pitch", "loudness", "duration_segments")):
        present = [all(name in example for name in names) for example in examples]
        if any(present) and not all(present):
            raise ValueError(f"all batch examples must agree on optional fields {names}")
        if not all(present):
            continue
        if names == ("native_text_inputs",):
            batch["native_text_inputs"] = pad("native_text_inputs")
            native_lengths = torch.tensor([len(example["native_text_inputs"]) for example in examples])
            batch["native_text_masks"] = torch.arange(batch["native_text_inputs"].shape[1])[None, :] < native_lengths[:, None]
        else:
            # These timelines have their own 100 Hz clock; keep them unpadded.
            for name in names:
                batch[name] = [example[name] for example in examples]
    return batch


class LengthBucketBatchSampler(Sampler):
    """Shuffle nearby lengths; exclude overlong examples without truncating phones."""

    def __init__(self, lengths, batch_size, *, max_frames, seed=0, epoch=0, shuffle=True):
        if batch_size < 1 or max_frames < 1:
            raise ValueError("batch_size and max_frames must be positive")
        self.lengths = lengths
        self.batch_size = batch_size
        self.max_frames = max_frames
        self.seed = seed
        self.epoch = epoch
        self.shuffle = shuffle
        self.indices = [i for i, length in enumerate(lengths) if length <= max_frames]
        if not self.indices:
            raise ValueError(f"no examples fit the curriculum limit of {max_frames} latent frames")

    def __len__(self):
        return (len(self.indices) + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        indices = self.indices.copy()
        if self.shuffle:
            order = torch.randperm(len(indices), generator=generator).tolist()
            indices = [indices[i] for i in order]
        batches = []
        bucket_size = self.batch_size * 32
        for start in range(0, len(indices), bucket_size):
            bucket = sorted(indices[start:start + bucket_size], key=self.lengths.__getitem__)
            batches.extend(bucket[i:i + self.batch_size] for i in range(0, len(bucket), self.batch_size))
        if self.shuffle:
            order = torch.randperm(len(batches), generator=generator).tolist()
            batches = [batches[i] for i in order]
        yield from batches

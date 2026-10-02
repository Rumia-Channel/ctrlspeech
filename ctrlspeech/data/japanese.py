"""Model-ready Japanese text preprocessing for CtrlSpeech-JA."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch

from ..frontend import (
    JAPANESE_PHONE_TO_ID,
    JapaneseFrontend,
    JapaneseFrontendResult,
    join_linguistic_features,
    validate_japanese_phones,
)


@dataclass(frozen=True)
class JapaneseTextEncoding:
    """One Japanese utterance after linguistic analysis."""

    result: JapaneseFrontendResult
    phone_ids: tuple[int, ...]

    @property
    def phones(self) -> tuple[str, ...]:
        return self.result.phones

    @property
    def linguistic_features(self) -> dict[str, tuple]:
        return self.result.model_linguistic_features()


@dataclass(frozen=True)
class JapaneseSequenceEncoding:
    """Prompt + separator + target sequence consumed by the AR backbone."""

    prompt: JapaneseTextEncoding
    target: JapaneseTextEncoding
    phone_ids: tuple[int, ...]
    linguistic_features: dict[str, tuple]

    def __post_init__(self):
        length = len(self.phone_ids)
        for name, values in self.linguistic_features.items():
            if len(values) != length:
                raise ValueError(
                    f"linguistic feature {name!r} has {len(values)} values "
                    f"for {length} phone tokens"
                )


def encode_japanese_text(
    text: str,
    *,
    frontend: JapaneseFrontend | None = None,
    phone_vocab: Mapping[str, int] | None = None,
    run_marine: bool = False,
    allow_unknown: bool = False,
) -> JapaneseTextEncoding:
    """Analyze text and map canonical OpenJTalk phones to model IDs."""

    frontend = frontend or JapaneseFrontend()
    vocab = dict(phone_vocab or JAPANESE_PHONE_TO_ID)
    result = frontend.analyze(text, run_marine=run_marine)

    unknown_surfaces = [
        morpheme.surface
        for morpheme in result.morphemes
        if morpheme.is_unknown
    ]
    if unknown_surfaces and not allow_unknown:
        raise ValueError(
            "pyopenjtalk-plus reported unknown Japanese token(s): "
            + ", ".join(unknown_surfaces)
        )

    phones = validate_japanese_phones(result.phones, allow_separator=False)
    if "unk" in phones and not allow_unknown:
        raise ValueError(
            "pyopenjtalk-plus emitted an unknown phone; fix the reading or "
            "explicitly set allow_unknown=True for diagnostics"
        )
    missing = sorted({phone for phone in phones if phone not in vocab})
    if missing:
        raise ValueError(
            "Japanese phone vocabulary is missing: " + ", ".join(missing)
        )

    return JapaneseTextEncoding(
        result=result,
        phone_ids=tuple(vocab[phone] for phone in phones),
    )


def join_prompt_target(
    prompt: JapaneseTextEncoding,
    target: JapaneseTextEncoding,
    *,
    phone_vocab: Mapping[str, int] | None = None,
) -> JapaneseSequenceEncoding:
    """Join prompt and target with the canonical phone separator token."""

    vocab = dict(phone_vocab or JAPANESE_PHONE_TO_ID)
    if "|" not in vocab:
        raise ValueError("phone vocabulary must contain the '|' separator")

    phone_ids = (
        prompt.phone_ids
        + (int(vocab["|"]),)
        + target.phone_ids
    )
    linguistic = join_linguistic_features(prompt.result, target.result)

    return JapaneseSequenceEncoding(
        prompt=prompt,
        target=target,
        phone_ids=phone_ids,
        linguistic_features=linguistic,
    )


def collate_japanese_sequences(
    sequences: list[JapaneseSequenceEncoding],
    *,
    pad_id: int = 0,
) -> dict[str, torch.Tensor]:
    """Pad model-ready prompt/target sequences into a training batch."""

    if not sequences:
        raise ValueError("sequences must not be empty")

    max_length = max(len(sequence.phone_ids) for sequence in sequences)
    batch = len(sequences)

    input_ids = torch.full(
        (batch, max_length),
        int(pad_id),
        dtype=torch.long,
    )
    text_mask = torch.zeros(
        (batch, max_length),
        dtype=torch.bool,
    )

    feature_tensors = {
        "accent_pitch": torch.zeros((batch, max_length), dtype=torch.long),
        "phrase_boundary": torch.zeros((batch, max_length), dtype=torch.long),
        "accent_nucleus": torch.zeros((batch, max_length), dtype=torch.float32),
        "phrase_mora_count": torch.zeros((batch, max_length), dtype=torch.float32),
        "valid": torch.zeros((batch, max_length), dtype=torch.bool),
    }

    for row, sequence in enumerate(sequences):
        length = len(sequence.phone_ids)
        input_ids[row, :length] = torch.tensor(
            sequence.phone_ids,
            dtype=torch.long,
        )
        text_mask[row, :length] = True

        for name, target in feature_tensors.items():
            values = sequence.linguistic_features[name]
            target[row, :length] = torch.as_tensor(
                values,
                dtype=target.dtype,
            )

    return {
        "input_ids": input_ids,
        "text_mask": text_mask,
        "linguistic_features": feature_tensors,
    }

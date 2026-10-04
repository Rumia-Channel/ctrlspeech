"""Model-ready Japanese text preprocessing for CtrlSpeech-JA."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral
from typing import Mapping

import torch

from ..frontend import (
    JAPANESE_PHONE_TO_ID,
    JapaneseFrontend,
    JapaneseFrontendResult,
    join_linguistic_features,
    validate_japanese_phones,
)


_LINGUISTIC_FIELDS = frozenset({
    "accent_pitch", "phrase_boundary", "accent_nucleus", "phrase_mora_count", "valid",
})


def _validate_ids(values, name: str) -> None:
    if any(isinstance(value, bool) or not isinstance(value, Integral) or value < 0
           for value in values):
        raise ValueError(f"{name} must contain non-negative integer IDs")


def _phone_vocab(phone_vocab: Mapping[str, int] | None) -> dict[str, int]:
    # An explicitly empty mapping is an error, not a request for defaults.
    vocab = dict(JAPANESE_PHONE_TO_ID if phone_vocab is None else phone_vocab)
    _validate_ids(vocab.values(), "phone vocabulary")
    if len(set(vocab.values())) != len(vocab):
        raise ValueError("phone vocabulary IDs must be unique")
    return vocab


@dataclass(frozen=True)
class JapaneseTextEncoding:
    """One Japanese utterance after linguistic analysis."""

    result: JapaneseFrontendResult
    phone_ids: tuple[int, ...]

    def __post_init__(self):
        if len(self.phone_ids) != len(self.result.phones):
            raise ValueError("phone_ids must have one ID per frontend phone")
        if tuple(feature.phone for feature in self.result.phone_features) != self.result.phones:
            raise ValueError("linguistic features must align with the frontend phones")
        _validate_ids(self.phone_ids, "phone_ids")

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
    native_text_ids: tuple[int, ...] = ()

    def __post_init__(self):
        length = len(self.phone_ids)
        _validate_ids(self.phone_ids, "phone_ids")
        _validate_ids(self.native_text_ids, "native_text_ids")
        missing = _LINGUISTIC_FIELDS.difference(self.linguistic_features)
        if missing:
            raise ValueError("linguistic_features is missing: " + ", ".join(sorted(missing)))
        prompt_length = len(self.prompt.phone_ids)
        if (
            length != prompt_length + 1 + len(self.target.phone_ids)
            or self.phone_ids[:prompt_length] != self.prompt.phone_ids
            or self.phone_ids[prompt_length + 1:] != self.target.phone_ids
        ):
            raise ValueError("phone_ids must contain prompt, separator, and target in order")
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
    vocab = _phone_vocab(phone_vocab)
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
    if not any(phone not in {"pau", "sil", "sp"} for phone in phones):
        raise ValueError("text did not contain any pronounceable Japanese phones")
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

    vocab = _phone_vocab(phone_vocab)
    if "|" not in vocab:
        raise ValueError("phone vocabulary must contain the '|' separator")
    for encoding in (prompt, target):
        if any(vocab.get(phone) != phone_id
               for phone, phone_id in zip(encoding.phones, encoding.phone_ids)):
            raise ValueError("prompt and target must use the same phone vocabulary as the separator")

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


def encode_japanese_pair(
    prompt_text: str,
    target_text: str,
    *,
    frontend: JapaneseFrontend | None = None,
    phone_vocab: Mapping[str, int] | None = None,
    native_tokenizer=None,
    run_marine: bool = False,
) -> JapaneseSequenceEncoding:
    """Encode a Japanese prompt/target pair for the dual LFM2 text prefix.

    The phone stream is authoritative for pronunciation and accent control.
    When a native LFM tokenizer is supplied, the normalized raw Japanese pair
    is also tokenized as a semantic prefix so LFM2.5 can retain useful language
    representations from pretraining.
    """
    frontend = frontend or JapaneseFrontend()
    prompt = encode_japanese_text(
        prompt_text,
        frontend=frontend,
        phone_vocab=phone_vocab,
        run_marine=run_marine,
    )
    target = encode_japanese_text(
        target_text,
        frontend=frontend,
        phone_vocab=phone_vocab,
        run_marine=run_marine,
    )
    sequence = join_prompt_target(
        prompt,
        target,
        phone_vocab=phone_vocab,
    )

    if native_tokenizer is None:
        return sequence

    raw_pair = prompt.result.normalized_text + "\n" + target.result.normalized_text
    encoded = native_tokenizer(
        raw_pair,
        add_special_tokens=True,
        return_attention_mask=False,
    )
    native_ids = torch.as_tensor(encoded["input_ids"])
    if native_ids.ndim == 2 and native_ids.shape[0] == 1:
        native_ids = native_ids[0]
    if native_ids.ndim != 1:
        raise ValueError("native tokenizer returned an unexpected batch or token shape")
    native_ids = native_ids.tolist()
    if not native_ids:
        raise ValueError("native tokenizer returned no tokens")
    _validate_ids(native_ids, "native tokenizer input_ids")

    return JapaneseSequenceEncoding(
        prompt=sequence.prompt,
        target=sequence.target,
        phone_ids=sequence.phone_ids,
        linguistic_features=sequence.linguistic_features,
        native_text_ids=tuple(native_ids),
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

    output = {
        # Dataset-friendly names.
        "input_ids": input_ids,
        "text_mask": text_mask,
        # Direct DiTar.forward aliases.
        "text_inputs": input_ids,
        "text_masks": text_mask,
        "linguistic_features": feature_tensors,
    }

    native_max_length = max(len(sequence.native_text_ids) for sequence in sequences)
    if native_max_length:
        native_ids = torch.zeros(
            (batch, native_max_length),
            dtype=torch.long,
        )
        native_mask = torch.zeros(
            (batch, native_max_length),
            dtype=torch.bool,
        )
        for row, sequence in enumerate(sequences):
            length = len(sequence.native_text_ids)
            if not length:
                continue
            native_ids[row, :length] = torch.tensor(
                sequence.native_text_ids,
                dtype=torch.long,
            )
            native_mask[row, :length] = True
        output["native_text_ids"] = native_ids
        output["native_text_mask"] = native_mask
        output["native_text_inputs"] = native_ids
        output["native_text_masks"] = native_mask

    return output

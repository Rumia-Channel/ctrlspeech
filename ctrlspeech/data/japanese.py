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
    native_text_ids: tuple[int, ...] = ()

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
    native_ids = encoded["input_ids"]
    if native_ids and isinstance(native_ids[0], list):
        if len(native_ids) != 1:
            raise ValueError("native tokenizer returned an unexpected batch")
        native_ids = native_ids[0]

    return JapaneseSequenceEncoding(
        prompt=sequence.prompt,
        target=sequence.target,
        phone_ids=sequence.phone_ids,
        linguistic_features=sequence.linguistic_features,
        native_text_ids=tuple(int(token_id) for token_id in native_ids),
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
        "input_ids": input_ids,
        "text_mask": text_mask,
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

    return output

from ctrlspeech.data import (
    collate_japanese_sequences,
    encode_japanese_pair,
    encode_japanese_text,
    join_prompt_target,
)
from ctrlspeech.frontend import JapaneseFrontend


class FakePlus:
    def __init__(self, mapping_by_text):
        self.mapping_by_text = mapping_by_text

    def g2p_mapping_prosody(self, text, run_marine=False, normalize_mode="None"):
        return self.mapping_by_text[text]

    def extract_fullcontext(self, text, run_marine=False):
        return []


def _entry(surface, phones, *, accent=0, mora=1):
    return {
        "surface": surface,
        "read": surface,
        "pron": surface,
        "pos": "名詞",
        "accent_nucleus": accent,
        "mora_count": mora,
        "chain_flag": 0,
        "is_unknown": False,
        "char_span": (0, len(surface)),
        "phonemes": [
            {"kind": "Phoneme", "phoneme": phone, "pitch": "Low"}
            for phone in phones
        ],
    }


def test_japanese_sequence_encoding_and_collation():
    backend = FakePlus(
        {
            "猫": [_entry("猫", ["n", "e", "k", "o"], accent=1, mora=2)],
            "犬": [_entry("犬", ["i", "n", "u"], accent=2, mora=2)],
        }
    )
    frontend = JapaneseFrontend(backend=backend)

    prompt = encode_japanese_text("猫", frontend=frontend)
    target = encode_japanese_text("犬", frontend=frontend)
    sequence = join_prompt_target(prompt, target)

    assert len(sequence.phone_ids) == 8
    assert sequence.linguistic_features["valid"][4] is False

    batch = collate_japanese_sequences([sequence])
    assert batch["input_ids"].shape == (1, 8)
    assert batch["text_mask"].all()
    assert batch["linguistic_features"]["accent_pitch"].shape == (1, 8)


class FakeNativeTokenizer:
    def __call__(
        self,
        text,
        *,
        add_special_tokens=True,
        return_attention_mask=False,
    ):
        assert text == "猫\n犬"
        assert add_special_tokens is True
        assert return_attention_mask is False
        return {"input_ids": [1, 10, 11, 7]}


def test_japanese_pair_includes_native_lfm_text_prefix():
    backend = FakePlus(
        {
            "猫": [_entry("猫", ["n", "e", "k", "o"], accent=1, mora=2)],
            "犬": [_entry("犬", ["i", "n", "u"], accent=2, mora=2)],
        }
    )
    frontend = JapaneseFrontend(backend=backend)

    sequence = encode_japanese_pair(
        "猫",
        "犬",
        frontend=frontend,
        native_tokenizer=FakeNativeTokenizer(),
    )
    assert sequence.native_text_ids == (1, 10, 11, 7)

    batch = collate_japanese_sequences([sequence])
    assert batch["native_text_ids"].tolist() == [[1, 10, 11, 7]]
    assert batch["native_text_mask"].tolist() == [[True, True, True, True]]
    assert batch["native_text_inputs"] is batch["native_text_ids"]
    assert batch["native_text_masks"] is batch["native_text_mask"]
    assert batch["text_inputs"] is batch["input_ids"]
    assert batch["text_masks"] is batch["text_mask"]


def _frontend():
    return JapaneseFrontend(backend=FakePlus({
        "猫": [_entry("猫", ["n", "e", "k", "o"], accent=1, mora=2)],
        "犬": [_entry("犬", ["i", "n", "u"], accent=2, mora=2)],
    }))


def test_empty_vocab_is_not_silently_replaced():
    import pytest

    with pytest.raises(ValueError, match="vocabulary is missing"):
        encode_japanese_text("猫", frontend=_frontend(), phone_vocab={})


def test_vocab_rejects_invalid_and_colliding_ids():
    import pytest

    for vocab in [{"n": -1}, {"n": 1.5}, {"n": True}, {"n": 1, "e": 1}]:
        with pytest.raises(ValueError, match="vocabulary"):
            encode_japanese_text("猫", frontend=_frontend(), phone_vocab=vocab)


def test_join_rejects_mixed_vocabularies():
    import pytest
    from ctrlspeech.frontend import JAPANESE_PHONE_TO_ID

    frontend = _frontend()
    custom = {phone: phone_id + 1 for phone, phone_id in JAPANESE_PHONE_TO_ID.items()}
    prompt = encode_japanese_text("猫", frontend=frontend, phone_vocab=custom)
    target = encode_japanese_text("犬", frontend=frontend)
    with pytest.raises(ValueError, match="same phone vocabulary"):
        join_prompt_target(prompt, target)


def test_pause_only_or_empty_utterance_cannot_enter_training():
    import pytest

    for phones in [[], ["pau"], ["sp", "sil"]]:
        frontend = JapaneseFrontend(backend=FakePlus({"。": [_entry("。", phones)]}))
        with pytest.raises(ValueError, match="pronounceable"):
            encode_japanese_text("。", frontend=frontend)


def test_unknown_reading_is_opt_in_and_has_no_linguistic_conditioning():
    import pytest

    entry = _entry("😺", ["unk"])
    entry["is_unknown"] = True
    frontend = JapaneseFrontend(backend=FakePlus({"😺": [entry]}))
    with pytest.raises(ValueError, match="unknown Japanese token"):
        encode_japanese_text("😺", frontend=frontend)
    encoding = encode_japanese_text("😺", frontend=frontend, allow_unknown=True)
    assert encoding.phones == ("unk",)
    assert encoding.linguistic_features["valid"] == (False,)
    assert encoding.linguistic_features["accent_pitch"] == (0,)


def test_native_tokenizer_accepts_single_tensor_and_tuple_batches():
    import torch

    for ids in [torch.tensor([[1, 10, 11, 7]]), ((1, 10, 11, 7),), [1, 10, 11, 7]]:
        sequence = encode_japanese_pair(
            "猫", "犬", frontend=_frontend(),
            native_tokenizer=lambda *args, **kwargs: {"input_ids": ids},
        )
        assert sequence.native_text_ids == (1, 10, 11, 7)


def test_native_tokenizer_rejects_malformed_ids():
    import pytest

    for ids in [[], [[1], [2]], [[[1]]], [1, -1], [1.5], [True], 3]:
        with pytest.raises(ValueError, match="native tokenizer"):
            encode_japanese_pair(
                "猫", "犬", frontend=_frontend(),
                native_tokenizer=lambda *args, **kwargs: {"input_ids": ids},
            )


def test_sequence_rejects_missing_linguistic_fields_and_misalignment():
    import pytest
    from dataclasses import replace

    sequence = encode_japanese_pair("猫", "犬", frontend=_frontend())
    with pytest.raises(ValueError, match="missing: valid"):
        replace(sequence, linguistic_features={k: v for k, v in sequence.linguistic_features.items() if k != "valid"})
    with pytest.raises(ValueError, match="in order"):
        replace(sequence, phone_ids=sequence.phone_ids[:-1])
    with pytest.raises(ValueError, match="align"):
        replace(sequence.prompt, result=replace(sequence.prompt.result, phone_features=()))


def test_variable_length_collation_masks_only_padding():
    sequence = encode_japanese_pair("猫", "犬", frontend=_frontend())
    shorter = encode_japanese_pair("犬", "犬", frontend=_frontend())
    batch = collate_japanese_sequences([sequence, shorter])
    assert batch["text_mask"].tolist() == [[True] * 8, [True] * 7 + [False]]
    assert not batch["linguistic_features"]["valid"][0, 4]
    assert not batch["linguistic_features"]["valid"][1, 3]
    assert not batch["linguistic_features"]["valid"][1, 7]

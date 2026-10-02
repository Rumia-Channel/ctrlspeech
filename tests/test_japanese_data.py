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

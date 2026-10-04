from ctrlspeech.frontend import JapaneseFrontend


class FakePlus:
    def g2p_mapping_prosody(self, text, run_marine=False, normalize_mode="None"):
        assert text == "今日は、晴れ。"
        assert run_marine is False
        assert normalize_mode == "None"
        return [
            {
                "surface": "今日",
                "read": "キョウ",
                "pron": "キョー",
                "pos": "名詞",
                "accent_nucleus": 1,
                "mora_count": 2,
                "chain_flag": -1,
                "is_unknown": False,
                "char_span": (0, 2),
                "phonemes": [
                    {"kind": "Phoneme", "phoneme": "ky", "pitch": "High"},
                    {"kind": "Phoneme", "phoneme": "o", "pitch": "High"},
                    {"kind": "Phoneme", "phoneme": "o", "pitch": "Low"},
                ],
            },
            {
                "surface": "は",
                "read": "ハ",
                "pron": "ワ",
                "pos": "助詞",
                "accent_nucleus": 0,
                "mora_count": 1,
                "chain_flag": 1,
                "is_unknown": False,
                "char_span": (2, 3),
                "phonemes": [
                    {"kind": "Phoneme", "phoneme": "w", "pitch": "Low"},
                    {"kind": "Phoneme", "phoneme": "a", "pitch": "Low"},
                ],
            },
            {
                "surface": "、",
                "read": "、",
                "pron": "、",
                "pos": "記号",
                "accent_nucleus": 0,
                "mora_count": 0,
                "chain_flag": 0,
                "is_unknown": False,
                "char_span": (3, 4),
                "phonemes": [{"kind": "Pause"}],
            },
            {
                "surface": "晴れ",
                "read": "ハレ",
                "pron": "ハレ",
                "pos": "名詞",
                "accent_nucleus": 1,
                "mora_count": 2,
                "chain_flag": 0,
                "is_unknown": False,
                "char_span": (4, 6),
                "phonemes": [
                    {"kind": "Phoneme", "phoneme": "h", "pitch": "High"},
                    {"kind": "Phoneme", "phoneme": "a", "pitch": "High"},
                    {"kind": "Phoneme", "phoneme": "r", "pitch": "Low"},
                    {"kind": "Phoneme", "phoneme": "e", "pitch": "Low"},
                ],
            },
            {
                "surface": "。",
                "read": "。",
                "pron": "。",
                "pos": "記号",
                "accent_nucleus": 0,
                "mora_count": 0,
                "chain_flag": 0,
                "is_unknown": False,
                "char_span": (6, 7),
                "phonemes": [{"kind": "Pause"}],
            },
        ]

    def extract_fullcontext(self, text, run_marine=False):
        return ["label"]


def test_plus_prosody_mapping_is_phone_aligned():
    result = JapaneseFrontend(backend=FakePlus()).analyze("今日は、晴れ。")

    assert result.phones == (
        "ky", "o", "o", "w", "a", "pau", "h", "a", "r", "e", "pau"
    )
    assert len(result.phone_features) == len(result.phones)
    assert result.mfa_transcript == "今日 は 晴れ"

    first = result.phone_features[0]
    assert first.pitch == "High"
    assert first.is_accent_phrase_start is True
    assert first.accent_nucleus == 1
    assert first.phrase_mora_count == 3

    before_pause = result.phone_features[4]
    assert before_pause.is_accent_phrase_end is True
    pause = result.phone_features[5]
    assert pause.phone == "pau"
    assert pause.pause_kind == "Pause"
    assert pause.is_pause is True

    second_phrase = result.phone_features[6]
    assert second_phrase.is_accent_phrase_start is True
    assert second_phrase.accent_nucleus == 1
    assert second_phrase.phrase_mora_count == 2


class MappingOnlyPlus:
    """The public API supplied by the released pyopenjtalk-plus 0.4.1.post9."""

    def __init__(self, phones=None):
        self.phones = phones or ["t", "e", "s", "U", "t", "o"]

    def g2p_mapping(self, text, *, run_marine=False, normalize_mode="None"):
        assert text == "テスト"
        assert normalize_mode == "None"
        return [{
            "surface": text, "phonemes": self.phones, "pos": "名詞",
            "accent_nucleus": 1, "mora_count": 3, "char_span": (0, 3),
            "is_unknown": False,
        }]

    def extract_fullcontext(self, text, *, run_marine=False):
        return [
            _label("sil"),
            *[_label(phone, position=position) for phone, position in
              zip(["t", "e", "s", "U", "t", "o"], [1, 1, 2, 2, 3, 3])],
            _label("sil"),
        ]


def _label(phone, *, position=1, nucleus=1, mora_count=3, phrase=1):
    # F identifies the accent phrase; A gives mora position relative to its
    # nucleus. Context outside these fields is immaterial to this parser.
    return (
        f"xx^xx-{phone}+xx=xx/A:{position - nucleus}+{position}+{mora_count-position+1}"
        f"/F:{mora_count}_{nucleus}#0_0@{phrase}_1|1_3/G:xx"
        "/I:1-3@1+1&1-1|1+3/J:xx"
    )


def test_released_mapping_api_retains_prosody_without_retaining_labels():
    result = JapaneseFrontend(backend=MappingOnlyPlus()).analyze(
        "テスト", include_fullcontext=False,
    )
    assert result.phones == ("t", "e", "s", "U", "t", "o")
    assert [f.pitch for f in result.phone_features] == ["High", "High", "Low", "Low", "Low", "Low"]
    assert result.morphemes[0].char_span == (0, 3)
    assert result.phone_features[0].accent_nucleus == 1
    assert result.phone_features[0].phrase_mora_count == 3
    assert result.phone_features[0].is_accent_phrase_start
    assert result.phone_features[-1].is_accent_phrase_end
    assert result.fullcontext_labels == ()


def test_released_mapping_api_rejects_misaligned_fullcontext_phones():
    import pytest

    with pytest.raises(RuntimeError, match="mapping and full-context phones do not match"):
        JapaneseFrontend(backend=MappingOnlyPlus(["k", "e"])).analyze("テスト")


def test_released_mapping_api_preserves_devoiced_vowel_identity():
    result = JapaneseFrontend(backend=MappingOnlyPlus(["t", "e", "s", "u", "t", "o"])).analyze("テスト")
    assert result.phones[3] == "U"


def test_repeated_and_literal_pauses_do_not_create_empty_accent_phrases():
    class Backend:
        def g2p_mapping_prosody(self, *args, **kwargs):
            return [{
                "surface": "、あ、。い", "mora_count": 2,
                "phonemes": [
                    {"kind": "Pause"},
                    {"kind": "AccentPhraseBoundary"},
                    {"kind": "Phoneme", "phoneme": "a", "pitch": "Low"},
                    {"kind": "Phoneme", "phoneme": "pau"},
                    {"kind": "AccentPhraseBoundary"},
                    {"kind": "Pause"},
                    {"kind": "Phoneme", "phoneme": "i", "pitch": "High"},
                ],
            }]

    result = JapaneseFrontend(backend=Backend()).analyze("、あ、。い", include_fullcontext=False)
    assert result.accent_phrase_count == 2
    speech = [f for f in result.phone_features if not f.is_pause]
    assert [f.accent_phrase_index for f in speech] == [0, 1]
    assert all(f.is_accent_phrase_start and f.is_accent_phrase_end for f in speech)
    assert all(f.accent_nucleus == f.phrase_mora_count == 0
               for f in result.phone_features if f.is_pause)


def test_phone_helper_uses_the_conditioned_phone_stream():
    frontend = JapaneseFrontend(backend=FakePlus())
    assert frontend.phones("今日は、晴れ。") == frontend.analyze("今日は、晴れ。").phones

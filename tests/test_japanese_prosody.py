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

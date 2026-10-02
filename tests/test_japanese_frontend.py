import pytest

from ctrlspeech.frontend import JapaneseFrontend, validate_japanese_phones


class FakeOpenJTalk:
    def run_frontend(self, text):
        assert text == "今日は、いい天気。"
        return [
            {
                "string": "今日",
                "read": "キョウ",
                "pron": "キョー",
                "pos": "名詞",
                "acc": 1,
                "mora_size": 2,
                "chain_flag": -1,
            },
            {
                "string": "は",
                "read": "ハ",
                "pron": "ワ",
                "pos": "助詞",
                "acc": 0,
                "mora_size": 1,
                "chain_flag": 1,
            },
            {
                "string": "、",
                "read": "、",
                "pron": "、",
                "pos": "記号",
                "acc": 0,
                "mora_size": 0,
                "chain_flag": 0,
            },
            {
                "string": "いい",
                "read": "イイ",
                "pron": "イイ",
                "pos": "形容詞",
                "acc": 1,
                "mora_size": 2,
                "chain_flag": 0,
            },
            {
                "string": "天気",
                "read": "テンキ",
                "pron": "テンキ",
                "pos": "名詞",
                "acc": 1,
                "mora_size": 3,
                "chain_flag": 1,
            },
            {
                "string": "。",
                "read": "。",
                "pron": "。",
                "pos": "記号",
                "acc": 0,
                "mora_size": 0,
                "chain_flag": 0,
            },
        ]

    def g2p(self, text, kana=False, join=False):
        mapping = {
            "今日は、いい天気。": ["ky", "o", "o", "w", "a", "pau", "i", "i", "t", "e", "N", "k", "i"],
            "キョー": ["ky", "o", "o"],
            "ワ": ["w", "a"],
            "イイ": ["i", "i"],
            "テンキ": ["t", "e", "N", "k", "i"],
        }
        if text in {"、", "。"}:
            raise ValueError("symbol")
        result = mapping[text]
        return result if not join else " ".join(result)

    def extract_fullcontext(self, text, run_marine=False):
        assert text == "今日は、いい天気。"
        assert run_marine is False
        return ["label-1", "label-2"]


def test_japanese_frontend_keeps_phone_and_accent_metadata():
    frontend = JapaneseFrontend(backend=FakeOpenJTalk())
    result = frontend.analyze("今日は、いい天気。")

    assert result.phone_string.startswith("ky o o w a")
    assert result.mfa_transcript == "今日 は いい 天気"
    assert result.fullcontext_labels == ("label-1", "label-2")
    assert result.morphemes[0].accent == 1
    assert result.morphemes[0].mora_size == 2
    assert result.morphemes[2].is_symbol is True


def test_japanese_frontend_normalizes_fullwidth_ascii():
    frontend = JapaneseFrontend(backend=FakeOpenJTalk())
    assert frontend.normalize("ＡＢＣ  １２３") == "ABC 123"


def test_japanese_phone_inventory_rejects_unknown_tokens():
    assert validate_japanese_phones(["ky", "o", "N", "pau"]) == (
        "ky",
        "o",
        "N",
        "pau",
    )
    with pytest.raises(ValueError, match="Unsupported CtrlSpeech-JA phone"):
        validate_japanese_phones(["not-a-phone"])

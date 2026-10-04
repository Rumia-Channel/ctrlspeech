"""Text-only checks of the actual released backend; no model downloads or TTS."""

from types import SimpleNamespace

import pytest

from ctrlspeech.data import encode_japanese_text
from ctrlspeech.frontend import JapaneseFrontend, validate_japanese_phones


@pytest.fixture(scope="module")
def frontend():
    pyopenjtalk = pytest.importorskip("pyopenjtalk")

    def offline_call(function):
        def call(*args, **kwargs):
            # Exercise the public installed API with neural reading correction
            # disabled so this regression suite needs only bundled dictionaries.
            return function(*args, **kwargs, predict_nani=False, use_sudachi_kanji_yomi=False)
        return call

    names = ["g2p_mapping", "g2p_mapping_prosody", "extract_fullcontext"]
    backend = SimpleNamespace(**{
        name: offline_call(getattr(pyopenjtalk, name))
        for name in names if hasattr(pyopenjtalk, name)
    })
    return JapaneseFrontend(backend=backend)


def test_real_backend_retains_accent_metadata_and_char_spans(frontend):
    result = frontend.analyze("今日は、晴れ。")
    assert result.phones == ("ky", "o", "o", "w", "a", "pau", "h", "a", "r", "e", "pau")
    assert result.accent_phrase_count == 2
    assert [f.pitch for f in result.phone_features[:5]] == ["High", "High", "Low", "Low", "Low"]
    assert result.phone_features[0].accent_nucleus == 1
    assert result.phone_features[0].phrase_mora_count == 3
    assert result.morphemes[0].char_span == (0, 2)
    assert result.morphemes[1].char_span == (2, 3)
    assert result.mfa_transcript == "今日 は 晴れ"


@pytest.mark.parametrize("text", [
    "猫と犬が走る。", "こんにちは。", "テスト", "１２３円です。",
    "「猫」！犬？", "、、、猫。。。犬！", "猫 犬",
])
def test_real_backend_phone_and_feature_invariants(frontend, text):
    result = frontend.analyze(text)
    assert validate_japanese_phones(result.phones) == result.phones
    assert tuple(f.phone for f in result.phone_features) == result.phones
    assert frontend.phones(text) == result.phones
    assert result.fullcontext_labels
    speech = [f for f in result.phone_features if not f.is_pause]
    assert all(f.pitch in {"Low", "High"} for f in speech)
    phrases = sorted({f.accent_phrase_index for f in speech})
    assert phrases == list(range(result.accent_phrase_count))
    for phrase in phrases:
        features = [f for f in speech if f.accent_phrase_index == phrase]
        assert sum(f.is_accent_phrase_start for f in features) == 1
        assert sum(f.is_accent_phrase_end for f in features) == 1
        assert features[0].phrase_mora_count > 0
    assert all(not valid for f, valid in zip(result.phone_features, result.model_linguistic_features()["valid"]) if f.is_pause)


def test_real_backend_splits_phrases_without_punctuation(frontend):
    result = frontend.analyze("猫と犬が走る。")
    assert result.accent_phrase_count == 3
    assert [f.phone for f in result.phone_features if f.is_accent_phrase_start] == ["n", "i", "h"]


def test_real_backend_whitespace_preserves_the_accent_phrase(frontend):
    result = frontend.analyze("猫 犬")
    assert result.phones == ("n", "e", "k", "o", "sp", "i", "n", "u")
    assert result.accent_phrase_count == 1
    assert result.phone_features[0].phrase_mora_count == 4
    assert result.phone_features[-1].phrase_mora_count == 4
    assert not result.phone_features[4].is_accent_phrase_end


def test_real_backend_rejects_unknown_and_unpronounceable_text(frontend):
    with pytest.raises(ValueError, match="unknown"):
        encode_japanese_text("猫😺犬", frontend=frontend)
    with pytest.raises(ValueError, match="pronounceable"):
        encode_japanese_text("、。", frontend=frontend)


def test_real_backend_preserves_question_and_exclamation_boundaries(frontend):
    result = frontend.analyze("猫！犬？")
    assert [f.pause_kind for f in result.phone_features if f.is_pause] == ["Exclamatory", "Interrogative"]


def test_real_backend_preserves_devoiced_vowels(frontend):
    assert frontend.analyze("テスト").phones == ("t", "e", "s", "U", "t", "o")

"""Text frontends for language-specific CtrlSpeech preprocessing."""

from .japanese import (
    JapaneseFrontend,
    JapaneseFrontendResult,
    JapaneseMorpheme,
    JapanesePhoneFeature,
    join_linguistic_features,
)

__all__ = [
    "JapaneseFrontend",
    "JapaneseFrontendResult",
    "JapaneseMorpheme",
    "JapanesePhoneFeature",
    "join_linguistic_features",
]


from .japanese_phones import (
    JAPANESE_PHONE_TO_ID,
    JAPANESE_PHONE_TOKENS,
    JAPANESE_PHONE_VOCAB_SIZE,
    OPENJTALK_PHONES,
    japanese_phone_vocab,
    validate_japanese_phones,
)

__all__ += [
    "JAPANESE_PHONE_TO_ID",
    "JAPANESE_PHONE_TOKENS",
    "JAPANESE_PHONE_VOCAB_SIZE",
    "OPENJTALK_PHONES",
    "japanese_phone_vocab",
    "validate_japanese_phones",
]

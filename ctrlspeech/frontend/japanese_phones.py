"""Canonical phone inventory for CtrlSpeech-JA.

The model vocabulary follows OpenJTalk/HTS phones.  It intentionally keeps
devoiced vowels (uppercase) distinct because pyopenjtalk can emit them in
full-context phonetic representations.
"""

from __future__ import annotations

from typing import Final, Iterable


SPECIAL_PHONE_TOKENS: Final[tuple[str, ...]] = (
    "<pad>",
    "<unk>",
    "|",
)

OPENJTALK_PHONES: Final[tuple[str, ...]] = (
    # vowels
    "a", "i", "u", "e", "o",
    "A", "I", "U", "E", "O",
    # consonants / contracted consonants
    "b", "by",
    "ch",
    "d", "dy",
    "f",
    "g", "gy", "gw",
    "h", "hy",
    "j",
    "k", "ky", "kw",
    "m", "my",
    "n", "ny",
    "p", "py",
    "r", "ry",
    "s", "sh",
    "t", "ts", "ty",
    "v",
    "w",
    "y",
    "z", "zy",
    # Japanese special phones / boundaries
    "N",
    "cl",
    "pau",
    "sil",
    "sp",
    "unk",
)

JAPANESE_PHONE_TOKENS: Final[tuple[str, ...]] = (
    SPECIAL_PHONE_TOKENS + OPENJTALK_PHONES
)
JAPANESE_PHONE_TO_ID: Final[dict[str, int]] = {
    token: index for index, token in enumerate(JAPANESE_PHONE_TOKENS)
}
JAPANESE_PHONE_VOCAB_SIZE: Final[int] = len(JAPANESE_PHONE_TOKENS)


def validate_japanese_phones(
    phones: Iterable[str],
    *,
    allow_separator: bool = True,
) -> tuple[str, ...]:
    """Return phones as a tuple or raise on an unknown model token."""
    values = tuple(str(phone) for phone in phones)
    allowed = set(OPENJTALK_PHONES)
    if allow_separator:
        allowed.add("|")
    unknown = sorted({phone for phone in values if phone not in allowed})
    if unknown:
        raise ValueError(
            "Unsupported CtrlSpeech-JA phone(s): " + ", ".join(unknown)
        )
    return values


def japanese_phone_vocab() -> dict[str, int]:
    """Return a mutable copy suitable for JSON serialization."""
    return dict(JAPANESE_PHONE_TO_ID)

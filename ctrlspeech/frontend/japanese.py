"""Japanese text frontend built on pyopenjtalk.

This module deliberately stops at linguistic preprocessing. The published
CtrlSpeech checkpoints use an English-oriented phone vocabulary, so feeding
these Japanese phones into those checkpoints would collapse most symbols to
unknown tokens. A Japanese-trained tokenizer/checkpoint is required before
this frontend can be wired into inference.

The frontend exposes two views of the same sentence:

* OpenJTalk phones and NJD accent metadata for model/data preprocessing.
* A whitespace-segmented orthographic transcript suitable for the Japanese MFA
  dictionary/acoustic model.
"""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass
from typing import Any


_SYMBOL_POS = {"記号", "補助記号"}


def _as_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


@dataclass(frozen=True)
class JapaneseMorpheme:
    """One NJD/OpenJTalk frontend node."""

    surface: str
    reading: str
    pronunciation: str
    part_of_speech: str
    accent: int
    mora_size: int
    chain_flag: int
    phones: tuple[str, ...]
    is_symbol: bool = False


@dataclass(frozen=True)
class JapaneseFrontendResult:
    """Structured linguistic representation of one Japanese utterance."""

    text: str
    normalized_text: str
    phones: tuple[str, ...]
    morphemes: tuple[JapaneseMorpheme, ...]
    mfa_transcript: str
    fullcontext_labels: tuple[str, ...] = ()

    @property
    def phone_string(self) -> str:
        """Return model-friendly space-separated OpenJTalk phones."""
        return " ".join(self.phones)


class JapaneseFrontend:
    """Japanese normalization, G2P, NJD metadata and MFA tokenization.

    backend is injectable for tests. In production it defaults to the
    pyopenjtalk module and is loaded lazily so English-only installations do
    not acquire a hard dependency on OpenJTalk.
    """

    def __init__(self, backend=None, normalization: str = "NFKC"):
        self._backend = backend
        self.normalization = normalization

    @property
    def backend(self):
        if self._backend is None:
            try:
                import pyopenjtalk
            except ImportError as exc:
                raise RuntimeError(
                    "Japanese preprocessing needs pyopenjtalk. Install it with "
                    "pip install -e '.[japanese]'."
                ) from exc
            self._backend = pyopenjtalk
        return self._backend

    def normalize(self, text: str) -> str:
        """Unicode-normalize and collapse whitespace without changing wording."""
        if not isinstance(text, str):
            raise TypeError("text must be a str")
        text = unicodedata.normalize(self.normalization, text)
        return " ".join(text.split())

    def analyze(
        self,
        text: str,
        *,
        include_fullcontext: bool = True,
        run_marine: bool = False,
    ) -> JapaneseFrontendResult:
        """Analyze Japanese text with OpenJTalk.

        Accent-related values are the raw NJD fields (acc, mora_size and
        chain_flag). They are intentionally preserved rather than converted to
        a custom accent-phrase scheme until the training representation is
        fixed.
        """
        normalized = self.normalize(text)
        if not normalized:
            raise ValueError("text must not be empty")

        nodes = self.backend.run_frontend(normalized)
        morphemes = []
        mfa_words = []

        for node in nodes:
            surface = str(node.get("string") or "")
            reading = str(node.get("read") or "")
            pronunciation = str(node.get("pron") or "")
            pos = str(node.get("pos") or "")
            is_symbol = pos in _SYMBOL_POS

            phone_source = pronunciation
            if not phone_source or phone_source == "*":
                phone_source = reading
            if not phone_source or phone_source == "*":
                phone_source = surface

            node_phones: tuple[str, ...] = ()
            if phone_source:
                try:
                    node_phones = tuple(
                        self.backend.g2p(phone_source, kana=False, join=False)
                    )
                except Exception:
                    # Sentence-level G2P below remains authoritative. Some
                    # symbol-only nodes are not valid standalone G2P inputs.
                    node_phones = ()

            morphemes.append(
                JapaneseMorpheme(
                    surface=surface,
                    reading=reading,
                    pronunciation=pronunciation,
                    part_of_speech=pos,
                    accent=_as_int(node.get("acc")),
                    mora_size=_as_int(node.get("mora_size")),
                    chain_flag=_as_int(node.get("chain_flag")),
                    phones=node_phones,
                    is_symbol=is_symbol,
                )
            )
            if surface and not is_symbol:
                mfa_words.append(surface)

        phones = tuple(self.backend.g2p(normalized, kana=False, join=False))
        labels: tuple[str, ...] = ()
        if include_fullcontext:
            labels = tuple(
                self.backend.extract_fullcontext(normalized, run_marine=run_marine)
            )

        return JapaneseFrontendResult(
            text=text,
            normalized_text=normalized,
            phones=phones,
            morphemes=tuple(morphemes),
            mfa_transcript=" ".join(mfa_words),
            fullcontext_labels=labels,
        )

    def phones(self, text: str) -> tuple[str, ...]:
        """Return only the OpenJTalk phone sequence."""
        normalized = self.normalize(text)
        if not normalized:
            raise ValueError("text must not be empty")
        return tuple(self.backend.g2p(normalized, kana=False, join=False))

    def to_mfa_transcript(self, text: str) -> str:
        """Segment Japanese orthography into the word sequence MFA expects."""
        result = self.analyze(text, include_fullcontext=False)
        if not result.mfa_transcript:
            raise ValueError("text did not contain any alignable Japanese words")
        return result.mfa_transcript

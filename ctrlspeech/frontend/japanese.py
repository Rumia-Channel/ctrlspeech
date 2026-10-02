"""Japanese text frontend built on pyopenjtalk-plus.

CtrlSpeech-JA prefers pyopenjtalk-plus structured prosody mapping so phone
identity, phrase boundaries and OpenJTalk Low/High accent trajectories stay
aligned. A compatibility path remains for injected test backends that only
implement the classic pyopenjtalk API.
"""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass, replace
from typing import Any, Literal


_SYMBOL_POS = {"記号", "補助記号"}
_PAUSE_MARKERS = {"Pause", "Interrogative", "Exclamatory"}
PitchLevel = Literal["Low", "High"]


def _as_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


@dataclass(frozen=True)
class JapaneseMorpheme:
    """One pyopenjtalk-plus surface/phoneme mapping entry."""

    surface: str
    reading: str
    pronunciation: str
    part_of_speech: str
    accent: int
    mora_size: int
    chain_flag: int
    phones: tuple[str, ...]
    is_symbol: bool = False
    is_unknown: bool = False
    char_span: tuple[int, int] = (0, 0)


@dataclass(frozen=True)
class JapanesePhoneFeature:
    """Phone-aligned Japanese linguistic/prosodic information.

    pitch is OpenJTalk categorical accent trajectory, not measured F0.
    accent_nucleus and phrase_mora_count are NJD phrase metadata broadcast
    onto each phone. pitch and phrase-boundary flags come from full-context
    labels through pyopenjtalk-plus and are authoritative.
    """

    phone: str
    morpheme_index: int
    accent_phrase_index: int
    pitch: PitchLevel | None = None
    accent_nucleus: int = 0
    phrase_mora_count: int = 0
    is_accent_phrase_start: bool = False
    is_accent_phrase_end: bool = False
    is_pause: bool = False
    pause_kind: str | None = None


@dataclass(frozen=True)
class JapaneseFrontendResult:
    """Structured linguistic representation of one Japanese utterance."""

    text: str
    normalized_text: str
    phones: tuple[str, ...]
    morphemes: tuple[JapaneseMorpheme, ...]
    phone_features: tuple[JapanesePhoneFeature, ...]
    mfa_transcript: str
    fullcontext_labels: tuple[str, ...] = ()

    @property
    def phone_string(self) -> str:
        return " ".join(self.phones)

    @property
    def accent_phrase_count(self) -> int:
        phrase_ids = [
            feature.accent_phrase_index
            for feature in self.phone_features
            if not feature.is_pause
        ]
        return max(phrase_ids, default=-1) + 1


    def model_linguistic_features(self) -> dict[str, tuple]:
        """Return phone-aligned values consumed by DiTar.

        accent_pitch IDs are 0=unknown, 1=Low, 2=High. phrase_boundary uses
        bit0=start and bit1=end. valid distinguishes an actual linguistic phone
        from pauses/separators, so accent nucleus 0 remains a meaningful flat
        accent value rather than an absence sentinel.
        """
        pitch_id = {None: 0, "Low": 1, "High": 2}
        return {
            "accent_pitch": tuple(
                pitch_id[feature.pitch] for feature in self.phone_features
            ),
            "phrase_boundary": tuple(
                int(feature.is_accent_phrase_start)
                | (int(feature.is_accent_phrase_end) << 1)
                for feature in self.phone_features
            ),
            "accent_nucleus": tuple(
                feature.accent_nucleus for feature in self.phone_features
            ),
            "phrase_mora_count": tuple(
                feature.phrase_mora_count for feature in self.phone_features
            ),
            "valid": tuple(
                not feature.is_pause for feature in self.phone_features
            ),
        }


class JapaneseFrontend:
    """Japanese normalization, G2P and phone-aligned accent analysis."""

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
                    "Japanese preprocessing needs pyopenjtalk-plus. "
                    "Run 'uv sync --extra japanese'."
                ) from exc
            self._backend = pyopenjtalk
        return self._backend

    def normalize(self, text: str) -> str:
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
        normalized = self.normalize(text)
        if not normalized:
            raise ValueError("text must not be empty")

        if hasattr(self.backend, "g2p_mapping_prosody"):
            return self._analyze_plus(
                text=text,
                normalized=normalized,
                include_fullcontext=include_fullcontext,
                run_marine=run_marine,
            )
        return self._analyze_legacy(
            text=text,
            normalized=normalized,
            include_fullcontext=include_fullcontext,
            run_marine=run_marine,
        )

    def _analyze_plus(
        self,
        *,
        text: str,
        normalized: str,
        include_fullcontext: bool,
        run_marine: bool,
    ) -> JapaneseFrontendResult:
        mapping = self.backend.g2p_mapping_prosody(
            normalized,
            run_marine=run_marine,
            normalize_mode="None",
        )

        morphemes: list[JapaneseMorpheme] = []
        mfa_words: list[str] = []
        raw_features: list[JapanesePhoneFeature] = []
        phrase_index = 0
        next_phrase_start = True
        last_speech_feature: int | None = None

        for morpheme_index, entry in enumerate(mapping):
            surface = str(entry.get("surface") or "")
            pos = str(entry.get("pos") or "")
            is_symbol = pos in _SYMBOL_POS
            entry_phones: list[str] = []

            for item in entry.get("phonemes", ()):
                kind = str(item.get("kind") or "")
                if kind == "Phoneme":
                    phone = str(item.get("phoneme") or "")
                    if not phone:
                        continue
                    entry_phones.append(phone)
                    is_pause = phone in {"pau", "sil", "sp"}
                    raw_features.append(
                        JapanesePhoneFeature(
                            phone=phone,
                            morpheme_index=morpheme_index,
                            accent_phrase_index=phrase_index,
                            pitch=item.get("pitch"),
                            is_accent_phrase_start=(
                                next_phrase_start and not is_pause
                            ),
                            is_pause=is_pause,
                        )
                    )
                    if not is_pause:
                        last_speech_feature = len(raw_features) - 1
                        next_phrase_start = False
                    continue

                if kind == "AccentPhraseBoundary":
                    if last_speech_feature is not None:
                        raw_features[last_speech_feature] = replace(
                            raw_features[last_speech_feature],
                            is_accent_phrase_end=True,
                        )
                    phrase_index += 1
                    next_phrase_start = True
                    last_speech_feature = None
                    continue

                if kind in _PAUSE_MARKERS:
                    if last_speech_feature is not None:
                        raw_features[last_speech_feature] = replace(
                            raw_features[last_speech_feature],
                            is_accent_phrase_end=True,
                        )
                    entry_phones.append("pau")
                    raw_features.append(
                        JapanesePhoneFeature(
                            phone="pau",
                            morpheme_index=morpheme_index,
                            accent_phrase_index=phrase_index,
                            is_accent_phrase_end=True,
                            is_pause=True,
                            pause_kind=kind,
                        )
                    )
                    phrase_index += 1
                    next_phrase_start = True
                    last_speech_feature = None

            morphemes.append(
                JapaneseMorpheme(
                    surface=surface,
                    reading=str(entry.get("read") or ""),
                    pronunciation=str(entry.get("pron") or ""),
                    part_of_speech=pos,
                    accent=_as_int(entry.get("accent_nucleus")),
                    mora_size=_as_int(entry.get("mora_count")),
                    chain_flag=_as_int(entry.get("chain_flag")),
                    phones=tuple(entry_phones),
                    is_symbol=is_symbol,
                    is_unknown=bool(entry.get("is_unknown", False)),
                    char_span=tuple(entry.get("char_span") or (0, 0)),
                )
            )
            if surface and not is_symbol:
                mfa_words.append(surface)

        if last_speech_feature is not None:
            raw_features[last_speech_feature] = replace(
                raw_features[last_speech_feature],
                is_accent_phrase_end=True,
            )

        phone_features = self._broadcast_phrase_metadata(raw_features, morphemes)
        phones = tuple(feature.phone for feature in phone_features)

        labels: tuple[str, ...] = ()
        if include_fullcontext:
            labels = tuple(
                self.backend.extract_fullcontext(
                    normalized,
                    run_marine=run_marine,
                )
            )

        return JapaneseFrontendResult(
            text=text,
            normalized_text=normalized,
            phones=phones,
            morphemes=tuple(morphemes),
            phone_features=tuple(phone_features),
            mfa_transcript=" ".join(mfa_words),
            fullcontext_labels=labels,
        )

    @staticmethod
    def _broadcast_phrase_metadata(
        features: list[JapanesePhoneFeature],
        morphemes: list[JapaneseMorpheme],
    ) -> list[JapanesePhoneFeature]:
        phrase_to_morphemes: dict[int, list[int]] = {}
        for feature in features:
            if feature.is_pause:
                continue
            indices = phrase_to_morphemes.setdefault(
                feature.accent_phrase_index,
                [],
            )
            if feature.morpheme_index not in indices:
                indices.append(feature.morpheme_index)

        metadata: dict[int, tuple[int, int]] = {}
        for phrase, indices in phrase_to_morphemes.items():
            if not indices:
                metadata[phrase] = (0, 0)
                continue
            first = morphemes[indices[0]]
            mora_count = sum(
                max(0, morphemes[index].mora_size)
                for index in indices
            )
            metadata[phrase] = (max(0, first.accent), mora_count)

        return [
            replace(
                feature,
                accent_nucleus=metadata.get(
                    feature.accent_phrase_index,
                    (0, 0),
                )[0],
                phrase_mora_count=metadata.get(
                    feature.accent_phrase_index,
                    (0, 0),
                )[1],
            )
            for feature in features
        ]

    def _analyze_legacy(
        self,
        *,
        text: str,
        normalized: str,
        include_fullcontext: bool,
        run_marine: bool,
    ) -> JapaneseFrontendResult:
        nodes = self.backend.run_frontend(normalized)
        morphemes: list[JapaneseMorpheme] = []
        mfa_words: list[str] = []

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
                        self.backend.g2p(
                            phone_source,
                            kana=False,
                            join=False,
                        )
                    )
                except Exception:
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

        phones = tuple(
            self.backend.g2p(
                normalized,
                kana=False,
                join=False,
            )
        )
        phone_features = self._legacy_phone_features(phones)

        labels: tuple[str, ...] = ()
        if include_fullcontext:
            labels = tuple(
                self.backend.extract_fullcontext(
                    normalized,
                    run_marine=run_marine,
                )
            )

        return JapaneseFrontendResult(
            text=text,
            normalized_text=normalized,
            phones=phones,
            morphemes=tuple(morphemes),
            phone_features=phone_features,
            mfa_transcript=" ".join(mfa_words),
            fullcontext_labels=labels,
        )

    @staticmethod
    def _legacy_phone_features(
        phones: tuple[str, ...],
    ) -> tuple[JapanesePhoneFeature, ...]:
        features: list[JapanesePhoneFeature] = []
        phrase = 0
        next_start = True
        last_speech: int | None = None
        for phone in phones:
            is_pause = phone in {"pau", "sil", "sp"}
            if is_pause:
                if last_speech is not None:
                    features[last_speech] = replace(
                        features[last_speech],
                        is_accent_phrase_end=True,
                    )
                features.append(
                    JapanesePhoneFeature(
                        phone=phone,
                        morpheme_index=-1,
                        accent_phrase_index=phrase,
                        is_pause=True,
                        is_accent_phrase_end=True,
                    )
                )
                phrase += 1
                next_start = True
                last_speech = None
                continue
            features.append(
                JapanesePhoneFeature(
                    phone=phone,
                    morpheme_index=-1,
                    accent_phrase_index=phrase,
                    is_accent_phrase_start=next_start,
                )
            )
            last_speech = len(features) - 1
            next_start = False

        if last_speech is not None:
            features[last_speech] = replace(
                features[last_speech],
                is_accent_phrase_end=True,
            )
        return tuple(features)

    def phones(self, text: str) -> tuple[str, ...]:
        normalized = self.normalize(text)
        if not normalized:
            raise ValueError("text must not be empty")
        return tuple(
            self.backend.g2p(
                normalized,
                kana=False,
                join=False,
            )
        )

    def to_mfa_transcript(self, text: str) -> str:
        result = self.analyze(text, include_fullcontext=False)
        if not result.mfa_transcript:
            raise ValueError(
                "text did not contain any alignable Japanese words"
            )
        return result.mfa_transcript



def join_linguistic_features(
    prompt: JapaneseFrontendResult,
    target: JapaneseFrontendResult,
) -> dict[str, tuple]:
    """Join prompt and target features around the explicit phone separator."""
    prompt_features = prompt.model_linguistic_features()
    target_features = target.model_linguistic_features()
    separator = {
        "accent_pitch": 0,
        "phrase_boundary": 0,
        "accent_nucleus": 0,
        "phrase_mora_count": 0,
        "valid": False,
    }
    return {
        key: (
            tuple(prompt_features[key])
            + (separator[key],)
            + tuple(target_features[key])
        )
        for key in separator
    }

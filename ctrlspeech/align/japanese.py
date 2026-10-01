"""Japanese Montreal Forced Aligner integration.

The Japanese MFA acoustic model and dictionary use MFA's Japanese phone set,
which is not the same inventory as OpenJTalk. This class therefore exposes
alignment as a standalone preprocessing primitive; callers must not compare its
phones directly with JapaneseFrontend.phones until a canonical phone mapping
has been selected for the Japanese training pipeline.
"""

from __future__ import annotations

import os
from pathlib import Path

from .mfa import MFAAligner
from ..frontend import JapaneseFrontend


DEFAULT_JAPANESE_DICT = Path(
    os.environ.get(
        "CTRLSPEECH_MFA_JA_DICT",
        Path.home()
        / "Documents/MFA/pretrained_models/dictionary/japanese_mfa.dict",
    )
)


class JapaneseMFAAligner(MFAAligner):
    """MFA aligner configured for the official japanese_mfa models."""

    def __init__(
        self,
        cache_dir=None,
        dictionary_path=None,
        acoustic_model: str = "japanese_mfa",
        frontend: JapaneseFrontend | None = None,
    ):
        super().__init__(
            cache_dir=cache_dir,
            dictionary_path=dictionary_path or DEFAULT_JAPANESE_DICT,
            acoustic_model=acoustic_model,
        )
        self.frontend = frontend or JapaneseFrontend()

    def prepare_transcript(self, transcript: str) -> str:
        """Convert ordinary Japanese text to whitespace-segmented MFA text."""
        return self.frontend.to_mfa_transcript(transcript)

    def align(self, waveform, transcript, expected_phones=None):
        """Align Japanese audio.

        expected_phones is accepted for API compatibility, but it must use
        MFA's Japanese phone set. OpenJTalk phones are intentionally not mapped
        here because a lossy mapping would corrupt duration supervision.
        """
        prepared = self.prepare_transcript(transcript)
        result = super().align(
            waveform,
            prepared,
            expected_phones=expected_phones,
        )
        result["source_transcript"] = transcript
        result["mfa_transcript"] = prepared
        return result

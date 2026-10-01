"""Japanese Montreal Forced Aligner integration.

The Japanese MFA acoustic model and dictionary use MFA's Japanese phone set,
which is not the same inventory as OpenJTalk. This class therefore exposes
alignment as a standalone preprocessing primitive; callers must not compare its
phones directly with JapaneseFrontend.phones until a canonical phone mapping
has been selected for the Japanese training pipeline.
"""

from __future__ import annotations

import shutil
import subprocess
from typing import Optional

import torchaudio

from .mfa import DEFAULT_CACHE, MFAAligner, SAMPLE_RATE, read_mfa_csv
from ..frontend import JapaneseFrontend


class JapaneseMFAAligner(MFAAligner):
    """MFA aligner configured for the official japanese_mfa models.

    MFA 3.x performs Japanese morphological tokenization internally and can use
    the Japanese G2P model for out-of-vocabulary words. The alignment path
    therefore keeps the original Japanese transcript instead of forcing an
    OpenJTalk segmentation into the pronunciation dictionary.
    """

    def __init__(
        self,
        cache_dir=None,
        dictionary_model: str = "japanese_mfa",
        acoustic_model: str = "japanese_mfa",
        g2p_model: str = "japanese_mfa",
        frontend: Optional[JapaneseFrontend] = None,
    ):
        # dictionary_path is unused by this subclass because Japanese alignment
        # deliberately avoids the English mini-dictionary optimization.
        super().__init__(
            cache_dir=cache_dir or (DEFAULT_CACHE / "japanese"),
            dictionary_path="japanese_mfa",
            acoustic_model=acoustic_model,
        )
        self.dictionary_model = dictionary_model
        self.g2p_model = g2p_model
        self.frontend = frontend or JapaneseFrontend()

    def prepare_transcript(self, transcript: str) -> str:
        """Return OpenJTalk's diagnostic word segmentation for inspection."""
        return self.frontend.to_mfa_transcript(transcript)

    def align(self, waveform, transcript, expected_phones=None):
        """Align Japanese audio using MFA's own Japanese tokenizer and G2P.

        expected_phones, when supplied, must use MFA's Japanese phone set.
        OpenJTalk phones are intentionally not mapped here because a lossy
        mapping would corrupt duration supervision.
        """
        mfa_bin = shutil.which("mfa")
        if mfa_bin is None:
            raise RuntimeError(
                "The mfa command was not found. Install Montreal Forced Aligner "
                "and download the japanese_mfa acoustic, dictionary and G2P models."
            )

        transcript = " ".join((transcript or "").split())
        if not transcript:
            raise ValueError("transcript must not be empty")

        wav = waveform
        if wav.dim() == 1:
            wav = wav.unsqueeze(0)

        with self._lock:
            corpus = self.cache_dir / "corpus"
            temp_root = self.cache_dir / "tmp"
            aligned = self.cache_dir / "aligned"
            shutil.rmtree(corpus, ignore_errors=True)
            shutil.rmtree(aligned, ignore_errors=True)
            corpus.mkdir(parents=True, exist_ok=True)
            temp_root.mkdir(parents=True, exist_ok=True)

            torchaudio.save(str(corpus / "audio.wav"), wav, SAMPLE_RATE)
            (corpus / "audio.txt").write_text(transcript, encoding="utf-8")

            cmd = [
                mfa_bin,
                "align",
                "--clean",
                "-j",
                "1",
                "--output_format",
                "csv",
                "--language",
                "japanese",
                "--g2p_model_path",
                self.g2p_model,
                str(corpus),
                self.dictionary_model,
                self.acoustic_model,
                str(aligned),
                "-t",
                str(temp_root),
            ]
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
            )
            if proc.returncode != 0:
                raise RuntimeError(
                    "Japanese MFA alignment failed. Ensure the Japanese models "
                    "and Sudachi dependencies are installed.\n"
                    + proc.stderr[-1800:]
                )

            result = read_mfa_csv(
                aligned / "audio.csv",
                words_text=transcript,
                expected_phones=expected_phones,
            )
            result["source_transcript"] = transcript
            # Useful for debugging OpenJTalk-vs-MFA tokenization without making
            # the diagnostic segmentation part of the alignment contract.
            result["openjtalk_mfa_transcript"] = self.prepare_transcript(transcript)
            return result

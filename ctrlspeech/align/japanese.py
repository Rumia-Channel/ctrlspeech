"""Japanese Montreal Forced Aligner integration.

The Japanese MFA acoustic model and dictionary use MFA's Japanese phone set,
which is not the same inventory as OpenJTalk. This class therefore exposes
alignment as a standalone preprocessing primitive; callers must not compare its
phones directly with JapaneseFrontend.phones until a canonical phone mapping
has been selected for the Japanese training pipeline.
"""

from __future__ import annotations

import math
import os
import shutil
import signal
import subprocess
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Optional

import soundfile as sf
import torch

from .mfa import DEFAULT_CACHE, MFAAligner, SAMPLE_RATE, read_mfa_csv
from ..frontend import JapaneseFrontend


_WINDOWS = os.name == "nt"


class _MFACleanupError(RuntimeError):
    """Process-tree termination failed; retain its scratch files for safety."""


def _terminate_mfa_tree(process: subprocess.Popen) -> None:
    """Stop only this invocation's process group/tree, including Kaldi workers."""
    if _WINDOWS:
        # Unlike Popen.kill(), /T includes descendants. Never use an image-name
        # filter: another user's or another aligner's MFA may also be running.
        result = subprocess.run(
            ["taskkill", "/PID", str(process.pid), "/T", "/F"],
            capture_output=True, text=True, encoding="utf-8", errors="replace",
            timeout=10,
        )
        if result.returncode:
            raise OSError("taskkill failed: " + result.stderr[-1000:])
    else:
        # start_new_session=True makes the child PID its process-group ID;
        # the caller's group and concurrent alignments are never targeted.
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass  # The entire group already exited.


def _run_mfa_command(command: list[str], *, timeout: float | None):
    options = (
        {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}
        if _WINDOWS else {"start_new_session": True}
    )
    # Files rather than pipes avoid hangs from an inherited pipe handle in a
    # descendant, including Windows' background communicate() reader threads.
    with tempfile.TemporaryFile() as stdout_file, tempfile.TemporaryFile() as stderr_file:
        process = subprocess.Popen(
            command, stdout=stdout_file, stderr=stderr_file, **options,
        )
        try:
            process.wait(timeout=timeout)
        except BaseException:
            # Cover cancellation as well as timeouts. Do not remove scratch
            # until the command and its normal children have been stopped.
            try:
                _terminate_mfa_tree(process)
                process.wait(timeout=10)
            except (OSError, subprocess.SubprocessError) as cleanup_error:
                # Reap the direct child if possible, but do not pretend this
                # stopped descendants when process-tree termination failed.
                try:
                    process.kill()
                    process.wait(timeout=5)
                except (OSError, subprocess.SubprocessError):
                    pass
                raise _MFACleanupError(
                    "Could not confirm termination of the Japanese MFA process tree; "
                    "child processes may still be running. " + str(cleanup_error)
                ) from cleanup_error
            raise
        stdout_file.seek(0)
        stderr_file.seek(0)
        stdout = stdout_file.read().decode("utf-8", errors="replace")
        stderr = stderr_file.read().decode("utf-8", errors="replace")
    return subprocess.CompletedProcess(command, process.returncode, stdout, stderr)


@contextmanager
def _mfa_workspace(cache_dir: Path):
    run_root = Path(tempfile.mkdtemp(prefix="run-", dir=cache_dir))
    safe_to_remove = True
    try:
        yield run_root
    except _MFACleanupError as exc:
        safe_to_remove = False
        raise _MFACleanupError(f"{exc} Scratch files retained at {run_root}") from exc
    finally:
        if safe_to_remove:
            shutil.rmtree(run_root)


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
        timeout_seconds: float | None = 600,
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
        if timeout_seconds is not None and (
            isinstance(timeout_seconds, bool)
            or not math.isfinite(timeout_seconds)
            or timeout_seconds <= 0
        ):
            raise ValueError("timeout_seconds must be a positive finite number or None")
        self.timeout_seconds = timeout_seconds

    def prepare_transcript(self, transcript: str) -> str:
        """Return OpenJTalk's diagnostic word segmentation for inspection."""
        return self.frontend.to_mfa_transcript(transcript)

    def align(self, waveform, transcript, expected_phones=None):
        """Align Japanese audio using MFA's own Japanese tokenizer and G2P.

        expected_phones, when supplied, must use MFA's Japanese phone set.
        OpenJTalk phones are intentionally not mapped here because a lossy
        mapping would corrupt duration supervision.
        """
        if not isinstance(transcript, str):
            raise TypeError("transcript must be a str")
        transcript = " ".join(transcript.split())
        if not transcript:
            raise ValueError("transcript must not be empty")
        if not isinstance(waveform, torch.Tensor):
            raise TypeError("waveform must be a torch.Tensor")
        wav = waveform
        if wav.dim() == 1:
            wav = wav.unsqueeze(0)
        if wav.dim() != 2 or wav.shape[0] != 1 or wav.shape[1] == 0:
            raise ValueError("waveform must be non-empty mono audio with shape [T] or [1, T]")
        if not wav.is_floating_point():
            raise ValueError("waveform must contain floating-point audio samples")
        if not torch.isfinite(wav).all():
            raise ValueError("waveform must contain only finite samples")

        mfa_bin = shutil.which("mfa")
        if mfa_bin is None:
            raise RuntimeError(
                "The mfa command was not found. Install Montreal Forced Aligner "
                "and download the japanese_mfa acoustic, dictionary and G2P models."
            )

        self.cache_dir.mkdir(parents=True, exist_ok=True)
        # An instance-local lock cannot protect a shared cache across aligner
        # instances or processes. Never delete another run's corpus/results.
        with self._lock, _mfa_workspace(self.cache_dir) as run_root:
            corpus = run_root / "corpus"
            temp_root = run_root / "tmp"
            aligned = run_root / "aligned"
            corpus.mkdir(parents=True, exist_ok=True)
            temp_root.mkdir(parents=True, exist_ok=True)

            # torchaudio >= 2.9 delegates save() to optional TorchCodec. WAV
            # scratch files need only the already-required soundfile backend.
            sf.write(
                str(corpus / "audio.wav"),
                wav.detach().cpu().float().squeeze(0).numpy(),
                SAMPLE_RATE,
                subtype="PCM_16",
            )
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
            try:
                proc = _run_mfa_command(cmd, timeout=self.timeout_seconds)
            except subprocess.TimeoutExpired as exc:
                raise RuntimeError(
                    f"Japanese MFA alignment timed out after {self.timeout_seconds} seconds; "
                    "check the MFA models or increase timeout_seconds."
                ) from exc
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

            # This is diagnostic metadata only. Do not make successful MFA
            # alignment depend on pyopenjtalk being installed or on its parser
            # accepting a particular input sentence.
            try:
                result["openjtalk_mfa_transcript"] = self.prepare_transcript(transcript)
            except (RuntimeError, ValueError):
                pass
            return result

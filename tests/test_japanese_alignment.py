"""MFA integration contract checks; the external executable is mocked."""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
import subprocess
import threading

import pytest
import torch

import ctrlspeech.align.japanese as japanese
from ctrlspeech.align.japanese import JapaneseMFAAligner


class DiagnosticFrontend:
    def to_mfa_transcript(self, text):
        return "猫 は"


@pytest.fixture
def mfa(monkeypatch):
    monkeypatch.setattr(japanese.shutil, "which", lambda name: "/usr/bin/mfa")
    write_audio = japanese.sf.write
    monkeypatch.setattr(japanese.sf, "write", lambda path, wav, rate, **kwargs: Path(path).touch())
    monkeypatch.setattr(japanese, "read_mfa_csv", lambda path, **kwargs: {
        "phones": "n e k o", "starts": "0 .1 .2 .3", "ends": ".1 .2 .3 .4",
    })
    return write_audio


def test_alignment_uses_isolated_runs_and_preserves_shared_cache(tmp_path, mfa, monkeypatch):
    existing = tmp_path / "corpus" / "important.txt"
    existing.parent.mkdir()
    existing.write_text("another run")
    barrier = threading.Barrier(2)
    corpus_paths = []

    def run(command, **kwargs):
        corpus = Path(command[command.index("japanese_mfa") + 1])
        corpus_paths.append(corpus)
        assert (corpus / "audio.txt").read_text(encoding="utf-8") == "猫 は"
        assert (corpus / "audio.wav").exists()
        assert kwargs["timeout"] == 600
        barrier.wait(timeout=5)
        assert all(path.exists() for path in corpus_paths)
        return SimpleNamespace(returncode=0, stderr="")

    monkeypatch.setattr(japanese, "_run_mfa_command", run)
    aligners = [JapaneseMFAAligner(cache_dir=tmp_path, frontend=DiagnosticFrontend()) for _ in range(2)]
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda aligner: aligner.align(torch.zeros(160), " 猫  は "), aligners))
    assert corpus_paths[0] != corpus_paths[1]
    assert all(not path.exists() for path in corpus_paths)
    assert existing.read_text() == "another run"
    assert all(result["source_transcript"] == "猫 は" for result in results)


def test_alignment_timeout_is_actionable_and_cleans_scratch(tmp_path, mfa, monkeypatch):
    def run(command, **kwargs):
        raise subprocess.TimeoutExpired(command, kwargs["timeout"])

    monkeypatch.setattr(japanese, "_run_mfa_command", run)
    aligner = JapaneseMFAAligner(cache_dir=tmp_path, timeout_seconds=1)
    with pytest.raises(RuntimeError, match="timed out after 1 seconds"):
        aligner.align(torch.zeros(160), "猫")
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("timeout", [0, -1, float("nan"), float("inf"), True])
def test_alignment_rejects_invalid_timeouts(tmp_path, timeout):
    with pytest.raises(ValueError, match="timeout_seconds"):
        JapaneseMFAAligner(cache_dir=tmp_path, timeout_seconds=timeout)


@pytest.mark.parametrize("waveform", [
    torch.zeros(0), torch.zeros(2, 10), torch.zeros(1, 1, 10),
    torch.tensor([float("nan")]), torch.tensor([float("inf")]),
    torch.zeros(10, dtype=torch.long), torch.zeros(10, dtype=torch.complex64),
])
def test_alignment_rejects_invalid_waveforms_before_launch(tmp_path, waveform):
    aligner = JapaneseMFAAligner(cache_dir=tmp_path)
    with pytest.raises(ValueError, match="waveform"):
        aligner.align(waveform, "猫")
    assert list(tmp_path.iterdir()) == []


def test_alignment_writes_pcm_wav_without_torchcodec(tmp_path, mfa, monkeypatch):
    monkeypatch.setattr(japanese.sf, "write", mfa)

    def run(command, **kwargs):
        corpus = Path(command[command.index("japanese_mfa") + 1])
        audio, sample_rate = japanese.sf.read(corpus / "audio.wav")
        assert sample_rate == 16000
        assert audio.shape == (160,)
        assert (audio == 0).all()
        assert japanese.sf.info(corpus / "audio.wav").subtype == "PCM_16"
        return SimpleNamespace(returncode=0, stderr="")

    monkeypatch.setattr(japanese, "_run_mfa_command", run)
    aligner = JapaneseMFAAligner(cache_dir=tmp_path, frontend=DiagnosticFrontend())
    assert aligner.align(torch.zeros(160, requires_grad=True), "猫")["phones"] == "n e k o"


def test_alignment_failure_cleans_scratch_and_keeps_error(tmp_path, mfa, monkeypatch):
    monkeypatch.setattr(japanese, "_run_mfa_command", lambda *args, **kwargs: SimpleNamespace(returncode=1, stderr="model missing"))
    with pytest.raises(RuntimeError, match="model missing"):
        JapaneseMFAAligner(cache_dir=tmp_path).align(torch.zeros(10), "猫")
    assert list(tmp_path.iterdir()) == []


def test_timeout_stops_descendants_before_removing_workspace(tmp_path, monkeypatch):
    import sys
    import time

    marker = tmp_path / "escaped-child.txt"
    launched = tmp_path / "child-started.txt"
    script = tmp_path / "fake_mfa.py"
    child_code = (
        "import pathlib,time; time.sleep(0.8); "
        f"pathlib.Path({str(marker)!r}).write_text('escaped')"
    )
    script.write_text(
        "import pathlib, subprocess, sys, time\n"
        f"child = subprocess.Popen([sys.executable, '-c', {child_code!r}])\n"
        f"pathlib.Path({str(launched)!r}).write_text(str(child.pid))\n"
        "time.sleep(30)\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(japanese.shutil, "which", lambda name: sys.executable)
    run_command = japanese._run_mfa_command
    popen = subprocess.Popen

    def start_after_ready(command, **kwargs):
        process = popen(command, **kwargs)
        # Exclude interpreter startup jitter from the tiny timeout, especially
        # on loaded Windows CI. Only the fake MFA command uses this hook.
        deadline = time.monotonic() + 10
        while not launched.exists() and time.monotonic() < deadline:
            time.sleep(0.01)
        return process

    monkeypatch.setattr(japanese.subprocess, "Popen", start_after_ready)

    def run_fake_mfa(command, **kwargs):
        return run_command([sys.executable, str(script)], **kwargs)

    monkeypatch.setattr(japanese, "_run_mfa_command", run_fake_mfa)
    cache = tmp_path / "cache"
    aligner = JapaneseMFAAligner(cache_dir=cache, timeout_seconds=0.3)
    with pytest.raises(RuntimeError, match="timed out after 0.3 seconds"):
        aligner.align(torch.zeros(160), "猫")
    assert launched.exists(), "The fake MFA must actually spawn its child before timing out"
    time.sleep(0.9)
    assert not marker.exists(), "A descendant survived the MFA timeout"
    assert list(cache.iterdir()) == []


def test_windows_cleanup_targets_only_the_launched_pid(monkeypatch):
    calls = []
    monkeypatch.setattr(japanese, "_WINDOWS", True)

    def taskkill(command, **kwargs):
        calls.append((command, kwargs))
        return SimpleNamespace(returncode=0, stderr="")

    monkeypatch.setattr(japanese.subprocess, "run", taskkill)
    japanese._terminate_mfa_tree(SimpleNamespace(pid=12345))
    assert calls[0][0] == ["taskkill", "/PID", "12345", "/T", "/F"]
    assert calls[0][1]["timeout"] == 10


def test_failed_tree_cleanup_retains_scratch_and_reports_uncertainty(tmp_path, mfa, monkeypatch):
    def fail_cleanup(*args, **kwargs):
        raise japanese._MFACleanupError("child processes may still be running")

    monkeypatch.setattr(japanese, "_run_mfa_command", fail_cleanup)
    with pytest.raises(RuntimeError, match="Scratch files retained"):
        JapaneseMFAAligner(cache_dir=tmp_path).align(torch.zeros(10), "猫")
    roots = list(tmp_path.glob("run-*"))
    assert len(roots) == 1
    assert (roots[0] / "corpus" / "audio.wav").exists()


def test_windows_tree_cleanup_errors_are_not_silently_ignored(monkeypatch):
    monkeypatch.setattr(japanese, "_WINDOWS", True)
    monkeypatch.setattr(japanese.subprocess, "run", lambda *args, **kwargs:
                        SimpleNamespace(returncode=1, stderr="access denied"))
    with pytest.raises(OSError, match="taskkill failed: access denied"):
        japanese._terminate_mfa_tree(SimpleNamespace(pid=12345))


def test_process_runner_captures_non_ascii_errors():
    import sys

    result = japanese._run_mfa_command(
        [sys.executable, "-c",
         "import sys; sys.stderr.buffer.write('日本語エラー'.encode('utf-8')); sys.exit(3)"],
        timeout=10,
    )
    assert result.returncode == 3
    assert "日本語エラー" in result.stderr

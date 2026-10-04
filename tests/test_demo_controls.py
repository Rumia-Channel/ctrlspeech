"""Headless Panel event tests with CPU stubs, not voice-quality evaluation."""
import asyncio
from concurrent.futures import ThreadPoolExecutor
import inspect
from pathlib import Path
import threading
from types import SimpleNamespace

import numpy as np
import panel as pn
import pytest

from demo import app as demo
from ctrlspeech.pipeline import Generation
from test_inference_controls import make_baseline, stub_tts


def walk(node):
    yield node
    for child in getattr(node, "objects", []) or []:
        yield from walk(child)


def button(layout, name):
    return next(node for node in walk(layout) if isinstance(node, pn.widgets.Button) and node.name == name)


def click(widget):
    handlers = [w.fn for w in widget.param.watchers["clicks"]["value"]
                if inspect.iscoroutinefunction(w.fn) or getattr(w.fn, "__name__", "").startswith(("on_", "apply_", "reset_"))]
    result = handlers[0](None)
    if inspect.isawaitable(result):
        asyncio.run(result)


@pytest.mark.parametrize("focus", ["pitch", "loudness", "duration"])
def test_panel_repeated_apply_reset_and_regenerate(focus, monkeypatch, tmp_path):
    transcript = tmp_path / "example.txt"
    transcript.write_text("あい", encoding="utf-8")
    example = dict(name="CPU stub", prompt_txt=transcript, target_txt=transcript,
                   prompt_wav=tmp_path / "prompt.wav", target_wav=tmp_path / "target.wav",
                   defaults=dict(word="あ", ratio=2.0, semitones=5, db=8))
    monkeypatch.setattr(demo, "EXAMPLES", [example])
    state, tts = make_baseline(), stub_tts()
    calls = []

    def regenerate(baseline, **kwargs):
        tts.build_edited_controls(baseline, **kwargs)
        calls.append(kwargs)
        # Fixed dummy output: completion/control plumbing, never quality evidence.
        return Generation(np.zeros(16000), baseline.gen_f0.copy(), baseline.gen_loud.copy())

    monkeypatch.setattr(demo, "get_model", lambda: SimpleNamespace(
        regenerate=regenerate, align=lambda *a: dict(word_data=state.word_data)))
    monkeypatch.setattr(demo, "synthesize_baseline", lambda *a: state)
    layout = demo.build_app()
    selector = next(node for node in walk(layout) if isinstance(node, pn.widgets.RadioButtonGroup) and node.options == demo.CONTROL_LABELS)
    selector.value = focus
    click(button(layout, "Generate baseline & align"))
    editor = next(node for node in walk(layout) if hasattr(node, "get_pitch") and hasattr(node, "get_words"))
    click(button(layout, "Apply"))
    edited = (editor.get_pitch(), editor.get_loudness(), editor.get_phonemes())
    click(button(layout, "Apply"))
    np.testing.assert_array_equal(editor.get_pitch(), edited[0])
    np.testing.assert_array_equal(editor.get_loudness(), edited[1])
    assert editor.get_phonemes() == edited[2]
    click(button(layout, "Reset"))
    np.testing.assert_array_equal(editor.get_pitch(), state.gen_f0)
    np.testing.assert_array_equal(editor.get_loudness(), state.gen_loud)
    assert editor.get_phonemes() == state.gen_phoneme_data
    click(button(layout, "Apply"))
    click(button(layout, "Regenerate with current control"))
    click(button(layout, "Regenerate with current control"))
    assert len(calls) == 2
    assert all(not node.object.startswith("Error:") for node in walk(layout)
               if isinstance(node, pn.pane.Markdown) and isinstance(node.object, str))
    np.testing.assert_array_equal(calls[0]["pitch"], calls[1]["pitch"])
    assert not button(layout, "Regenerate with current control").disabled


def test_concurrent_uploads_use_isolated_files_and_cleanup(monkeypatch):
    import librosa

    barrier = threading.Barrier(2)
    paths = []

    def load(path, **kwargs):
        path = Path(path)
        paths.append(path)
        barrier.wait(timeout=10)
        return np.array([int(path.read_bytes())], dtype=np.float32), 16000

    monkeypatch.setattr(librosa, "load", load)
    monkeypatch.setattr(demo, "get_model", lambda: SimpleNamespace(from_audio=lambda audio, *a, **k: audio))
    with ThreadPoolExecutor(max_workers=2) as pool:
        pending = [pool.submit(demo.adopt_upload, value, "same.wav", "あ", 32, 1.5)
                   for value in (b"1", b"2")]
        results = [future.result(timeout=15) for future in pending]
    assert sorted(float(result[0]) for result in results) == [1, 2]
    assert paths[0] != paths[1]
    assert all(not path.exists() and not path.parent.exists() for path in paths)


def test_bad_upload_is_cleaned_without_loading_model(monkeypatch):
    import librosa

    paths = []

    def load(path, **kwargs):
        paths.append(Path(path))
        raise RuntimeError("broken audio")

    monkeypatch.setattr(librosa, "load", load)
    monkeypatch.setattr(demo, "get_model", lambda: pytest.fail("loaded model before decoding"))
    with pytest.raises(ValueError, match="Could not decode"):
        demo.adopt_upload(b"broken", "bad.wav", "あ", 32, 1.5)
    assert all(not path.parent.exists() for path in paths)


def test_japanese_demo_does_not_offer_legacy_english_example(tmp_path, monkeypatch):
    import json

    for name in ("prompt.wav", "target.wav", "prompt.txt", "target.txt"):
        (tmp_path / name).write_bytes(b"fixture")
    entry = dict(name="Legacy", prompt_wav="prompt.wav", target_wav="target.wav",
                 prompt_txt="prompt.txt", target_txt="target.txt")
    (tmp_path / "examples.json").write_text(json.dumps([entry, dict(entry, name="Japanese", language="ja")]))
    monkeypatch.setattr(demo, "ASSET_DIR", tmp_path)
    assert [entry["name"] for entry in demo.load_examples()] == ["Japanese"]


def test_canonical_annotation_upload_uses_its_transcript_and_skips_temp_leaks(monkeypatch):
    import io
    import soundfile as sf

    wav = io.BytesIO()
    sf.write(wav, np.zeros(1600), 16000, format="WAV")
    calls = []
    monkeypatch.setattr(demo, "get_model", lambda: SimpleNamespace(
        from_audio=lambda *args, **kwargs: calls.append((args, kwargs)) or "baseline"))
    annotation = "あい\na i\n0 .05\n.05 .1\n".encode("utf-8")
    result = demo.adopt_upload(wav.getvalue(), "silence.wav", "", 32, 1.5, annotation)
    assert result == "baseline"
    assert calls[0][0][0].shape == (1600,)
    assert calls[0][0][1] == "あい"
    assert calls[0][1]["annotation"]["starts"] == "0 .05"


def test_canonical_upload_rejects_bad_annotation_before_loading(monkeypatch):
    monkeypatch.setattr(demo, "get_model", lambda: pytest.fail("loaded assets"))
    with pytest.raises(ValueError, match="four lines"):
        demo.adopt_upload(b"unused", "x.wav", "", 32, 1.5, b"one\ntwo\n")


def test_queued_baseline_clicks_do_not_start_duplicate_jobs(monkeypatch, tmp_path):
    transcript = tmp_path / "example.txt"
    transcript.write_text("あい", encoding="utf-8")
    monkeypatch.setattr(demo, "EXAMPLES", [dict(name="CPU stub", prompt_txt=transcript,
                       target_txt=transcript, prompt_wav=tmp_path / "prompt.wav")])
    started, release = threading.Event(), threading.Event()
    calls = []

    def generate(*args):
        calls.append("generate")
        started.set()
        assert release.wait(timeout=10)
        return make_baseline()

    monkeypatch.setattr(demo, "synthesize_baseline", generate)
    layout = demo.build_app()
    handler = next(w.fn for w in button(layout, "Generate baseline & align").param.watchers["clicks"]["value"]
                   if inspect.iscoroutinefunction(w.fn))

    async def scenario():
        first = asyncio.create_task(handler(None))
        try:
            assert await asyncio.to_thread(started.wait, 5)
            await handler(None)
            assert calls == ["generate"]
        finally:
            release.set()
            await first

    asyncio.run(scenario())
    assert calls == ["generate"]
    assert not button(layout, "Generate baseline & align").disabled

"""CPU-only inference/control contracts; these tests do not synthesize speech."""
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from ctrlspeech import pipeline
from ctrlspeech.cli import main, parse_args, read_transcript
from ctrlspeech.pipeline import Baseline, CtrlSpeech
from ctrlspeech.retime import retime_word_curves, time_to_frame_index


def make_baseline():
    return Baseline(
        input_ids=torch.tensor([[1, 2, 3, 1, 2]]),
        text_masks=torch.ones((1, 5), dtype=torch.bool),
        prompt_audio=torch.ones(1, 8000), speaker_emb=torch.ones(192),
        prompt_segments=[(0, 20), (20, 40)], prompt_np=np.ones(8000),
        prompt_duration=0.5, prompt_f0=np.full(51, 40), prompt_loud=np.full(51, 30),
        gen_np=np.ones(16000), gen_f0=np.tile([0, 1, 40, 60], 26)[:101],
        gen_loud=np.full(101, 30),
        gen_phoneme_data=[["a", .1, .3], ["i", .5, .8]],
        word_data=[
            dict(label="あ", start=.1, end=.3, phone_start=0, phone_end=1),
            dict(label="い", start=.5, end=.8, phone_start=1, phone_end=2),
        ], target_token_refs=[0, 1], target_words="あい", target_phones="a i",
        linguistic_features={"valid": torch.ones((1, 5), dtype=torch.bool)},
        native_text_inputs=torch.tensor([[3, 4]]),
        native_text_masks=torch.ones((1, 2), dtype=torch.bool),
    )


def stub_tts():
    return CtrlSpeech(None, None, {"a": 1, "i": 2, "|": 3}, None, "cpu")


def test_cli_uses_japanese_default():
    args = parse_args(["--audio", "x.wav", "--transcript-text", "あ", "--out", "out.wav"])
    assert args.model == "japanese-lfm2-350m"


@pytest.mark.parametrize("flags", [
    ["--steps", "0"], ["--pitch-shift", "nan"], ["--loudness-shift", "inf"],
    ["--cfg-strength", "-1"], ["--stretch-ratio", "2"], ["--stretch-word", "あ"],
    ["--stretch-word", "あ", "--stretch-ratio", "0"],
    ["--stretch-word", "あ", "--stretch-ratio", "nan"],
    ["--stretch-word", "あ", "--stretch-ratio", "2", "--stretch-seconds", "1"],
    ["--prompt-text", "prompt.txt"], ["--transcript", "other.txt"],
])
def test_cli_rejects_invalid_controls_before_loading(flags, monkeypatch):
    monkeypatch.setattr(CtrlSpeech, "from_pretrained", lambda *a, **k: pytest.fail("loaded assets"))
    with pytest.raises(SystemExit):
        main(["--audio", "x.wav", "--transcript-text", "あ", "--out", "out.wav", *flags])


def test_blank_transcript_is_rejected_before_loading(monkeypatch):
    monkeypatch.setattr(CtrlSpeech, "from_pretrained", lambda *a, **k: pytest.fail("loaded assets"))
    with pytest.raises(SystemExit, match="nonempty"):
        main(["--audio", "x.wav", "--transcript-text", "  ", "--out", "out.wav"])


def test_shift_pitch_keeps_unvoiced_and_shifts_lowest_voiced_bin():
    original = np.array([0, 1, 40, 127])
    changed = pipeline.shift_pitch_semitones(original, 12)
    assert changed[0] == 0
    assert changed[1] > 1
    assert changed[-1] == 127
    np.testing.assert_array_equal(original, [0, 1, 40, 127])
    assert np.all(np.isfinite(pipeline.shift_pitch_semitones(original, 1e200)))


@pytest.mark.parametrize("function,arg", [(pipeline.shift_pitch_semitones, np.nan), (pipeline.shift_loudness_db, np.inf)])
def test_shift_rejects_nonfinite_argument(function, arg):
    with pytest.raises(ValueError, match="finite"):
        function([0, 1, 2], arg)


@pytest.mark.parametrize("bad", [[], [[1, 2]], [np.nan], [np.inf], np.zeros(6002)])
def test_edited_controls_reject_invalid_curves(bad):
    with pytest.raises(ValueError):
        stub_tts().build_edited_controls(make_baseline(), pitch=bad)


def test_edited_controls_preserve_prompt_offset_and_quantize():
    state = make_baseline()
    segments, pitch, loudness = stub_tts().build_edited_controls(state)
    assert segments == [(0, 20), (20, 40), (0, 0), (61, 81), (101, 131)]
    np.testing.assert_array_equal(pitch[:51], state.prompt_f0)
    np.testing.assert_array_equal(loudness[51:], state.gen_loud)


@pytest.mark.parametrize("phones", [
    [["i", .1, .3], ["a", .5, .8]],
    [["a", .1, .6], ["i", .5, .8]],
    [["a", .1, np.nan], ["i", .5, .8]],
])
def test_edited_controls_reject_changed_labels_or_invalid_boundaries(phones):
    with pytest.raises(ValueError):
        stub_tts().build_edited_controls(make_baseline(), phonemes=phones)


def test_unaligned_generation_cannot_be_regenerated():
    state = make_baseline()
    state.gen_phoneme_data = []
    state.target_token_refs = []
    with pytest.raises(ValueError, match="align=True"):
        stub_tts().build_edited_controls(state)


def test_regeneration_reuses_both_japanese_condition_streams(monkeypatch):
    state, tts = make_baseline(), stub_tts()
    calls = []
    monkeypatch.setattr(tts, "_sample", lambda **kw: calls.append(kw) or np.zeros(16000))
    monkeypatch.setattr(pipeline, "get_pitch_and_loudness", lambda *a, **k: (None, np.zeros(101), None, np.zeros(101)))
    tts.regenerate(state)
    tts.regenerate(state, pitch=state.gen_f0 + 1)
    assert len(calls) == 2
    for call in calls:
        assert call["linguistic_features"] is state.linguistic_features
        assert call["native_text_inputs"] is state.native_text_inputs
        assert call["native_text_masks"] is state.native_text_masks
    np.testing.assert_array_equal(state.gen_f0[:4], [0, 1, 40, 60])


def test_generate_passes_japanese_inputs_and_uses_frame_offset(monkeypatch):
    state, tts = make_baseline(), stub_tts()
    sequence = SimpleNamespace(prompt=SimpleNamespace(phones=("a", "i")), target=SimpleNamespace(phones=("a", "i")))
    batch = dict(input_ids=state.input_ids, text_masks=state.text_masks,
                 linguistic_features=state.linguistic_features,
                 native_text_inputs=state.native_text_inputs, native_text_masks=state.native_text_masks)
    monkeypatch.setattr(tts, "_encode_text_pair", lambda *args: (sequence, batch))
    monkeypatch.setattr(pipeline.librosa, "load", lambda *a, **k: (np.ones(8000, np.float32), 16000))
    monkeypatch.setattr(pipeline.librosa.effects, "trim", lambda x, **k: (x, (0, len(x))))
    monkeypatch.setattr(pipeline, "get_pitch_and_loudness", lambda x, **k: (None, np.ones(len(x)//160+1), None, np.ones(len(x)//160+1)))
    tts.speaker_embedding = SimpleNamespace(_extract_spk_embedding=lambda x: [torch.ones(192)])
    calls = []
    monkeypatch.setattr(tts, "_sample", lambda **kw: calls.append(kw) or np.ones(16000))
    annotation = dict(words="あい", phones="a i", starts="0.1 0.2", ends="0.2 0.4")
    baseline = tts.generate("dummy.wav", annotation, annotation, align=False)
    assert calls[0]["duration_segments"] == [(10, 20), (20, 40), (0, 0), (61, 71), (71, 91)]
    assert calls[0]["linguistic_features"] is state.linguistic_features
    assert baseline.native_text_inputs is state.native_text_inputs


@pytest.mark.parametrize("phones", ["a | i", "a ɪ", "A I"])
def test_raw_mfa_annotation_never_silently_enters_japanese_model(phones):
    with pytest.raises(ValueError, match="reconciled OpenJTalk timings"):
        pipeline._require_canonical_phones({"phones": phones}, ["a", "i"], "test")


@pytest.mark.parametrize("missing,unexpected", [(["causalAR.model.layers.0.weight"], []), (["generator.encoder.weight"], []), ([], ["old_qwen.weight"])])
def test_partial_or_wrong_checkpoint_is_rejected(missing, unexpected):
    with pytest.raises(RuntimeError, match="Incompatible or incomplete"):
        pipeline._check_checkpoint_keys(missing, unexpected)


@pytest.mark.parametrize("annotation", [
    dict(phones="a i", starts="0.0", ends="0.1 0.2"),
    dict(phones="a", starts="nan", ends="1"),
    dict(phones="a i", starts="0 .1", ends=".2 .3"),
    dict(phones="a", starts="0", ends="61"),
])
def test_annotation_validation_does_not_truncate_or_accept_bad_times(annotation):
    with pytest.raises(ValueError):
        pipeline.build_prompt_segments(annotation)


def test_word_retime_preserves_gaps_sentinels_and_is_repeatable():
    state = make_baseline()
    result = retime_word_curves(state.gen_f0, state.gen_loud, state.gen_phoneme_data, state.word_data, 0, .4, 160, 16000)
    pitch, loud, phones, words, summary = result
    assert len(pitch) == 121
    assert words[1]["start"] - words[0]["end"] == pytest.approx(.2)
    np.testing.assert_array_equal(pitch[50:], state.gen_f0[30:])
    assert set(pitch) <= set(state.gen_f0)
    restored = retime_word_curves(pitch, loud, phones, words, 0, .2, 160, 16000)
    assert len(restored[0]) == 101
    np.testing.assert_allclose(np.array(restored[2], dtype=object)[:, 1:].astype(float), np.array(state.gen_phoneme_data, dtype=object)[:, 1:].astype(float))


def test_subframe_retime_moves_curve_and_later_boundaries_by_identical_frames():
    state = make_baseline()
    state.gen_phoneme_data[0][1:3] = [.104, .315]
    state.word_data[0].update(start=.104, end=.315)
    pitch, _, phones, words, summary = retime_word_curves(state.gen_f0, state.gen_loud, state.gen_phoneme_data, state.word_data, 0, .4, 160, 16000)
    added = len(pitch) - len(state.gen_f0)
    assert summary["delta"] == pytest.approx(added / 100)
    assert words[1]["start"] - state.word_data[1]["start"] == pytest.approx(added / 100)
    assert words[1]["start"] - words[0]["end"] == pytest.approx(.5 - .315)


@pytest.mark.parametrize("value", [np.nan, np.inf, -1])
def test_frame_index_rejects_invalid_time(value):
    with pytest.raises(ValueError):
        time_to_frame_index(value, 160, 16000)


def test_subframe_final_boundary_is_preserved_by_noop_and_stretch():
    curves = np.arange(124)
    phones = [["a", 0, .5], ["i", .5, 1.234]]
    words = [dict(label="あ", start=0, end=.5, phone_start=0, phone_end=1),
             dict(label="い", start=.5, end=1.234, phone_start=1, phone_end=2)]
    for duration in (.5, .6):
        result = retime_word_curves(curves, curves, phones, words, 0, duration, 160, 16000)
        assert result[2][-1][2] == pytest.approx(1.234 + duration - .5)
        assert len(result[0]) == 124 + round((duration - .5) * 100)


def test_cli_accepts_canonical_annotations_without_mfa_paths():
    args = parse_args(["--prompt-wav", "prompt.wav", "--prompt-annotation", "prompt.txt",
                       "--target-annotation", "target.txt", "--out", "out.wav"])
    assert args.target_wav is None
    args = parse_args(["--audio", "x.wav", "--annotation", "x.txt", "--out", "out.wav"])
    assert args.transcript is None


def test_known_audio_annotation_preserves_untrimmed_timebase(monkeypatch):
    from ctrlspeech.frontend import JapaneseFrontend
    from ctrlspeech.frontend import JAPANESE_PHONE_TO_ID

    # Uses installed linguistic preprocessing only, without acoustic models.
    encoding = pipeline.encode_japanese_pair("あい", "あい")
    assert encoding.target.phones == ("a", "i")
    tts = CtrlSpeech(None, None, JAPANESE_PHONE_TO_ID,
                     SimpleNamespace(_extract_spk_embedding=lambda x: [torch.ones(192)]), "cpu")
    monkeypatch.setattr(pipeline.librosa.effects, "trim", lambda *a, **k: pytest.fail("trimmed annotated waveform"))
    monkeypatch.setattr(tts, "align", lambda *a, **k: pytest.fail("invoked MFA"))
    monkeypatch.setattr(pipeline, "get_pitch_and_loudness", lambda x, **k: (None, np.ones(len(x)//160+1), None, np.ones(len(x)//160+1)))
    audio = np.concatenate([np.zeros(3200), np.ones(12800)]).astype(np.float32)
    annotation = dict(words="あい", phones="a i", starts=".2 .5", ends=".5 .8")
    state = tts.from_audio(audio, "あい", annotation=annotation)
    np.testing.assert_array_equal(state.gen_np, audio)
    assert state.prompt_segments == [(20, 50), (50, 80)]
    assert state.gen_phoneme_data == [["a", .2, .5], ["i", .5, .8]]
    assert state.target_token_refs == [0, 1]
    assert state.word_data[0]["start"] == .2


def test_generation_save_writes_real_pcm_without_torchcodec(tmp_path):
    import soundfile as sf

    output = tmp_path / "nested" / "silence.wav"
    pipeline.Generation(np.zeros(160), np.zeros(2), np.zeros(2)).save(output)
    data, sr = sf.read(output)
    assert data.shape == (160,) and sr == 16000


def test_cli_canonical_synthesis_preserves_prompt_timebase_and_skips_mfa(tmp_path, monkeypatch):
    prompt, target = tmp_path / "prompt.txt", tmp_path / "target.txt"
    for path in (prompt, target):
        path.write_text("あい\na i\n0.2 0.5\n0.5 0.8\n", encoding="utf-8")
    calls = []

    def generate(*args, **kwargs):
        calls.append((args, kwargs))
        return make_baseline()

    monkeypatch.setattr(CtrlSpeech, "from_pretrained", lambda *a, **k: SimpleNamespace(generate=generate))
    out = tmp_path / "out.wav"
    assert main(["--prompt-wav", "prompt.wav", "--prompt-annotation", str(prompt),
                 "--target-annotation", str(target), "--out", str(out)]) == 0
    assert out.exists()
    assert calls[0][1]["align"] is False
    assert calls[0][1]["trim_prompt"] is False


def test_cli_canonical_duration_edit_never_invokes_mfa_verification(tmp_path, monkeypatch, capsys):
    annotation = tmp_path / "clip.txt"
    annotation.write_text("あい\na i\n0.1 0.5\n0.3 0.8\n", encoding="utf-8")
    state, calls = make_baseline(), []

    def from_audio(*args, **kwargs):
        assert kwargs["annotation"]["phones"] == "a i"
        calls.append("adopt")
        return state

    def regenerate(baseline, **kwargs):
        stub_tts().build_edited_controls(baseline, **kwargs)
        calls.append("regenerate")
        return pipeline.Generation(np.zeros(16000), state.gen_f0, state.gen_loud)

    fake = SimpleNamespace(from_audio=from_audio, regenerate=regenerate,
                           align=lambda *a, **k: pytest.fail("invoked MFA"))
    monkeypatch.setattr(CtrlSpeech, "from_pretrained", lambda *a, **k: fake)
    assert main(["--audio", "clip.wav", "--annotation", str(annotation), "--stretch-word", "あ",
                 "--stretch-ratio", "2", "--out", str(tmp_path / "out.wav")]) == 0
    assert calls == ["adopt", "regenerate"]
    assert "achieved duration is unverified" in capsys.readouterr().out


def test_cli_rejects_synthesis_controls_before_loading(monkeypatch):
    monkeypatch.setattr(CtrlSpeech, "from_pretrained", lambda *a, **k: pytest.fail("loaded assets"))
    with pytest.raises(SystemExit):
        main(["--prompt-wav", "prompt.wav", "--prompt-annotation", "prompt.txt",
              "--target-annotation", "target.txt", "--pitch-shift", "2", "--out", "out.wav"])

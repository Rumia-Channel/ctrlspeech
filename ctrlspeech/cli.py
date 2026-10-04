"""Japanese CtrlSpeech: synthesize or edit with canonical OpenJTalk timings.

Plain generation from a voice reference and canonical four-line annotations::

    ctrlspeech --prompt-wav prompt.wav --prompt-annotation prompt.txt \
        --target-annotation target.txt --out baseline.wav

Adopt a recording with canonical timings, then edit pitch or a word duration::

    ctrlspeech --audio clip.wav --annotation clip.txt \
        --pitch-shift 5 --out edited.wav

    ctrlspeech --audio clip.wav --annotation clip.txt \
        --stretch-word 今日 --stretch-ratio 2 --out edited.wav

Annotation lines are transcript / phones / starts / ends. Times use seconds
against the full untrimmed waveform. The canonical phone sequence must match
pyopenjtalk-plus exactly. Raw Japanese MFA phones cannot be used without
explicit reconciliation. A generated waveform needs its own reconciled timing
annotation before a controlled second pass; input timings are not reused.
"""

import argparse
from pathlib import Path

import numpy as np

from .align import annotate_audio, normalize_word, read_four_line_annotation
from .assets import MODELS
from .pipeline import (
    HOP_LENGTH,
    SAMPLE_RATE,
    CtrlSpeech,
    shift_loudness_db,
    shift_pitch_semitones,
)
from .retime import retime_word_curves


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--model", default="japanese-lfm2-350m", choices=sorted(MODELS))
    p.add_argument("--device", default=None, help="cuda, cuda:1, cpu (default: auto)")
    p.add_argument("--out", type=Path, required=True)

    src = p.add_argument_group("baseline from an existing recording")
    src.add_argument("--audio", type=Path, help="Clip to adopt as the baseline.")
    src.add_argument("--transcript", type=Path, help="Text file for --audio.")
    src.add_argument("--transcript-text", help="Inline text for --audio.")
    src.add_argument("--annotation", type=Path,
                     help="Canonical four-line annotation for the full untrimmed --audio; skips MFA.")

    syn = p.add_argument_group("baseline by synthesis")
    syn.add_argument("--prompt-wav", type=Path, help="Voice reference.")
    syn.add_argument("--prompt-text", type=Path, help="Transcript of --prompt-wav.")
    syn.add_argument("--target-wav", type=Path, help="Timing reference for the text.")
    syn.add_argument("--target-text", type=Path, help="Transcript of --target-wav.")
    syn.add_argument("--prompt-annotation", type=Path,
                     help="Canonical four-line annotation timed against the full untrimmed prompt.")
    syn.add_argument("--target-annotation", type=Path,
                     help="Canonical four-line target annotation; replaces --target-wav/--target-text.")

    ctl = p.add_argument_group("controls (applied to the baseline)")
    ctl.add_argument("--pitch-shift", type=float, default=0.0, help="Semitones.")
    ctl.add_argument("--loudness-shift", type=float, default=0.0, help="dB.")
    ctl.add_argument("--stretch-word", help="Word whose duration to change.")
    duration = ctl.add_mutually_exclusive_group()
    duration.add_argument("--stretch-ratio", type=float, help="Multiplier, e.g. 2.0.")
    duration.add_argument("--stretch-seconds", type=float, help="Absolute target length.")

    p.add_argument("--steps", type=int, default=32, help="Flow-matching ODE steps.")
    p.add_argument("--cfg-strength", type=float, default=1.5)
    p.add_argument(
        "--save-baseline", type=Path,
        help="Also write the un-edited baseline audio here.",
    )
    args = p.parse_args(argv)
    has_recording = args.audio is not None
    synthesis_args = (args.prompt_wav, args.prompt_text, args.target_wav, args.target_text,
                      args.prompt_annotation, args.target_annotation)
    if has_recording:
        if any(value is not None for value in synthesis_args):
            p.error("--audio cannot be combined with synthesis arguments")
        sources = (args.transcript, args.transcript_text, args.annotation)
        if sum(value is not None for value in sources) != 1:
            p.error("--audio needs exactly one of --transcript, --transcript-text or --annotation")
    else:
        if args.prompt_wav is None:
            p.error("Pass --audio or --prompt-wav")
        if (args.prompt_text is None) == (args.prompt_annotation is None):
            p.error("--prompt-wav needs exactly one of --prompt-text or --prompt-annotation")
        if args.target_annotation is not None:
            if args.target_wav is not None or args.target_text is not None:
                p.error("--target-annotation replaces --target-wav and --target-text")
        elif args.target_wav is None or args.target_text is None:
            p.error("Supply --target-annotation or both --target-wav and --target-text")
        if args.transcript is not None or args.transcript_text is not None or args.annotation is not None:
            p.error("--transcript, --transcript-text and --annotation are only used with --audio")
    if args.steps < 1:
        p.error("--steps must be positive")
    for name in ("pitch_shift", "loudness_shift", "cfg_strength"):
        if not np.isfinite(getattr(args, name)):
            p.error(f"--{name.replace('_', '-')} must be finite")
    if args.cfg_strength < 0:
        p.error("--cfg-strength must be nonnegative")
    duration_value = args.stretch_seconds if args.stretch_seconds is not None else args.stretch_ratio
    if bool(args.stretch_word) != (duration_value is not None):
        p.error("--stretch-word needs exactly one of --stretch-ratio or --stretch-seconds")
    if duration_value is not None and (not np.isfinite(duration_value) or duration_value <= 0):
        p.error("The requested word duration/ratio must be positive and finite")
    if not has_recording and (args.pitch_shift or args.loudness_shift or args.stretch_word):
        p.error("Synthesis currently supports plain generation only. Reconcile timings for "
                "the generated audio, then edit it using --audio and --annotation.")
    return args


def read_transcript(path, inline, what):
    if inline is not None:
        transcript = " ".join(inline.split())
    elif path is not None:
        transcript = " ".join(Path(path).read_text(encoding="utf-8").split())
    else:
        transcript = ""
    if not transcript:
        raise SystemExit(f"A nonempty transcript is required for {what}.")
    return transcript


def find_word(words, wanted):
    target = normalize_word(wanted)
    for idx, word in enumerate(words):
        if normalize_word(str(word["label"])) == target:
            return idx
    labels = ", ".join(str(w["label"]) for w in words)
    raise SystemExit(f"Word {wanted!r} is not in the utterance. Words: {labels}")


def main(argv=None):
    args = parse_args(argv)

    has_recording = args.audio is not None
    has_synthesis = args.prompt_wav is not None
    if has_recording == has_synthesis:
        raise SystemExit("Pass either --audio or --prompt-wav/--target-wav, not both.")

    wants_control = bool(
        args.pitch_shift or args.loudness_shift or args.stretch_word
    )
    if wants_control and not MODELS[args.model].controllable:
        raise SystemExit(
            f"{args.model} has no prosody embeddings; use a controllable Japanese checkpoint to "
            "apply pitch / loudness / duration edits."
        )

    # Validate cheap user inputs before loading large assets or invoking MFA.
    if has_recording:
        audio_annotation = read_four_line_annotation(args.annotation) if args.annotation else None
        transcript = read_transcript(args.transcript,
                                     audio_annotation["words"] if audio_annotation else args.transcript_text,
                                     "--audio")
    else:
        prompt_annotation = read_four_line_annotation(args.prompt_annotation) if args.prompt_annotation else None
        target_annotation = read_four_line_annotation(args.target_annotation) if args.target_annotation else None
        prompt_text = read_transcript(args.prompt_text, prompt_annotation["words"] if prompt_annotation else None, "--prompt-wav")
        target_text = read_transcript(args.target_text, target_annotation["words"] if target_annotation else None, "target")
    print(f"Loading {args.model} …", flush=True)
    tts = CtrlSpeech.from_pretrained(args.model, device=args.device, progress=True)

    if has_recording:
        print(f"Adopting {args.audio.name} as the baseline …")
        baseline = tts.from_audio(
            args.audio, transcript, steps=args.steps, cfg_strength=args.cfg_strength,
            annotation=audio_annotation,
        )
    else:
        if prompt_annotation is None:
            prompt_annotation = annotate_audio(args.prompt_wav, prompt_text, tts.aligner)
        if target_annotation is None:
            target_annotation = annotate_audio(args.target_wav, target_text, tts.aligner)
        print("Generating the baseline …")
        baseline = tts.generate(
            args.prompt_wav, prompt_annotation, target_annotation,
            steps=args.steps, cfg_strength=args.cfg_strength,
            align=wants_control, trim_prompt=args.prompt_annotation is None,
        )

    print(
        f"Baseline: {len(baseline.gen_np) / SAMPLE_RATE:.2f}s, "
        f"{len(baseline.word_data)} words aligned."
    )
    if args.save_baseline:
        baseline.generation.save(args.save_baseline)
        print(f"  wrote {args.save_baseline}")

    if not wants_control:
        baseline.generation.save(args.out)
        print(f"No control requested; wrote the baseline to {args.out}")
        return 0

    pitch = np.asarray(baseline.gen_f0, dtype=float)
    loudness = np.asarray(baseline.gen_loud, dtype=float)
    phonemes = baseline.gen_phoneme_data

    if args.pitch_shift:
        pitch = shift_pitch_semitones(pitch, args.pitch_shift)
        print(f"Pitch: {args.pitch_shift:+g} semitones")
    if args.loudness_shift:
        loudness = shift_loudness_db(loudness, args.loudness_shift)
        print(f"Loudness: {args.loudness_shift:+g} dB")

    if args.stretch_word:
        if not baseline.word_data:
            raise SystemExit("Duration editing needs canonical phone timings.")
        idx = find_word(baseline.word_data, args.stretch_word)
        word = baseline.word_data[idx]
        old = float(word["end"] - word["start"])
        if args.stretch_seconds is not None:
            new = args.stretch_seconds
        elif args.stretch_ratio is not None:
            new = old * args.stretch_ratio
        else:
            raise SystemExit(
                "--stretch-word needs --stretch-ratio or --stretch-seconds."
            )
        pitch, loudness, phonemes, words, summary = retime_word_curves(
            pitch, loudness, phonemes, baseline.word_data, idx, new,
            hop_length=HOP_LENGTH, sample_rate=SAMPLE_RATE,
        )
        print(
            f"Duration: {summary['label']} {summary['old_duration']:.2f}s -> "
            f"{summary['new_duration']:.2f}s ({summary['scale']:.2f}x), "
            f"timeline now {summary['total_frames']} frames"
        )

    print("Regenerating under the edited controls …")
    result = tts.regenerate(baseline, pitch=pitch, loudness=loudness, phonemes=phonemes)
    result.save(args.out)
    print(f"Wrote {args.out} ({result.duration:.2f}s)")

    if args.stretch_word and args.annotation is not None:
        print("Audio saved; achieved duration is unverified. Canonical annotations bypass MFA; "
              "new output needs independently reconciled timings for verification.")
    elif args.stretch_word:
        try:
            achieved = tts.align(result.audio, baseline.target_words, baseline.target_phones)
            got = achieved["word_data"][idx]
            print(
                f"Verified: {got['label']} lasts {got['end'] - got['start']:.2f}s "
                f"(requested {summary['new_duration']:.2f}s)"
            )
        except (RuntimeError, ValueError, IndexError) as exc:
            print(f"Audio saved; duration verification unavailable: {exc}")
    elif args.pitch_shift:
        def voiced(curve):
            values = np.asarray(curve, dtype=float)
            values = values[values > 0]
            return float(values.mean()) if values.size else 0.0
        print(
            f"Voiced pitch mean: {voiced(baseline.gen_f0):.1f} -> "
            f"{voiced(result.pitch):.1f} bins (requested {voiced(pitch):.1f})"
        )
    elif args.loudness_shift:
        print(
            f"Loudness mean: {np.mean(baseline.gen_loud):.1f} -> "
            f"{np.mean(result.loudness):.1f} bins "
            f"(requested {np.mean(loudness):.1f})"
        )
    return 0

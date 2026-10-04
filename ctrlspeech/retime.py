"""Word-level retiming, shared by the CLI and the interactive editor.

Stretching a word means three things have to move together: the pitch and
loudness curves inside it, its phoneme boundaries, and everything that follows.
Doing that consistently is what makes the model's duration control land on the
right phonemes, so it lives in the package rather than in the demo.

All functions here are pure NumPy/SciPy — no Panel or Bokeh — so they work
headlessly.
"""

import numpy as np
from scipy.interpolate import interp1d as scipy_interp1d

# CtrlSpeech-JA targets utterances up to one minute.  Durations are continuous
# now (no 192-entry lookup table), so this is a safety bound rather than an
# embedding-index limit.
MAX_TARGET_SECONDS = 60
CONTROL_FPS = 100
MAX_TIMELINE_FRAMES = MAX_TARGET_SECONDS * CONTROL_FPS + 1
MAX_DURATION_FRAMES = MAX_TIMELINE_FRAMES - 1


def time_to_frame_index(time_sec, hop_length=256, sample_rate=24000):
    """Round a nonnegative boundary to its nearest frame (ties round up)."""
    if (not np.isfinite(time_sec) or time_sec < 0 or not np.isfinite(hop_length) or
            not np.isfinite(sample_rate) or hop_length <= 0 or sample_rate <= 0):
        raise ValueError("Times and frame rates must be finite with positive rates")
    return int(np.floor(float(time_sec) * sample_rate / hop_length + 0.5 + 1e-9))


def _resample_segment(data, old_n_frames, new_n_frames, method='cubic'):
    """
    Resample a data segment from old_n_frames to new_n_frames using interpolation.
    """
    if old_n_frames == new_n_frames:
        return data.copy()
    if old_n_frames == 0 or new_n_frames == 0:
        return np.zeros(new_n_frames) if new_n_frames > 0 else np.array([])

    x_old = np.linspace(0, 1, old_n_frames)
    x_new = np.linspace(0, 1, new_n_frames)

    if old_n_frames == 1:
        return np.full(new_n_frames, data[0])

    if method == 'nearest':
        f = scipy_interp1d(x_old, data, kind='nearest', fill_value='extrapolate')
    elif method == 'cubic' and old_n_frames >= 4:
        f = scipy_interp1d(x_old, data, kind='cubic', fill_value='extrapolate')
    elif method == 'linear' or old_n_frames < 4:
        f = scipy_interp1d(x_old, data, kind='linear', fill_value='extrapolate')
    else:
        f = scipy_interp1d(x_old, data, kind='nearest', fill_value='extrapolate')

    return f(x_new)


def _retime_waveform_segment(
    waveform,
    waveform_sample_rate,
    start_sec,
    end_sec,
    new_duration,
):
    """Stretch/compress one aligned waveform region and shift later audio.

    Samples outside ``[start_sec, end_sec)`` are copied verbatim.  This mirrors
    :func:`retime_word_curves`: any silence before or after the selected word is
    preserved, while only the samples belonging to the word are resampled.
    """
    waveform = np.asarray(waveform, dtype=float)
    if waveform.ndim != 1:
        raise ValueError("waveform_data must be one-dimensional")
    if not np.isfinite(waveform_sample_rate) or waveform_sample_rate <= 0:
        raise ValueError("waveform_sample_rate must be positive")

    start_sample = int(round(float(start_sec) * waveform_sample_rate))
    end_sample = int(round(float(end_sec) * waveform_sample_rate))
    if start_sample < 0 or end_sample <= start_sample:
        raise ValueError("word boundaries are invalid for waveform retiming")

    # MFA boundaries can exceed the decoded waveform by a fraction of a sample
    # because the two timelines are quantized independently.
    if start_sample >= len(waveform) or end_sample > len(waveform) + 1:
        raise ValueError(
            "word boundaries fall outside waveform_data; pass the waveform "
            "used for this alignment"
        )
    end_sample = min(end_sample, len(waveform))

    old_samples = end_sample - start_sample
    new_samples = int(round(float(new_duration) * waveform_sample_rate))
    if old_samples < 1 or new_samples < 1:
        raise ValueError("word duration must occupy at least one waveform sample")

    stretched = _resample_segment(
        waveform[start_sample:end_sample],
        old_samples,
        new_samples,
        method="linear",
    )
    return np.concatenate(
        [waveform[:start_sample].copy(), stretched, waveform[end_sample:].copy()]
    )


def retime_word_curves(
    pitch,
    loudness,
    phonemes,
    words,
    word_idx,
    new_duration,
    hop_length=256,
    sample_rate=24000,
):
    """Retime one word while preserving every region outside that word.

    ``phonemes`` contains ``[label, start_sec, end_sec]`` rows. ``words``
    contains dictionaries with ``label``, ``start``, ``end``, ``phone_start``
    and ``phone_end`` (exclusive). The selected word and its phone boundaries
    are scaled uniformly; all later phones/words are shifted by the resulting
    delta. Inter-word gaps therefore keep their original duration.

    Pitch uses nearest-neighbour resampling so the ``0 == unvoiced`` sentinel
    is not interpolated into spurious voiced values. Loudness uses linear
    interpolation. Inputs are never mutated.

    Returns
    -------
    new_pitch, new_loudness, new_phonemes, new_words, summary
    """
    pitch = np.asarray(pitch, dtype=float)
    loudness = np.asarray(loudness, dtype=float)
    if pitch.ndim != 1 or loudness.ndim != 1:
        raise ValueError("pitch and loudness must be one-dimensional")
    if len(pitch) != len(loudness):
        raise ValueError(
            f"pitch/loudness length mismatch: {len(pitch)} != {len(loudness)}"
        )
    if not np.all(np.isfinite(pitch)) or not np.all(np.isfinite(loudness)) or not len(pitch):
        raise ValueError("pitch and loudness must be finite and nonempty")
    if not np.isfinite(hop_length) or not np.isfinite(sample_rate) or hop_length <= 0 or sample_rate <= 0:
        raise ValueError("hop_length and sample_rate must be positive")
    if not np.isfinite(new_duration) or float(new_duration) <= 0:
        raise ValueError("new_duration must be a positive finite number")

    new_phonemes = [list(phone) for phone in phonemes]
    new_words = [dict(word) for word in words]
    if not new_words:
        raise ValueError("word_data is empty")
    if not new_phonemes:
        raise ValueError("phoneme_data is required for word duration editing")
    if isinstance(word_idx, (bool, np.bool_)) or not isinstance(word_idx, (int, np.integer)):
        raise TypeError("word_idx must be an integer")
    if not 0 <= word_idx < len(new_words):
        raise IndexError(f"word_idx {word_idx} is out of range")
    word_idx = int(word_idx)

    frame_sec = hop_length / sample_rate
    word = new_words[word_idx]
    required = {"label", "start", "end", "phone_start", "phone_end"}
    missing = required.difference(word)
    if missing:
        raise ValueError(f"word_data[{word_idx}] is missing: {sorted(missing)}")

    word_start = float(word["start"])
    word_end = float(word["end"])
    if not np.isfinite(word_start) or not np.isfinite(word_end) or word_start < 0 or word_end <= word_start:
        raise ValueError(f"word_data[{word_idx}] has invalid boundaries")

    if any(isinstance(word[key], (bool, np.bool_)) or not isinstance(word[key], (int, np.integer))
           for key in ("phone_start", "phone_end")):
        raise ValueError("Word phone ranges must contain integer indices")
    phone_start = int(word["phone_start"])
    phone_end = int(word["phone_end"])
    if not (0 <= phone_start < phone_end <= len(new_phonemes)):
        raise ValueError(
            f"word_data[{word_idx}] has invalid phone range "
            f"[{phone_start}, {phone_end})"
        )

    old_start_frame = time_to_frame_index(word_start, hop_length, sample_rate)
    old_end_frame = time_to_frame_index(word_end, hop_length, sample_rate)
    old_word_frames = old_end_frame - old_start_frame
    if old_word_frames < 1:
        raise ValueError(f"word_data[{word_idx}] occupies fewer than one frame")
    if old_start_frame < 0 or old_end_frame > len(pitch):
        raise ValueError(
            f"word_data[{word_idx}] frame range "
            f"[{old_start_frame}, {old_end_frame}) is outside a {len(pitch)}-frame curve"
        )

    old_duration = word_end - word_start
    # Shift by an integer number of frames so later curves and boundaries move
    # together even when the aligner supplied sub-frame timestamps.
    delta_frames = int(np.floor((float(new_duration) - old_duration) / frame_sec + 0.5 + 1e-9))
    new_word_frames = old_word_frames + delta_frames
    if new_word_frames < 1:
        raise ValueError("new_duration is shorter than one frame")
    new_total_frames = len(pitch) - old_word_frames + new_word_frames
    if new_total_frames > MAX_TIMELINE_FRAMES:
        raise ValueError(
            f"edited timeline would contain {new_total_frames} frames; "
            f"maximum is {MAX_TIMELINE_FRAMES}"
        )

    # The UI and model both operate on the frame grid, so expose the duration
    # that can actually be represented rather than the unquantized request.
    delta = delta_frames * frame_sec
    applied_duration = old_duration + delta
    if applied_duration <= 0:
        raise ValueError("new_duration is shorter than one frame")
    scale = applied_duration / old_duration
    new_word_end = word_end + delta

    # Scale every boundary inside the selected word around its fixed start.
    # Identical neighbouring boundaries remain identical under this transform.
    for phone_idx in range(phone_start, phone_end):
        phone = new_phonemes[phone_idx]
        if len(phone) < 3:
            raise ValueError(f"phoneme_data[{phone_idx}] must have label/start/end")
        start = float(phone[1])
        end = float(phone[2])
        if (not np.isfinite(start) or not np.isfinite(end) or end <= start or
                start < word_start - 1e-6 or end > word_end + 1e-6):
            raise ValueError(f"phoneme_data[{phone_idx}] has invalid boundaries for the selected word")
        phone[1] = word_start + (start - word_start) * scale
        phone[2] = word_start + (end - word_start) * scale

    word["end"] = new_word_end

    # Everything after the word moves as one block. Its content and every
    # inter-word gap are unchanged.
    for phone_idx in range(phone_end, len(new_phonemes)):
        phone = new_phonemes[phone_idx]
        if len(phone) < 3:
            raise ValueError(f"phoneme_data[{phone_idx}] must have label/start/end")
        phone[1] = float(phone[1]) + delta
        phone[2] = float(phone[2]) + delta
    for later_idx in range(word_idx + 1, len(new_words)):
        later = new_words[later_idx]
        later["start"] = float(later["start"]) + delta
        later["end"] = float(later["end"]) + delta

    # Validate every phone, not just the selected word. Zero-frame phones are
    # invalid; the upper bound is now the one-minute timeline safety ceiling,
    # not a lookup-table index.
    previous_end = 0.0
    for phone_idx, phone in enumerate(new_phonemes):
        if len(phone) < 3:
            raise ValueError(f"phoneme_data[{phone_idx}] must have label/start/end")
        start, end = float(phone[1]), float(phone[2])
        if (not np.isfinite(start) or not np.isfinite(end) or
                start < previous_end - 1e-6):
            raise ValueError("Phoneme boundaries must be finite, ordered and within the edited timeline")
        previous_end = end
        start_frame = time_to_frame_index(start, hop_length, sample_rate)
        end_frame = time_to_frame_index(end, hop_length, sample_rate)
        if start_frame < 0 or end_frame > new_total_frames:
            raise ValueError("Phoneme boundaries fall outside the edited control curve")
        duration_frames = end_frame - start_frame
        if not 1 <= duration_frames <= MAX_DURATION_FRAMES:
            raise ValueError(
                f"phoneme_data[{phone_idx}] ({phone[0]}) would last "
                f"{duration_frames} frames; allowed range is "
                f"1..{MAX_DURATION_FRAMES}"
            )

    pitch_word = pitch[old_start_frame:old_end_frame]
    loudness_word = loudness[old_start_frame:old_end_frame]
    new_pitch_word = _resample_segment(
        pitch_word, old_word_frames, new_word_frames, method="nearest"
    )
    new_loudness_word = _resample_segment(
        loudness_word, old_word_frames, new_word_frames, method="linear"
    )
    new_pitch = np.concatenate(
        [pitch[:old_start_frame].copy(), new_pitch_word, pitch[old_end_frame:].copy()]
    )
    new_loudness = np.concatenate(
        [
            loudness[:old_start_frame].copy(),
            new_loudness_word,
            loudness[old_end_frame:].copy(),
        ]
    )

    summary = {
        "word_idx": word_idx,
        "label": str(word["label"]),
        "old_duration": old_duration,
        "new_duration": applied_duration,
        "scale": scale,
        "delta": delta,
        "old_frames": old_word_frames,
        "new_frames": new_word_frames,
        "total_frames": new_total_frames,
    }
    return new_pitch, new_loudness, new_phonemes, new_words, summary



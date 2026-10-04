"""High-level CtrlSpeech inference API.

Two passes, and the second one is the point of the project:

1. **Baseline** — synthesise from a prompt voice and a target text
   (:meth:`CtrlSpeech.generate`), or skip synthesis entirely and adopt a real
   recording as the baseline (:meth:`CtrlSpeech.from_audio`).
2. **Controlled** — read the baseline's frame-level pitch and loudness curves
   and its phoneme boundaries, edit any of them, and resynthesise with the
   edited curves as conditioning (:meth:`CtrlSpeech.regenerate`).

Everything you do not edit is carried over from the baseline, so a pitch edit
leaves loudness and timing alone.
"""

import json
import re
import threading
from dataclasses import dataclass, field
from pathlib import Path

import librosa
import numpy as np
import torch
from omegaconf import OmegaConf

from .align import JapaneseMFAAligner
from .assets import download_assets
from .data import collate_japanese_sequences, encode_japanese_pair
from .features import CosyVoiceSpeakerEmbedding, f0_to_coarse, get_pitch_and_loudness
from .models import DiTar, load_decoder
from .models.backbone import DEFAULT_LFM2_MODEL_ID
from .retime import (
    MAX_DURATION_FRAMES,
    MAX_TARGET_SECONDS,
    MAX_TIMELINE_FRAMES,
    time_to_frame_index,
)

SAMPLE_RATE = 16000
HOP_LENGTH = 160                    # 100 frames/s, matching the *100 in duration_segments
FPS = SAMPLE_RATE // HOP_LENGTH     # = 100
PITCH_BINS = 128
LOUDNESS_BINS = 64
LOUDNESS_DB_RANGE = 60.0            # loudness_db_max - loudness_db_floor
LOUDNESS_BINS_PER_DB = (LOUDNESS_BINS - 1) / LOUDNESS_DB_RANGE   # ~1.05 bin/dB

# Aliases; ctrlspeech.retime owns both limits so they cannot drift apart.
MAX_PHONE_FRAMES = MAX_DURATION_FRAMES
MAX_CONTROL_FRAMES = MAX_TIMELINE_FRAMES

# The SVAE encoder rates [4,4,5,5] give 400 samples per latent -> 40 Hz, and the
# AR model emits patch_size latents per step, so one step is 0.1 s of audio.
VAE_FRAMES_PER_SECOND = 40
AR_PATCH_SIZE = 4
AR_STEP_SECONDS = AR_PATCH_SIZE / VAE_FRAMES_PER_SECOND

DEFAULT_STEPS = 32
DEFAULT_CFG_STRENGTH = 1.5
TRIM_TOP_DB = 40
AR_STOP_MARGIN_STEPS = 20
# Hard one-minute generation ceiling. Stop-margin logic below may use spare
# steps for shorter utterances, but it must never extend the target past 60 s.
MAX_AR_STEPS = int(np.ceil(MAX_TARGET_SECONDS / AR_STEP_SECONDS))


# ─────────────────────────────────────────────────────────────────────────────
# Text / timing helpers
# ─────────────────────────────────────────────────────────────────────────────
def estimate_max_seq_length(target_seconds):
    """Pick an AR step ceiling for utterances up to one minute.

    A small relative margin helps the stop predictor on ordinary utterances,
    while MAX_AR_STEPS remains a hard safety ceiling near 60 seconds.
    """
    target_seconds = float(target_seconds)
    if not np.isfinite(target_seconds) or target_seconds <= 0:
        return 100
    target_seconds = min(target_seconds, float(MAX_TARGET_SECONDS))
    steps = int(np.ceil(target_seconds / AR_STEP_SECONDS * 1.15)) + 10
    return int(np.clip(steps, 100, MAX_AR_STEPS))


def tokenize_phones(prompt_phones, target_phones, text_tokenizer):
    """Join prompt and target phones and map them without silent OOV collapse."""
    text_inputs = prompt_phones + " | " + target_phones
    text_inputs = [re.sub(r"\d", "", t) for t in text_inputs.split(" ") if t]
    unknown = sorted({token for token in text_inputs if token not in text_tokenizer})
    if unknown:
        raise ValueError(
            "Phone vocabulary is missing token(s): " + ", ".join(unknown)
        )
    input_ids = torch.LongTensor([text_tokenizer[token] for token in text_inputs])
    input_ids = input_ids.unsqueeze(0)
    return input_ids, input_ids != 0


def parse_times(times):
    return [float(t) if t != "|" else 0.0 for t in times.split()]


def build_prompt_segments(annotation):
    """Validate an annotation and quantise every boundary on the same grid."""
    phones = annotation["phones"].split()
    starts = annotation["starts"].split()
    ends = annotation["ends"].split()
    if not phones or len(phones) != len(starts) or len(phones) != len(ends):
        raise ValueError("Annotation phone/start/end counts must match and be nonempty")
    segments = []
    previous_end = 0.0
    for phone, start, end in zip(phones, starts, ends):
        if phone == "|":
            if start != "|" or end != "|":
                raise ValueError("Annotation separators must match in every field")
            segments.append((0, 0))
            continue
        start, end = float(start), float(end)
        if (not np.isfinite(start) or not np.isfinite(end) or
                start < previous_end - 1e-6 or end <= start or end > MAX_TARGET_SECONDS):
            raise ValueError("Annotation boundaries must be finite, ordered and within 60 seconds")
        first, last = time_to_frame_index(start, HOP_LENGTH, SAMPLE_RATE), time_to_frame_index(end, HOP_LENGTH, SAMPLE_RATE)
        if not 1 <= last - first <= MAX_PHONE_FRAMES:
            raise ValueError(f"Annotation phone {phone!r} must occupy at least one control frame")
        segments.append((first, last))
        previous_end = end
    return segments


def _finite_curve(values, name, *, max_frames=None):
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or not len(values) or not np.all(np.isfinite(values)):
        raise ValueError(f"{name} must be a nonempty, finite one-dimensional curve")
    if max_frames is not None and len(values) > max_frames:
        raise ValueError(f"{name} exceeds the {max_frames}-frame timeline limit")
    return values


def _validate_sampling(steps, cfg_strength):
    if isinstance(steps, (bool, np.bool_)) or not isinstance(steps, (int, np.integer)) or steps < 1:
        raise ValueError("steps must be a positive integer")
    if not np.isfinite(cfg_strength) or cfg_strength < 0:
        raise ValueError("cfg_strength must be finite and nonnegative")


def _validate_audio(audio):
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim != 1 or not audio.size or not np.all(np.isfinite(audio)):
        raise ValueError("audio must be a nonempty finite mono waveform at 16 kHz")
    return audio


# ─────────────────────────────────────────────────────────────────────────────
# Global prosody edits, applied in coarse-bin space so they compose with drawing
# ─────────────────────────────────────────────────────────────────────────────
def shift_pitch_semitones(coarse, semitones):
    """Transpose a quantised pitch contour; unvoiced frames stay untouched."""
    coarse = _finite_curve(coarse, "pitch")
    if not np.isfinite(semitones):
        raise ValueError("semitones must be finite")
    coarse = np.clip(coarse, 0, PITCH_BINS - 1)
    voiced = coarse > 0
    if not voiced.any() or semitones == 0:
        return coarse.copy()

    mel_min = 1127 * np.log(1 + 65.0 / 700)
    mel_max = 1127 * np.log(1 + 650.0 / 700)
    f0_mel = (coarse - 1) * (mel_max - mel_min) / (PITCH_BINS - 2) + mel_min
    f0 = 700 * (np.exp(f0_mel / 1127) - 1)
    # Extreme shifts saturate at the valid voiced range instead of overflowing.
    f0_shifted = np.clip(f0 * np.exp2(np.clip(semitones / 12.0, -64, 64)), 65.0, 650.0)

    f0_input = np.where(voiced, f0_shifted, 0.0).astype(np.double)
    new_coarse = f0_to_coarse(f0_input.copy())

    out = coarse.copy()
    out[voiced] = new_coarse[voiced]
    return out


def shift_loudness_db(coarse, db):
    """Raise or lower a quantised loudness contour by a number of dB."""
    coarse = _finite_curve(coarse, "loudness")
    if not np.isfinite(db):
        raise ValueError("db must be finite")
    shift = round(float(np.clip(db, -LOUDNESS_DB_RANGE, LOUDNESS_DB_RANGE)) * LOUDNESS_BINS_PER_DB)
    return np.clip(coarse + shift, 0, LOUDNESS_BINS - 1)


# ─────────────────────────────────────────────────────────────────────────────
# Results
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class Generation:
    """Audio plus the prosody the model actually produced."""

    audio: np.ndarray
    pitch: np.ndarray
    loudness: np.ndarray
    sample_rate: int = SAMPLE_RATE

    @property
    def duration(self):
        return len(self.audio) / self.sample_rate

    def save(self, path):
        import soundfile as sf

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(path), np.clip(self.audio, -1.0, 1.0), self.sample_rate,
                 subtype="PCM_16")
        return path


@dataclass
class Baseline:
    """Everything :meth:`CtrlSpeech.regenerate` needs to re-run with edits.

    ``gen_*`` describes the audio being edited (a first-pass generation, or an
    uploaded recording). ``prompt_*`` describes the voice reference, whose
    prosody is prepended to the control curves as context.
    """

    # model input
    input_ids: torch.Tensor
    text_masks: torch.Tensor
    prompt_audio: torch.Tensor
    speaker_emb: torch.Tensor
    prompt_segments: list
    # prompt-side context
    prompt_np: np.ndarray
    prompt_duration: float
    prompt_f0: np.ndarray
    prompt_loud: np.ndarray
    # the editable baseline
    gen_np: np.ndarray
    gen_f0: np.ndarray
    gen_loud: np.ndarray
    gen_phoneme_data: list
    word_data: list
    target_token_refs: list
    target_words: str
    target_phones: str
    # sampling settings used to produce it
    steps: int = DEFAULT_STEPS
    cfg_strength: float = DEFAULT_CFG_STRENGTH
    source: str = "generated"
    extras: dict = field(default_factory=dict)
    linguistic_features: dict | None = None
    native_text_inputs: torch.Tensor | None = None
    native_text_masks: torch.Tensor | None = None

    @property
    def generation(self):
        return Generation(self.gen_np, self.gen_f0, self.gen_loud)


# ─────────────────────────────────────────────────────────────────────────────
# The model
# ─────────────────────────────────────────────────────────────────────────────
class CtrlSpeech:
    """Loaded DiTar + SVAE vocoder + speaker encoder."""

    def __init__(self, model, vocoder, text_tokenizer, speaker_embedding, device,
                 controllable=True, aligner=None, progress=False,
                 frontend=None, native_tokenizer=None):
        self.model = model
        self.vocoder = vocoder
        self.text_tokenizer = text_tokenizer
        self.speaker_embedding = speaker_embedding
        self.device = device
        self.controllable = controllable
        # Off by default: the AR progress bar is noise inside a web app or a
        # notebook. Set it on the instance for a long CLI run.
        self.progress = progress
        self._aligner = aligner
        self.frontend = frontend
        self.native_tokenizer = native_tokenizer
        self._inference_lock = threading.Lock()

    # -- construction ----------------------------------------------------
    @classmethod
    def from_pretrained(cls, model="japanese-lfm2-350m", device=None, repo_id=None,
                        revision=None, aligner=None, progress=False,
                        frontend=None, native_tokenizer=None):
        assets = download_assets(model=model, repo_id=repo_id, revision=revision)
        return cls.from_assets(
            assets, device=device, aligner=aligner, progress=progress,
            frontend=frontend, native_tokenizer=native_tokenizer,
        )

    @classmethod
    def from_assets(cls, assets, device=None, aligner=None, progress=False,
                    frontend=None, native_tokenizer=None):
        device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )

        config = OmegaConf.load(assets.config_path)
        # This branch is intentionally LFM2-only. Published upstream CtrlSpeech
        # Qwen checkpoints are architecture-incompatible and must not be loaded
        # partially with strict=False.
        backbone_name = str(getattr(config.model.backbone, "name", "")).lower()
        if backbone_name not in {"lfm2", "lfm2.5", "lfm2.5-350m"}:
            raise RuntimeError(
                "The japanese branch requires an LFM2.5-350M-trained "
                "CtrlSpeech checkpoint. Upstream Qwen3 checkpoints are not "
                "compatible with this architecture."
            )

        config.model.vocoder.path = str(assets.svae_dir)
        if not getattr(config.model.backbone, "model_id", None):
            config.model.backbone.model_id = DEFAULT_LFM2_MODEL_ID
        # A trained CtrlSpeech-JA checkpoint already contains the complete LFM
        # body, so inference must instantiate the architecture without fetching
        # the original base weights first.
        config.model.backbone.load_pretrained_weights = False

        text_tokenizer = json.loads(assets.vocab_path.read_text(encoding="utf-8"))

        net = DiTar(config.model)
        state_dict = _load_state_dict(assets.weights_path)
        state_dict = {k.removeprefix("model."): v for k, v in state_dict.items()}
        # The SVAE encoder is instantiated from metadata too: every model
        # weight must be present, including generator.*, before inference.
        result = net.load_state_dict(state_dict, strict=False)
        _check_control_weights(result.missing_keys, assets.spec)
        _check_checkpoint_keys(result.missing_keys, result.unexpected_keys)
        tokenizer_dir = assets.root / "shared" / "tokenizer"
        if native_tokenizer is None and tokenizer_dir.is_dir():
            from transformers import AutoTokenizer

            native_tokenizer = AutoTokenizer.from_pretrained(
                str(tokenizer_dir), local_files_only=True,
            )
        net = net.to(device).eval()

        vocoder = load_decoder(local_path=str(assets.svae_dir)).to(device).eval()
        speaker_embedding = CosyVoiceSpeakerEmbedding(
            campplus_model=str(assets.campplus_path)
        )
        return cls(
            model=net,
            vocoder=vocoder,
            text_tokenizer=text_tokenizer,
            speaker_embedding=speaker_embedding,
            device=device,
            controllable=assets.spec.controllable,
            aligner=aligner,
            progress=progress,
            frontend=frontend,
            native_tokenizer=native_tokenizer,
        )

    @property
    def aligner(self):
        if self._aligner is None:
            self._aligner = JapaneseMFAAligner(frontend=self.frontend)
        return self._aligner

    def align(self, waveform, transcript, expected_phones=None):
        """Force-align audio against its transcript (needs MFA installed)."""
        if isinstance(waveform, np.ndarray):
            waveform = torch.from_numpy(waveform).float()
        aligner = self.aligner
        # Its expected-phone contract uses the raw MFA inventory, not OpenJTalk.
        raw_expected = None if isinstance(aligner, JapaneseMFAAligner) else expected_phones
        result = aligner.align(waveform, transcript, raw_expected)
        if expected_phones is not None:
            _require_canonical_phones(result, expected_phones.split(), "alignment")
        return result

    def _encode_text_pair(self, prompt_text, target_text):
        sequence = encode_japanese_pair(
            prompt_text, target_text, frontend=self.frontend,
            phone_vocab=self.text_tokenizer, native_tokenizer=self.native_tokenizer,
        )
        return sequence, collate_japanese_sequences([sequence])

    # -- pass 1 ----------------------------------------------------------
    @torch.no_grad()
    def generate(self, prompt_wav, prompt_annotation, target_annotation,
                 steps=DEFAULT_STEPS, cfg_strength=DEFAULT_CFG_STRENGTH,
                 align=True, trim_prompt=False):
        """Synthesise ``target_annotation``'s text in ``prompt_wav``'s voice.

        Both annotations are 4-line dicts (words / phones / starts / ends); use
        :func:`ctrlspeech.align.read_four_line_annotation` to load them.

        Prompt times refer to the full waveform unless ``trim_prompt=True``;
        set that only for annotations produced against a trimmed reference.

        ``align=False`` skips the MFA pass, which makes the result unusable for
        controlled regeneration but avoids the dependency for plain synthesis.
        """
        _validate_sampling(steps, cfg_strength)
        sequence, text_batch = self._encode_text_pair(
            prompt_annotation["words"], target_annotation["words"],
        )
        _require_canonical_phones(prompt_annotation, sequence.prompt.phones, "prompt annotation")
        _require_canonical_phones(target_annotation, sequence.target.phones, "target annotation")
        prompt_segments = build_prompt_segments(prompt_annotation)
        target_segments = build_prompt_segments(target_annotation)
        input_ids, text_masks = text_batch["input_ids"], text_batch["text_masks"]

        prompt_np, _ = librosa.load(str(prompt_wav), sr=SAMPLE_RATE, mono=True)
        prompt_np = _validate_audio(prompt_np)
        if trim_prompt:
            prompt_np, _ = librosa.effects.trim(prompt_np, top_db=TRIM_TOP_DB)
            prompt_np = _validate_audio(prompt_np)
        prompt_duration = prompt_np.shape[-1] / SAMPLE_RATE
        prompt_audio = torch.from_numpy(prompt_np).unsqueeze(0).float()
        speaker_emb = self.speaker_embedding._extract_spk_embedding(prompt_audio)[0]

        _, prompt_f0, _, prompt_loud = get_pitch_and_loudness(
            prompt_np, sr=SAMPLE_RATE, hop_length=HOP_LENGTH
        )

        if max(end for _, end in prompt_segments) > len(prompt_f0):
            raise ValueError("Prompt annotation extends past the prompt audio")
        prompt_offset = len(prompt_f0)
        duration_segments = prompt_segments + [(0, 0)] + [
            (prompt_offset + start, prompt_offset + end)
            for start, end in target_segments
        ]

        gen_np = self._sample(
            prompt_audio=prompt_audio,
            speaker_emb=speaker_emb,
            input_ids=input_ids,
            text_masks=text_masks,
            duration_segments=duration_segments,
            pitch=prompt_f0,
            loudness=prompt_loud,
            max_seq_length=estimate_max_seq_length(
                max(parse_times(target_annotation["ends"]))
            ),
            steps=steps,
            cfg_strength=cfg_strength,
            linguistic_features=text_batch["linguistic_features"],
            native_text_inputs=text_batch.get("native_text_inputs"),
            native_text_masks=text_batch.get("native_text_masks"),
        )
        _, gen_f0, _, gen_loud = get_pitch_and_loudness(
            gen_np, sr=SAMPLE_RATE, hop_length=HOP_LENGTH
        )

        if align:
            alignment = self.align(
                gen_np, target_annotation["words"],
                expected_phones=target_annotation["phones"],
            )
        else:
            alignment = {"phoneme_data": [], "word_data": [], "target_token_refs": []}

        return Baseline(
            input_ids=input_ids,
            text_masks=text_masks,
            prompt_audio=prompt_audio,
            speaker_emb=speaker_emb,
            prompt_segments=prompt_segments,
            prompt_np=prompt_np,
            prompt_duration=prompt_duration,
            prompt_f0=prompt_f0,
            prompt_loud=prompt_loud,
            gen_np=gen_np,
            gen_f0=gen_f0,
            gen_loud=gen_loud,
            gen_phoneme_data=alignment["phoneme_data"],
            word_data=alignment["word_data"],
            target_token_refs=alignment["target_token_refs"],
            target_words=target_annotation["words"],
            target_phones=target_annotation["phones"],
            steps=steps,
            cfg_strength=cfg_strength,
            linguistic_features=text_batch["linguistic_features"],
            native_text_inputs=text_batch.get("native_text_inputs"),
            native_text_masks=text_batch.get("native_text_masks"),
        )

    # -- pass 1, alternative: adopt a recording as the baseline -----------
    @torch.no_grad()
    def from_audio(self, audio, transcript, steps=DEFAULT_STEPS,
                   cfg_strength=DEFAULT_CFG_STRENGTH, min_seconds=0.5,
                   max_seconds=10.0, annotation=None):
        """Use a real recording as the baseline, skipping the first synthesis.

        The clip supplies the voice *and* all of the baseline prosody. Editing
        one dimension reuses the clip's own values for the others, so the result
        has the same shape as :meth:`generate` and the caller need not care
        which path produced it.

        ``audio`` is a path or a mono float array already at 16 kHz.
        An optional canonical four-line ``annotation`` bypasses MFA. Its times
        refer to the full recording: silence is not trimmed in this mode.
        """
        _validate_sampling(steps, cfg_strength)
        if (not np.isfinite(min_seconds) or not np.isfinite(max_seconds) or
                not 0 < min_seconds <= max_seconds <= MAX_TARGET_SECONDS):
            raise ValueError("Audio duration limits must satisfy 0 < min_seconds <= max_seconds <= 60")
        transcript = " ".join((transcript or "").split())
        if not transcript:
            raise ValueError("A transcript of the audio is required.")

        if isinstance(audio, (str, Path)):
            audio_np, _ = librosa.load(str(audio), sr=SAMPLE_RATE, mono=True)
        else:
            audio_np = np.asarray(audio, dtype=np.float32)
        audio_np = _validate_audio(audio_np)
        if annotation is None:
            audio_np, _ = librosa.effects.trim(audio_np, top_db=TRIM_TOP_DB)
            audio_np = _validate_audio(audio_np)

        duration = audio_np.shape[-1] / SAMPLE_RATE
        if duration < min_seconds:
            raise ValueError(
                f"The audio is only {duration:.2f}s after preprocessing; "
                f"at least {min_seconds:g}s is required."
            )
        if duration > max_seconds:
            raise ValueError(
                f"The audio is {duration:.2f}s; at most {max_seconds:g}s is "
                "supported. Trim the clip and try again."
            )

        sequence, text_batch = self._encode_text_pair(transcript, transcript)
        audio_t = torch.from_numpy(audio_np).unsqueeze(0).float()
        speaker_emb = self.speaker_embedding._extract_spk_embedding(audio_t)[0]
        _, f0, _, loud = get_pitch_and_loudness(
            audio_np, sr=SAMPLE_RATE, hop_length=HOP_LENGTH
        )

        # One alignment supplies both the phone tokens and the duration segments,
        # so the prompt and target token streams line up by construction.
        if annotation is None:
            alignment = self.align(audio_t, transcript, " ".join(sequence.target.phones))
        else:
            if " ".join(annotation["words"].split()) != transcript:
                raise ValueError("Annotation transcript does not match the audio transcript")
            alignment = _canonical_annotation_alignment(annotation, sequence.target)
        prompt_segments = build_prompt_segments(alignment)
        input_ids, text_masks = text_batch["input_ids"], text_batch["text_masks"]

        last_end_frame = max(
            (time_to_frame_index(float(item[2]), HOP_LENGTH, SAMPLE_RATE) for item in alignment["phoneme_data"]),
            default=0,
        )
        if last_end_frame > len(f0):
            raise RuntimeError(
                "Canonical alignment extends past the extracted control curve "
                f"({last_end_frame} > {len(f0)} frames)."
            )

        return Baseline(
            input_ids=input_ids,
            text_masks=text_masks,
            prompt_audio=audio_t,
            speaker_emb=speaker_emb,
            prompt_segments=prompt_segments,
            prompt_np=audio_np,
            prompt_duration=duration,
            prompt_f0=f0,
            prompt_loud=loud,
            gen_np=audio_np,
            gen_f0=f0,
            gen_loud=loud,
            gen_phoneme_data=alignment["phoneme_data"],
            word_data=alignment["word_data"],
            target_token_refs=alignment["target_token_refs"],
            target_words=transcript,
            target_phones=alignment["phones"],
            steps=steps,
            cfg_strength=cfg_strength,
            source="upload",
            extras={"canonical_annotation": annotation is not None},
            linguistic_features=text_batch["linguistic_features"],
            native_text_inputs=text_batch.get("native_text_inputs"),
            native_text_masks=text_batch.get("native_text_masks"),
        )

    # -- pass 2 ----------------------------------------------------------
    def build_edited_controls(self, baseline, pitch=None, loudness=None,
                              phonemes=None):
        """Quantise the edits and rebuild token-aligned duration segments."""
        pitch = baseline.gen_f0 if pitch is None else pitch
        loudness = baseline.gen_loud if loudness is None else loudness
        phonemes = baseline.gen_phoneme_data if phonemes is None else phonemes

        pitch = _finite_curve(pitch, "Edited pitch", max_frames=MAX_CONTROL_FRAMES)
        loudness = _finite_curve(loudness, "Edited loudness", max_frames=MAX_CONTROL_FRAMES)
        prompt_f0 = _finite_curve(baseline.prompt_f0, "Prompt pitch")
        prompt_loud = _finite_curve(baseline.prompt_loud, "Prompt loudness")
        if len(prompt_f0) != len(prompt_loud):
            raise ValueError("Prompt pitch and loudness lengths do not match")
        edited_f0 = np.clip(np.rint(pitch), 0, PITCH_BINS - 1).astype(np.int32)
        edited_loud = np.clip(np.rint(loudness), 0, LOUDNESS_BINS - 1).astype(np.int32)
        if len(edited_f0) != len(edited_loud):
            raise ValueError("Edited pitch and loudness lengths do not match")
        if phonemes is None or not baseline.gen_phoneme_data or not baseline.target_token_refs:
            raise ValueError("Regeneration needs a canonical phoneme alignment; generate with align=True")
        if len(phonemes) != len(baseline.gen_phoneme_data):
            raise ValueError("Edited phoneme count changed")

        # Rebuild the target token segments from the (possibly retimed) phoneme
        # boundaries. The prompt offset matches the curve actually concatenated
        # below, so the two never drift apart.
        duration_segments = list(baseline.prompt_segments) + [(0, 0)]
        prompt_offset = len(baseline.prompt_f0)
        previous_end = 0
        for phone_ref in baseline.target_token_refs:
            if phone_ref is None:
                duration_segments.append((0, 0))
                continue
            if not isinstance(phone_ref, (int, np.integer)) or not 0 <= phone_ref < len(phonemes):
                raise ValueError("Target token contains an invalid phoneme reference")
            item = phonemes[phone_ref]
            if len(item) != 3 or item[0] != baseline.gen_phoneme_data[phone_ref][0]:
                raise ValueError("Edited phoneme labels/order changed")
            if not np.isfinite(float(item[1])) or not np.isfinite(float(item[2])):
                raise ValueError("Edited phoneme boundaries must be finite")
            start = time_to_frame_index(float(item[1]), HOP_LENGTH, SAMPLE_RATE)
            end = time_to_frame_index(float(item[2]), HOP_LENGTH, SAMPLE_RATE)
            if start < previous_end:
                raise ValueError("Edited phoneme boundaries overlap or are out of order")
            previous_end = end
            phone_frames = end - start
            if not 1 <= phone_frames <= MAX_PHONE_FRAMES:
                raise ValueError(
                    f"phoneme {item[0]!r} lasts {phone_frames} frames, outside the "
                    f"allowed range 1..{MAX_PHONE_FRAMES}"
                )
            if start < 0 or end > len(edited_f0):
                raise ValueError(
                    f"phoneme {item[0]!r} boundary [{start}, {end}) falls outside "
                    f"the control curve [0, {len(edited_f0)})"
                )
            duration_segments.append((prompt_offset + start, prompt_offset + end))

        if len(duration_segments) != baseline.input_ids.shape[1]:
            raise ValueError(
                "Duration-token count does not match text-token count: "
                f"{len(duration_segments)} vs {baseline.input_ids.shape[1]}"
            )

        control_f0 = np.concatenate([prompt_f0, edited_f0], axis=0)
        control_loud = np.concatenate([prompt_loud, edited_loud], axis=0)
        return duration_segments, control_f0, control_loud

    @torch.no_grad()
    def regenerate(self, baseline, pitch=None, loudness=None, phonemes=None,
                   steps=None, cfg_strength=None):
        """Resynthesise the baseline under edited control curves."""
        if not self.controllable:
            raise RuntimeError(
                "This checkpoint has no pitch / loudness / duration embeddings; "
                "load a controllable Japanese checkpoint to edit prosody."
            )
        duration_segments, control_f0, control_loud = self.build_edited_controls(
            baseline, pitch, loudness, phonemes
        )
        gen_np = self._sample(
            prompt_audio=baseline.prompt_audio,
            speaker_emb=baseline.speaker_emb,
            input_ids=baseline.input_ids,
            text_masks=baseline.text_masks,
            duration_segments=duration_segments,
            pitch=control_f0,
            loudness=control_loud,
            max_seq_length=estimate_max_seq_length(
                (len(control_f0) - len(baseline.prompt_f0)) / FPS
            ),
            steps=baseline.steps if steps is None else steps,
            cfg_strength=(
                baseline.cfg_strength if cfg_strength is None else cfg_strength
            ),
            linguistic_features=baseline.linguistic_features,
            native_text_inputs=baseline.native_text_inputs,
            native_text_masks=baseline.native_text_masks,
        )
        _, gen_f0, _, gen_loud = get_pitch_and_loudness(
            gen_np, sr=SAMPLE_RATE, hop_length=HOP_LENGTH
        )
        return Generation(gen_np, gen_f0, gen_loud)

    # -- shared sampling -------------------------------------------------
    def _sample(
        self,
        prompt_audio,
        speaker_emb,
        input_ids,
        text_masks,
        duration_segments,
        pitch,
        loudness,
        max_seq_length,
        steps,
        cfg_strength,
        *,
        linguistic_features=None,
        native_text_inputs=None,
        native_text_masks=None,
    ):
        _validate_sampling(steps, cfg_strength)
        if not self.controllable:
            # Base checkpoints never learned the prosody embeddings; feeding them
            # would add randomly initialised vectors to every text token.
            duration_segments = pitch = loudness = None

        with self._inference_lock:
            sampled = self.model.sample(
                prompt_audio=prompt_audio.to(self.device),
                speaker_embs=speaker_emb.unsqueeze(0).to(self.device),
                text_inputs=input_ids.to(self.device),
                text_masks=text_masks.to(self.device),
                duration_segments=None if duration_segments is None else [duration_segments],
                pitch=None if pitch is None else [pitch],
                loudness=None if loudness is None else [loudness],
                linguistic_features=(
                    None if linguistic_features is None else
                    {name: value.to(self.device) for name, value in linguistic_features.items()}
                ),
                native_text_inputs=(
                    None
                    if native_text_inputs is None
                    else native_text_inputs.to(self.device)
                ),
                native_text_masks=(
                    None
                    if native_text_masks is None
                    else native_text_masks.to(self.device)
                ),
                use_cache=True,
                max_seq_length=max_seq_length,
                steps=steps,
                cfg_strength=cfg_strength,
                progress=self.progress,
            )
            audio = self.vocoder.decode(sampled.float()).squeeze(0).detach().cpu()
            return audio.squeeze().numpy()


# ─────────────────────────────────────────────────────────────────────────────
# Loading helpers
# ─────────────────────────────────────────────────────────────────────────────
def _load_state_dict(path):
    path = Path(path)
    if path.suffix == ".safetensors":
        from safetensors.torch import load_file

        return load_file(str(path))
    checkpoint = torch.load(str(path), map_location="cpu", weights_only=True)
    return checkpoint.get("state_dict", checkpoint)


_CONTROL_KEYS = (
    "pitch_embedding",
    "loudness_embedding",
    "duration_conditioner",
)


def _check_control_weights(missing, spec):
    """Fail loudly if a control checkpoint arrived without its prosody weights.

    Because the load has to be non-strict, a truncated or mismatched file would
    otherwise leave the prosody embeddings randomly initialised — the model
    would still run and just quietly ignore every control curve.
    """
    if not spec.controllable:
        return
    absent = [key for key in _CONTROL_KEYS if any(key in name for name in missing)]
    if absent:
        raise RuntimeError(
            f"{spec.key} is a controllable checkpoint but these weights are "
            f"missing: {', '.join(absent)}. The weights file looks incomplete."
        )


def _require_canonical_phones(annotation, expected, source):
    if annotation["phones"].split() != list(expected):
        raise ValueError(
            f"{source} phones do not match the canonical OpenJTalk sequence. "
            "Raw Japanese MFA phones/word separators cannot be used as model "
            "tokens; provide explicitly reconciled OpenJTalk timings."
        )


def _check_checkpoint_keys(missing, unexpected):
    if missing or unexpected:
        raise RuntimeError(
            "Incompatible or incomplete CtrlSpeech-JA checkpoint: "
            f"missing={missing[:8]}, unexpected={unexpected[:8]}"
        )


def _canonical_annotation_alignment(annotation, encoding):
    """Attach editor metadata to supplied timings, without estimating any time."""
    _require_canonical_phones(annotation, encoding.phones, "audio annotation")
    build_prompt_segments(annotation)
    phones = [
        [label, float(start), float(end)]
        for label, start, end in zip(annotation["phones"].split(),
                                     annotation["starts"].split(), annotation["ends"].split())
    ]
    words = []
    features = encoding.result.phone_features
    for index, morpheme in enumerate(encoding.result.morphemes):
        indices = [i for i, feature in enumerate(features) if feature.morpheme_index == index]
        if not indices or morpheme.is_symbol:
            continue
        if indices != list(range(indices[0], indices[-1] + 1)):
            raise ValueError("Morpheme phone ownership must be contiguous for duration editing")
        words.append(dict(label=morpheme.surface, start=phones[indices[0]][1],
                          end=phones[indices[-1]][2], phone_start=indices[0],
                          phone_end=indices[-1] + 1))
    return dict(annotation, phoneme_data=phones, word_data=words,
                target_token_refs=list(range(len(phones))))

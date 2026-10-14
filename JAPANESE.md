# CtrlSpeech-JA / LFM2.5-350M

The `japanese` branch is a Japanese-specific CtrlSpeech redesign. It is not
binary-compatible with the upstream Qwen3 CtrlSpeech checkpoints.

## Target architecture

The autoregressive backbone is fixed to the unmodified Hugging Face
`LiquidAI/LFM2.5-350M-Base` architecture:

- hidden size: 1024
- 16 layers
- 10 LIV convolution layers
- 6 full GQA layers
- 16 query heads / 8 KV heads
- native mixed convolution + KV cache
- no vendored Qwen implementation
- no RWKV/TICA hybridization

CtrlSpeech-specific information is injected before LFM2 through separate phone,
modality and Japanese linguistic conditioners. The LFM2 source itself is not
forked.

The local model config is `configs/japanese-lfm2-350m.yaml`.

## Japanese frontend

Japanese preprocessing uses `pyopenjtalk-plus` (distribution name) through the
compatible `import pyopenjtalk` module name.

The production frontend prefers `g2p_mapping_prosody()` when available. The
released `pyopenjtalk-plus==0.4.1.post9` API exposes `g2p_mapping()` instead,
so the compatibility path combines its morpheme mapping with OpenJTalk
full-context labels. Both paths retain:

- canonical OpenJTalk phone sequence
- morpheme -> phone mapping
- MeCab unknown-word status
- input character spans
- NJD accent nucleus and mora count
- phone-level Low/High accent trajectory
- accent phrase start/end
- pause / interrogative / exclamatory boundaries

The model keeps linguistic pitch separate from measured/edited F0. OpenJTalk
Low/High is a linguistic prior; CtrlSpeech pitch control remains an explicit
acoustic control.

Canonical phone IDs live in:

- `ctrlspeech/frontend/japanese_phones.py`
- `configs/japanese-phone-vocab.json`

Unknown readings are representable as `unk` for diagnostics, but
`encode_japanese_text()` rejects them by default so malformed readings do not
silently enter training data.

## Model-ready text preprocessing

`ctrlspeech.data` provides:

```python
from transformers import AutoTokenizer
from ctrlspeech.data import (
    collate_japanese_sequences,
    encode_japanese_pair,
)

tokenizer = AutoTokenizer.from_pretrained("LiquidAI/LFM2.5-350M-Base")
sequence = encode_japanese_pair(
    "これは参照音声です。",
    "今日は良い天気ですね。",
    native_tokenizer=tokenizer,
)
batch = collate_japanese_sequences([sequence])
```

The resulting sequence contains two complementary inputs: native LFM Japanese
tokens as a semantic prefix, and canonical OpenJTalk phone IDs with phone-aligned
linguistic features. The phone stream remains authoritative for pronunciation.
The explicit `|` prompt/target separator has linguistic conditioning disabled.

## Prosody controls

The existing CtrlSpeech pitch and loudness controls remain discrete.

Duration no longer uses the upstream 192-entry embedding table. It is now a
continuous log-compressed MLP conditioner, so there is no 191-frame embedding
index ceiling. A zero-length segment is an explicit "conditioning unavailable"
sentinel and contributes exactly zero pitch/loudness/duration conditioning.

Japanese linguistic conditioning currently includes:

- OpenJTalk Low / High category
- accent phrase start / end
- accent nucleus
- accent phrase mora count

These features are additive conditions on the phone embeddings and are
independent from explicit F0/loudness/duration controls.

## One-minute target

The branch targets utterances up to 60 seconds:

- control timeline: 6001 points at 100 Hz
- acoustic AR step: 0.08 s (four 50 Hz SARA latents, native 48 kHz output)
- hard AR safety ceiling: 750 steps
- continuous duration safety limit: 6000 control frames

The LocDiT sampler is intentionally left architecturally unchanged for the
first LFM2 baseline. Quality should be established at the original sampling
settings before NFE reduction/distillation is evaluated.

## Environment

The project uses uv. Conda is not required for the CtrlSpeech Python
environment.

```bash
uv python install 3.11
uv sync --extra cpu --extra japanese --group dev
CTRLSPEECH_TEST_ISOLATE_ONNX=1 uv run --extra cpu --extra japanese pytest
```

CI runs CPU-only Torch/TorchAudio tests on Windows and Linux with the native
ONNX runtime explicitly isolated. Coverage includes real released-API text-only
frontend checks, tiny-model training/backward/resume, and stubbed Panel control
flows. ONNX inference and synthesis quality remain outside these checks. See the
CPU-only environment command in README.md.

The mutually exclusive `cpu`, `cuda`, `rocm` and `xpu` extras select the
Torch/TorchAudio binary profile at sync time. See [README.md](README.md#install)
for wheel versions, Windows/Linux driver prerequisites and an explicit-GPU
execution check. Training and inference share auto selection and explicit
`cpu`, `cuda[:N]`, `rocm[:N]` and `xpu[:N]` devices. ROCm uses the CUDA API;
XPU uses its own AMP, scaler and RNG APIs. Arc devices lacking FP64 should use
BF16 or FP32 training, not FP16 GradScaler.


Montreal Forced Aligner remains an external command-line dependency because its
Kaldi runtime is not self-contained through PyPI on every platform.

Once an `mfa` executable is on PATH:

```bash
mfa model download acoustic japanese_mfa
mfa model download dictionary japanese_mfa
mfa model download g2p japanese_mfa
```

MFA is a timing estimator. OpenJTalk/pyopenjtalk-plus remains the canonical
linguistic phone/accent source. Raw `japanese_mfa` phones use a different
inventory; the inference pipeline rejects mismatched annotations instead of
assigning their timings to unrelated phones. Until reconciliation is implemented,
use the canonical-annotation CLI described in README.md, or inject an aligner
that returns exactly that phone sequence. Installing MFA alone does not complete
this integration. Explicit annotations preserve the original untrimmed waveform
time base; generated-waveform timings are never inferred from a target template.

## Checkpoint status

There is currently no released CtrlSpeech-JA LFM2 checkpoint.

The upstream `control-600m`, `control-150m`, `base-600m` and
`base-150m` checkpoints were trained with the Qwen-based architecture and are
intentionally rejected by this branch rather than partially loaded with
`strict=False`.

For a locally trained Japanese checkpoint, point `CTRLSPEECH_ASSETS` at the
asset tree. After a checkpoint is published, `CTRLSPEECH_HF_REPO` or the
`repo_id` argument can select it.

## Remaining training work

The cached-feature dataset, collator and staged trainer are implemented. See
[TRAINING.md](TRAINING.md) for the feature schema, audit command, freeze/unfreeze
recipe, 5 -> 15 -> 30 -> 60 second curriculum and epoch-boundary resume.
The trainer supports mixed precision, gradient checkpointing, target-only
supervision and speaker-disjoint validation across CPU, CUDA/ROCm and XPU.
Checkpoint format 3 stores the selected accelerator's RNG and requires the
same backend/device index for resume. Formats 1/2 and old 40 Hz feature caches
are incompatible; inference state dictionaries remain backend-independent.

The SARA-based waveform codec is an independent `svae` workspace package under
`packages/svae`, with `svae-train` and `svae-export` commands. It uses a frozen
W2v-BERT 2.0 semantic branch, residual CNN plus two-layer unidirectional LSTM,
and native HiFi-GAN, with native 48 kHz audio and 50 Hz / 128-D latents by default. It trains from
WAVs without transcripts using reconstruction/KL and native multi-period/multi-band
GAN objectives, not the old semantic-distillation loss. Frozen SSL is used during
reference encoding too; decoder-only inference does not allocate it.
The 24 kHz / 64-D paper design is extended by doubling the first acoustic stride
and final decoder stride, retaining the 50 Hz clock. Latent size is configurable;
codec, TTS input/output dimensions and feature metadata must agree. Wider-band
quality requires appropriate source recordings; resampling low-rate WAVs does
not restore missing high frequencies.
Exports include the SSL weights and load offline. Codec training does not import
CtrlSpeech. Unpublished width/decoder presets are local choices, not official
checkpoint compatibility or reproduced paper quality. See [TRAINING.md](TRAINING.md)
for independent uv commands and CPU verification limits; large-model speech
quality and AMD/Arc hardware performance remain unverified. Regenerate features
and retrain TTS when migrating from the previous Semantic-VAE.


Producing a usable Japanese checkpoint still requires real corpus preparation
and quality evaluation:

1. OpenJTalk-to-MFA duration reconciliation and preprocessing QC.
2. Audio/F0/loudness/speaker/SARA feature preprocessing.
3. Training and tuning the supplied staged recipe on the prepared corpus.
4. Evaluation of CER, speaker similarity, accent/F0 control, duration accuracy,
   long-form drift, RTF and VRAM.
5. LocDiT NFE reduction only after the quality baseline is fixed.

## License note

Repository source code remains under its repository license. LFM2.5 pretrained
weights have their own model license; redistribution of a derived checkpoint
must follow the applicable LFM model terms separately from this source tree.

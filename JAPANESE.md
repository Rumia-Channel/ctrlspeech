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

The production frontend prefers `g2p_mapping_prosody()`, retaining:

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
from ctrlspeech.data import (
    encode_japanese_text,
    join_prompt_target,
    collate_japanese_sequences,
)

prompt = encode_japanese_text("これは参照音声です。")
target = encode_japanese_text("今日は良い天気ですね。")
sequence = join_prompt_target(prompt, target)
batch = collate_japanese_sequences([sequence])
```

The resulting sequence contains canonical phone IDs and phone-aligned Japanese
linguistic features. The explicit `|` prompt/target separator has linguistic
conditioning disabled.

## Prosody controls

The existing CtrlSpeech pitch and loudness controls remain discrete.

Duration no longer uses the upstream 192-entry embedding table. It is now a
continuous log-compressed MLP conditioner, so there is no 191-frame embedding
index ceiling.

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
- acoustic AR step: approximately 0.1 s
- hard AR safety ceiling: 620 steps
- continuous duration safety limit: 6000 control frames

The LocDiT sampler is intentionally left architecturally unchanged for the
first LFM2 baseline. Quality should be established at the original sampling
settings before NFE reduction/distillation is evaluated.

## Environment

The project uses uv. Conda is not required for the CtrlSpeech Python
environment.

```bash
uv python install 3.11
uv sync --extra japanese --group dev
uv run pytest
```

CI runs the same uv setup on Windows and Linux.

Montreal Forced Aligner remains an external command-line dependency because its
Kaldi runtime is not self-contained through PyPI on every platform.

Once an `mfa` executable is on PATH:

```bash
mfa model download acoustic japanese_mfa
mfa model download dictionary japanese_mfa
mfa model download g2p japanese_mfa
```

MFA is a timing estimator. OpenJTalk/pyopenjtalk-plus remains the canonical
linguistic phone/accent source.

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

The inference/model/frontend architecture is now LFM2-native, but producing a
usable Japanese checkpoint still requires the actual Japanese training
pipeline and data:

1. OpenJTalk-to-MFA duration reconciliation and preprocessing QC.
2. Audio/F0/loudness/speaker/SVAE feature preprocessing.
3. Long-form Japanese dataset/collator integration.
4. Staged fine-tuning of new speech adapters followed by the LFM2 body.
5. 5 -> 15 -> 30 -> 60 second curriculum.
6. Evaluation of CER, speaker similarity, accent/F0 control, duration accuracy,
   long-form drift, RTF and VRAM.
7. LocDiT NFE reduction only after the quality baseline is fixed.

## License note

Repository source code remains under its repository license. LFM2.5 pretrained
weights have their own model license; redistribution of a derived checkpoint
must follow the applicable LFM model terms separately from this source tree.

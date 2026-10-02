# CtrlSpeech: Coarse-to-Fine Control for Expressive Speech Synthesis.

<p align="left">
  <a href="https://arxiv.org/abs/2608.08362"><img src="https://img.shields.io/badge/arXiv-Paper-b31b1b?logo=arxiv&logoColor=white" alt="arXiv"></a>
  <a href="https://huggingface.co/zhisheng01/CtrlSpeech"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Models-ffcc00" alt="Hugging Face"></a>
  <a href="https://zhishengzheng.com/ctrlspeech/"><img src="https://img.shields.io/badge/Demo-Samples-1f8acb" alt="Demo"></a>
</p>


> **Japanese development branch:** this branch replaces the upstream Qwen3 AR
> backbone with the unmodified `LiquidAI/LFM2.5-350M-Base` architecture and
> uses `pyopenjtalk-plus` for Japanese linguistic/prosody preprocessing.
> Upstream CtrlSpeech checkpoints are intentionally incompatible. There is no
> released Japanese LFM2 checkpoint yet; use `main` for the original published
> models. See [JAPANESE.md](JAPANESE.md) for implementation status.

CtrlSpeech is a zero-shot TTS model you can *steer after the fact*. Generate a
sentence, read back its pitch contour, loudness contour and phoneme boundaries,
change one of them, and resynthesise — the model follows the edit and leaves
everything else alone.

The prosody is conditioned per phoneme token, not per utterance, which is why a
single word can be stretched to twice its length while the rest of the sentence
keeps its original timing.

```
prompt voice ─┐
              ├─► DiTar (AR + LocDiT flow matching) ─► SVAE latents ─► waveform
target text ──┘         ▲
                        │  per-token pitch (128 bins) / loudness (64 bins) /
                        │  duration (frames) embeddings
                        └── edited by you between pass 1 and pass 2
```

---

## Install

This branch uses **uv** for Python and dependency management. Python 3.11 is
recommended; the package requires Python 3.10 or newer.

```bash
uv python install 3.11
uv sync --extra japanese --group dev
```

Use `uv run` for commands so they always execute inside the project
environment.

**Montreal Forced Aligner** is required for anything that derives phoneme
boundaries from audio (duration editing, and adopting your own recording as a
baseline). MFA is treated as an external system command rather than a uv
dependency because its Kaldi runtime is not self-contained through PyPI on all
platforms. Once an `mfa` executable is available on PATH:

```bash
mfa model download acoustic japanese_mfa
mfa model download dictionary japanese_mfa
mfa model download g2p japanese_mfa
```

MFA is only needed for audio/text forced alignment. The canonical Japanese
linguistic representation comes from pyopenjtalk-plus, not from MFA.

There is no public Japanese checkpoint yet. For locally trained assets, set
`CTRLSPEECH_ASSETS`; after publishing a checkpoint, set
`CTRLSPEECH_HF_REPO` or pass `repo_id`.

---

## Development quick start

Inspect Japanese text and phone-aligned accent features:

```python
from ctrlspeech.frontend import JapaneseFrontend

frontend = JapaneseFrontend()
result = frontend.analyze("今日は良い天気ですね。")

print(result.phone_string)
for feature in result.phone_features:
    print(
        feature.phone,
        feature.pitch,
        feature.accent_phrase_index,
        feature.accent_nucleus,
        feature.phrase_mora_count,
    )
```

Build the exact prompt/target sequence consumed by the LFM2 AR backbone:

```python
from ctrlspeech.data import (
    collate_japanese_sequences,
    encode_japanese_text,
    join_prompt_target,
)

prompt = encode_japanese_text("これは参照音声です。")
target = encode_japanese_text("今日は良い天気ですね。")
sequence = join_prompt_target(prompt, target)
batch = collate_japanese_sequences([sequence])

print(batch["input_ids"].shape)
print(batch["linguistic_features"]["accent_pitch"].shape)
```

Run the current test suite:

```bash
uv run pytest
```

The Panel/demo inference path requires a trained
`japanese-lfm2-350m` checkpoint and intentionally does not fall back to the
upstream Qwen checkpoints.

For cached-feature training, staged LFM2 fine-tuning, dataset auditing and
checkpoint resume, see [TRAINING.md](TRAINING.md) and
[`configs/japanese-training.yaml`](configs/japanese-training.yaml).

---

## Models

| Model | Backbone | Status |
|---|---|---|
| `japanese-lfm2-350m` | LFM2.5-350M Base, 10 LIV Conv + 6 GQA | architecture implemented; checkpoint not yet released |

The `japanese` branch rejects upstream Qwen checkpoints rather than loading
them partially. The LFM2 body is loaded from pretrained weights for training;
a finished CtrlSpeech-JA checkpoint contains the complete fine-tuned body.

---

## How the control works

Audio is analysed at **100 frames per second** (16 kHz, hop 160).

- **Pitch** — F0 mapped to 128 mel-spaced bins; bin 0 means unvoiced. Slider
  shifts skip unvoiced frames so silence is not given a pitch.
- **Loudness** — A-weighted dB in 64 bins, about 1.05 bins per dB.
- **Duration** — per-phone frame counts are log-compressed by a continuous MLP
  conditioner; there is no 192-entry lookup-table ceiling.
- **Japanese linguistic accent** — pyopenjtalk-plus Low/High, accent-phrase
  boundaries, accent nucleus and phrase mora count are separate conditions from
  measured/edited F0.
- **Long form** — the current safety target is **60 s** / 6001 control frames,
  with at most 620 AR steps.

Stretching a word rescales its phoneme boundaries uniformly and shifts
everything after it, so inter-word pauses keep their original length.

The AR model emits `patch_size=4` SVAE latents per step at 40 Hz, i.e. **0.1 s
per step**. The step budget is derived from the requested duration
(`estimate_max_seq_length`) rather than fixed, so a stretched sentence is not
truncated.

---

## Environment variables

| Variable | Purpose |
|---|---|
| `CTRLSPEECH_ASSETS` | Use a local asset directory; skips all downloads |
| `CTRLSPEECH_HF_REPO` | Override the Hub repo id |
| `CTRLSPEECH_MFA_CACHE` | Where MFA scratch files go (default `~/.cache/ctrlspeech/mfa`) |
| `CTRLSPEECH_MFA_DICT` | Path to `english_us_arpa.dict` |
| `CTRLSPEECH_DEMO_MODEL` | Which model the demo loads (default `japanese-lfm2-350m`) |

MFA alignment uses a **sentence-level mini dictionary** built on the fly, and
falls back to the full 200k-word lexicon only when a word is out of vocabulary.
That is the difference between ~15 s and ~3 min per alignment; the fallback also
needs about 500 MB of scratch space, so point `CTRLSPEECH_MFA_CACHE` at a disk
with room.

---

## Repository layout

```
ctrlspeech/
  pipeline.py    CtrlSpeech: generate / from_audio / regenerate
  assets.py      Hub resolution for weights and support files
  retime.py      word-level retiming; owns the duration limits
  cli.py         the `ctrlspeech` command
  align/         MFA wrapper, 4-line annotation format
  features/      pitch, loudness and speaker-embedding extraction
  models/        DiTar, LFM2 backbone, LocDiT, SVAE/DAC vocoder
  training/      cached features, length curriculum, staged trainer and CLI
demo/
  app.py             Panel application
  interactive_plot.py Bokeh contour and phoneme editor
  assets/            bundled example clips + examples.json
scripts/
  generate.py        CLI entry point for a checkout
tests/
  test_demo_flow.py  headless check of all three control paths
```

Adding a demo example means dropping a wav plus a plain-text transcript into
`demo/assets/` and listing it in `examples.json`; the phoneme timings are
force-aligned on first use and cached next to the clip.

---

## Licence

The code in this repository is **MIT** (see [LICENSE](LICENSE)). The published **weights are CC BY-NC 4.0 — non-commercial**. 


### Intended use

These models clone a speaker's voice from a few seconds of reference audio. Use
only recordings you have the right to use, disclose synthetic speech as
synthetic, and do not impersonate anyone without their consent.

## Citation
```
@inproceedings{zheng2026ctrlspeech,
  title     = {CtrlSpeech: Coarse-to-Fine Control for Expressive Speech Synthesis},
  author    = {Zheng, Zhisheng and Sun, Xiaohang and Liu, Zhu and Chen, Caren and Kumar, Rohith and Aggarwal, Manoj and Medioni, Gerard and Harwath, David},
  year      = {2026},
  booktitle = {{Interspeech 2026}},
}
```

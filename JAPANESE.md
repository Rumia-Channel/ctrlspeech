# Japanese CtrlSpeech work

This branch is the staging area for a Japanese-specific CtrlSpeech model.

## Environment

This branch uses uv as the project/environment manager. Conda is not required
for the CtrlSpeech Python environment.

The Japanese frontend uses the pyopenjtalk-plus distribution. Its import name
remains pyopenjtalk, so the Python code intentionally still uses
`import pyopenjtalk`.

Recommended setup:

```bash
uv python install 3.11
uv sync --extra japanese
uv run pytest
```

The repository targets Python 3.10 or newer. Python 3.11 is the recommended
development version.

## Current status

The first foundation layer is present:

- `ctrlspeech.frontend.JapaneseFrontend`
  - Unicode normalization
  - OpenJTalk G2P via pyopenjtalk-plus
  - NJD accent/mora metadata
  - OpenJTalk full-context labels
  - diagnostic whitespace segmentation for Japanese text
- `ctrlspeech.align.JapaneseMFAAligner`
  - japanese_mfa acoustic model
  - japanese_mfa pronunciation dictionary
  - japanese_mfa G2P for OOV words
  - MFA's Japanese/Sudachi tokenizer for forced alignment

### Montreal Forced Aligner

MFA is intentionally treated as an external command-line dependency rather than
part of the uv environment. The PyPI package does not by itself provide a
self-contained Kaldi runtime on every platform, so CtrlSpeech only requires that
an `mfa` executable is available on PATH.

Once MFA is installed by a method appropriate for the host system, install the
Japanese models:

```bash
mfa model download acoustic japanese_mfa
mfa model download dictionary japanese_mfa
mfa model download g2p japanese_mfa
```

Inspect Japanese text:

```python
from ctrlspeech.frontend import JapaneseFrontend

ja = JapaneseFrontend()
x = ja.analyze("今日はいい天気ですね。")
print(x.phone_string)
print(x.mfa_transcript)
for token in x.morphemes:
    print(token.surface, token.accent, token.mora_size, token.chain_flag)
```

Prepare an aligner:

```python
from ctrlspeech.align import JapaneseMFAAligner

aligner = JapaneseMFAAligner()
```

`JapaneseMFAAligner` keeps the original transcript for MFA. MFA 3.x performs
Japanese morphological tokenization itself and can invoke japanese_mfa G2P for
out-of-vocabulary words. OpenJTalk segmentation is retained only as a diagnostic
view for the future model frontend.

## Important limitation

The released CtrlSpeech checkpoints are not Japanese checkpoints. Adding a
Japanese frontend does not make the existing phone tokenizer or DiTAR weights
understand Japanese. In particular, OpenJTalk's phone inventory and the
Japanese MFA phone inventory are different.

The next implementation milestone is therefore:

1. choose the canonical Japanese model phone inventory;
2. build a lossless-enough OpenJTalk/MFA normalization layer with alignment
   boundary handling;
3. build Japanese dataset preprocessing (audio, phone durations, F0, loudness,
   speaker embeddings and SVAE latents);
4. add the training dataset/collator/loss loop;
5. train a Japanese tokenizer/checkpoint before enabling Japanese inference in
   the public CLI.

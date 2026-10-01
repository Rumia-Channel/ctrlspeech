# Japanese CtrlSpeech work

This branch is the staging area for a Japanese-specific CtrlSpeech model.

## Current status

The first foundation layer is present:

- ctrlspeech.frontend.JapaneseFrontend
  - Unicode normalization
  - OpenJTalk G2P
  - NJD accent/mora metadata
  - OpenJTalk full-context labels
  - whitespace segmentation for Japanese MFA
- ctrlspeech.align.JapaneseMFAAligner
  - japanese_mfa acoustic model
  - japanese_mfa pronunciation dictionary
  - automatic OpenJTalk-based word segmentation before MFA

Install the Japanese frontend dependency with:

    pip install -e '.[japanese]'

Install the MFA models separately:

    mfa model download acoustic japanese_mfa
    mfa model download dictionary japanese_mfa

Inspect Japanese text:

    from ctrlspeech.frontend import JapaneseFrontend

    ja = JapaneseFrontend()
    x = ja.analyze("今日はいい天気ですね。")
    print(x.phone_string)
    print(x.mfa_transcript)
    for token in x.morphemes:
        print(token.surface, token.accent, token.mora_size, token.chain_flag)

Prepare an aligner:

    from ctrlspeech.align import JapaneseMFAAligner

    aligner = JapaneseMFAAligner()

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

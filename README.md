# CtrlSpeech-JA：日本語音声の抑揚・音量・長さを制御する

<p align="left">
  <a href="https://arxiv.org/abs/2608.08362"><img src="https://img.shields.io/badge/arXiv-Paper-b31b1b?logo=arxiv&logoColor=white" alt="原論文"></a>
  <a href="https://huggingface.co/zhisheng01/CtrlSpeech"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Models-ffcc00" alt="オリジナル版のモデル"></a>
  <a href="https://zhishengzheng.com/ctrlspeech/"><img src="https://img.shields.io/badge/Demo-Samples-1f8acb" alt="オリジナル版の音声サンプル"></a>
</p>

参照音声の声質を使って音声を合成し、 **声の高さ（Pitch）・音量（Loudness）・発音の長さ（Duration）** を編集して再生成するプロジェクトです。合成音声だけでなく、自分の録音を編集前の基準音声（ベースライン）として使えます。制御は音素単位で与えます。

> **この日本語開発版の状態**
>
> - 自己回帰モデルを Qwen3 から **`LiquidAI/LFM2.5-350M-Base`** に置き換え、日本語解析には **`pyopenjtalk-plus`** を使います。
> - 音声コーデックは SARA ベースです。標準仕様は **48 kHz 音声・50 Hz / 128 次元の潜在特徴量**です。
> - **日本語の学習済みチェックポイントは未公開です。インストールだけでは音声生成できません。** 学習済みモデルと補助アセットを自分で用意する必要があります。
> - 上流の Qwen 版や旧 16 kHz / 40 Hz 版のチェックポイントとは互換性がありません。公開済みのオリジナルモデルを使う場合は `main` を参照してください。上のモデル・サンプルへのリンクもオリジナル版のものです。
> - 日本語 MFA の音素と OpenJTalk の音素には違いがあります。**MFA を入れるだけでは日本語の自動アラインメントは完成しません。** 現状は、確認済みの OpenJTalk 音素時刻を指定する方法を使ってください。

## 目次

- [できることと基本の流れ](#overview)
- [動作環境とインストール](#install)
- [学習済みモデルとアセットの準備](#assets)
- [音素時刻ファイルの準備](#annotations)
- [Web UI の起動と各操作](#web-ui)
- [CLI での生成・編集・保存](#cli)
- [Python からの日本語解析](#python)
- [学習する場合](#training)
- [制御の仕様と上限](#controls)
- [環境変数](#environment)
- [開発時の確認](#development)
- [よくある問題](#troubleshooting)
- [ディレクトリ構成](#layout)
- [ライセンスと利用上の注意](#license)
- [引用](#citation)

<a id="overview"></a>
## できることと基本の流れ

1. **ベースラインを用意する**：参照音声から合成するか、自分の録音をそのまま読み込みます。
2. **制御を編集する**：声の高さや音量の曲線を変更する、または特定の単語を伸縮します。
3. **再生成して比較する**：編集した制御を条件として音声を作り、元音声・要求した制御・実際の結果を比較します。

編集しなかった制御にはベースラインの値を使います。ただし、これは生成モデルへの条件指定であり、出力波形や実測値が指定どおりに完全一致する保証ではありません。

Web UI は一度に一種類の制御を編集する構成です。CLI では録音に対して高さ・音量・一単語の長さを組み合わせて指定できます。いずれも、日本語モデルと正しい音素時刻が前提です。

<a id="install"></a>
## 動作環境とインストール

### 1. uv と Python を用意する

Python と依存関係の管理には [uv](https://docs.astral.sh/uv/getting-started/installation/) を使います。**uv 0.12.1 以上**が必要です。Python は 3.11 を推奨し、パッケージの要求は 3.10 以上です。

以下はリポジトリのルートで実行します。複数行のシェル継続記号を使わない例なので、PowerShell でもそのまま実行できます。

```sh
uv --version
uv python install 3.11
```

### 2. 実行バックエンドを一つ選ぶ

次の **いずれか一つだけ**を実行してください。`japanese` は日本語解析、`dev` は開発用の pytest を追加します。

| 実行環境 | インストールコマンド |
|---|---|
| CPU | `uv sync --frozen --extra cpu --extra japanese --group dev` |
| NVIDIA GPU / CUDA | `uv sync --frozen --extra cuda --extra japanese --group dev` |
| AMD GPU / ROCm | `uv sync --frozen --extra rocm --extra japanese --group dev` |
| Intel GPU / XPU | `uv sync --frozen --extra xpu --extra japanese --group dev` |

`cpu`・`cuda`・`rocm`・`xpu` は排他的です。**`--all-extras` は使わないでください。**

以降のコマンド例は CPU を使います。GPU を使う場合は `--extra cpu` を選択したものに置き換えてください。`uv run` にもバックエンドと `japanese` の指定を残します。同期済みの環境をそのまま使う場合は、代わりに `uv run --no-sync ...` を使えます。extra のない通常の `uv sync` / `uv run` ではバックエンドを選択できません。

### GPU とデバイス指定

| プロファイル | Torch / TorchAudio |
|---|---|
| CPU | 両方 2.11.0 |
| CUDA | 両方 2.11.0、CUDA 13.0 用 |
| XPU | 両方 2.11.0 |
| ROCm | Torch 2.14.0+rocm10.1.0 / TorchAudio 2.11.0.3+rocm10.1.0 |

ROCm は [AMD が案内するバイナリの組](https://rocm.docs.amd.com/projects/ai-ecosystem/en/latest/frameworks/pytorch/install.html)を使い、ROCm SDK と `device-all` のカーネル群を含むためダウンロードが大きくなります。Linux / Windows 間の依存メタデータの違いは uv 設定で扱います。Torch と TorchAudio を独立に入れ替えないでください。

GPU 用プロファイルには x86-64 の Windows / Linux 向け wheel がありますが、対応 GPU・OS・ドライバーが別途必要です。**uv は GPU ドライバーをインストールしません。** [AMD の対応表](https://rocm.docs.amd.com/en/docs-10.1.0/compatibility/compatibility-matrix.html)と [Intel GPU の前提条件](https://docs.pytorch.org/docs/2.11/notes/get_start_xpu.html)を確認してください。

学習・推論の `--device` は共通です。

| 指定 | 動作 |
|---|---|
| `auto` | CUDA / ROCm → XPU → CPU の順で選択 |
| `cpu` | CPU を使用 |
| `cuda` / `cuda:0` | CUDA API の GPU を使用 |
| `rocm` / `rocm:0` | HIP ビルドであることも確認して AMD GPU を使用 |
| `xpu` / `xpu:0` | Intel GPU を使用 |

明示したデバイスが利用できない場合や GPU 番号が不正な場合は、CPU に黙って切り替えずエラーになります。ROCm は内部では PyTorch の CUDA API を使います。

サンプリングとコーデックは標準の Torch を使い、NVIDIA 専用の畳み込み・融合ボコーダー拡張を必須としません。SSL 用 log-mel、高さ・音量、外部の話者埋め込み・アラインメントの前処理は CPU、凍結 W2v-BERT は選択したデバイスで実行します。

### MFA を使う場合

[Montreal Forced Aligner（MFA）](https://montreal-forced-aligner.readthedocs.io/)は、音声と書き起こしから音素の時刻を推定する外部コマンドです。Kaldi ランタイムが全環境で PyPI から完結しないため、uv の依存関係には含めていません。

MFA 側の手順で導入し、`mfa` が PATH から実行できる状態で次を実行します。

```sh
mfa model download acoustic japanese_mfa
mfa model download dictionary japanese_mfa
mfa model download g2p japanese_mfa
```

**日本語の正規の音素列は MFA ではなく OpenJTalk が決めます。** `japanese_mfa` の生の音素時刻はそのまま使えません。確認済みの音素時刻ファイルを明示する CLI / アップロード経路では、MFA の実行を省略できます。

<a id="assets"></a>
## 学習済みモデルとアセットの準備

対応モデル名は `japanese-lfm2-350m` です。構造は実装済みですが、学習済み重みは未公開です。学習方法は [TRAINING.md](TRAINING.md)、日本語版の設計と残作業は [JAPANESE.md](JAPANESE.md) を参照してください。

ローカルで用意したアセットは、次の構成で配置します。

```text
ctrlspeech-hf/
├── japanese-lfm2-350m/
│   ├── config.yaml
│   └── model.safetensors
├── svae/
│   ├── metainfo.json
│   ├── encoder.safetensors
│   ├── weights.safetensors
│   └── ema.safetensors
└── shared/
    ├── vocab.json
    ├── campplus.onnx
    └── tokenizer/          # 任意：ローカルの LFM トークナイザー
```

- `model.safetensors` の代わりに、対応形式の `model.ckpt` も読み込めます。学習用チェックポイントだけでなく、設定・コーデック・音素語彙・話者モデルも必要です。
- SARA のエンコーダーとデコーダーは同じ学習・出力ペアを使います。古いコーデックのファイルを混ぜないでください。
- LFM のネイティブトークンを使う場合は `shared/tokenizer/` を同梱するか、Python API の `native_tokenizer` で渡します。トークナイザーは暗黙にはダウンロードしません。

PowerShell：

```powershell
$env:CTRLSPEECH_ASSETS = "C:/path/to/ctrlspeech-hf"
```

Bash：

```bash
export CTRLSPEECH_ASSETS="/path/to/ctrlspeech-hf"
```

この変数には、モデルのサブフォルダーではなく **`japanese-lfm2-350m`・`svae`・`shared` を含む親ディレクトリ**を指定します。指定時はアセットの自動ダウンロードを行いません。

将来自分のチェックポイントを Hugging Face Hub に公開した場合は `CTRLSPEECH_HF_REPO`、または Python API の `repo_id` で取得先を指定できます。現状の日本語版に既定の公開チェックポイント取得先はありません。

<a id="annotations"></a>
## 音素時刻ファイルの準備

日本語版では、**音声と照合済みの正規 OpenJTalk 音素時刻**を使ってください。ファイルは UTF-8 の `.txt` で、項目名を付けず、次の **4 行**にします。

| 行 | 内容 |
|---|---|
| 1 | 元の日本語の書き起こし。句読点も含める |
| 2 | 正規 OpenJTalk 音素を半角スペースで区切った列。休止音素も含める |
| 3 | 各音素の開始時刻。秒単位、半角スペース区切り |
| 4 | 各音素の終了時刻。秒単位、半角スペース区切り |

準備時の条件：

- 音素列は `JapaneseFrontend().analyze(text).phones` と完全一致させます。[Python の解析例](#python)で列を確認できますが、**テキスト解析だけでは時刻は得られません。**
- 各音素に開始・終了を一つずつ与え、正の長さで、100 Hz の制御フレーム上でも最低 1 フレームを占めるようにします。
- 時刻は順序どおりで重複せず、その録音の範囲内に収めます。
- 明示指定した時刻は、先頭無音を切り落とす前の**元音声全体**を基準とします。先頭無音があれば、その分の時刻も保持してください。
- 単語の区切りとして `|` を挿入したり、MFA 音素の名前だけを置き換えたり、時刻を推測したり、別の録音の時刻を流用したりしないでください。

CLI の `--target-annotation` は、これから合成する対象文の要求タイムラインです。対象の WAV は不要ですが、**生成結果の実際の音素時刻を示すものではありません。** 生成結果をさらに編集する場合は、その出力音声について新たに時刻を取得・確認してください。

確認済み時刻を指定する経路では、再生成後に MFA で長さを自動検証しません。要求した長さと実際の長さが一致したかを測るには、出力音声に対応する正規アライナーが別途必要です。

<a id="web-ui"></a>
## Web UI の起動と各操作

### 起動・終了

アセットの環境変数を設定したターミナルで実行します。

```sh
uv run --extra cpu --extra japanese panel serve demo/app.py --show --port 5006
```

ブラウザーが自動で開かない場合は `http://localhost:5006/app` にアクセスします。終了はターミナルで `Ctrl+C` です。

GPU を固定する場合は `CTRLSPEECH_DEMO_DEVICE` も設定します。たとえば XPU なら、PowerShell で `$env:CTRLSPEECH_DEMO_DEVICE = "xpu"` としたうえで、起動コマンドの extra も `xpu` に変えてください。

SSH 経由では、手元で `ssh -L 5006:localhost:5006 host` を実行し、サーバー側で次のように起動できます。

```sh
uv run --extra cpu --extra japanese panel serve demo/app.py --port 5006 --allow-websocket-origin=localhost:5006
```

画面のボタン名は英語です。以下は実際の表示名に合わせた操作説明です。

### 1. ベースラインを用意する

#### 自分の録音を使う：現在の日本語版での利用経路

1. `Source` で **`Upload your own audio`** を選びます。
2. 音声ファイルを選びます。選択可能な拡張子は `.wav`・`.mp3`・`.flac`・`.ogg`・`.m4a` です。実際に読める形式は環境のデコーダーにも依存するため、迷ったら WAV を使ってください。
3. `Transcript · exactly the words spoken in the clip` に、実際に発話した日本語を入力します。
4. `Canonical OpenJTalk annotation (.txt)` に[確認済みの 4 行ファイル](#annotations)を指定します。書き起こし欄が空なら、ファイルの 1 行目で補います。欄にも入力する場合は内容を一致させてください。
5. `Control to edit` で **`Pitch`・`Loudness`・`Duration`** のいずれかを選びます。
6. **`Use uploaded audio as baseline`** を押します。

アップロード音声は **0.5～10 秒**が対象です。48 kHz のモノラルへ読み込み、録音そのものを声質参照とベースラインの両方に使うため、最初の音声合成は行いません。明示した音素時刻がある場合は、先頭無音を自動で切り落としません。

#### 登録済みの例を使う

日本語の例がある場合は `Bundled example` → `Example` → `Control to edit` を選び、**`Generate baseline & align`** を押します。

ただし、現在同梱の `Dreams` は日本語の例として登録されていないため、一覧には出ません。日本語の例を追加する場合は `demo/assets/` に参照・対象の WAV と書き起こしを置き、`examples.json` に `"language": "ja"` を含むエントリーを登録します。**この経路の自動処理には、正規 OpenJTalk 音素を返せるアライナーが必要です。標準の生 MFA 出力だけでは動きません。**

入力元や例を切り替えると、それまでのベースライン・編集・比較結果はクリアされます。編集種類を変えるときは、種類を選び直してからベースラインの読み込み／生成ボタンを押し直してください。

### 2-A. 声の高さを変える：Pitch

**全体を上下させる場合**：`Pitch shift (semitones)` を動かして **`Apply`** を押します。範囲は **−12～+12 半音、1 半音刻み**です。正の値で高く、負の値で低くなります。無声音のフレームはスライダー操作では変更しません。

**曲線を直接描く場合**：

1. **`Draw Pitch`** を押して描画モードにします。
2. グラフ上をドラッグして曲線を描きます。
3. **`Finish Pitch`** を押して確定します。
4. 続けて描くときは、もう一度 `Draw Pitch` を押します。

横軸は秒、縦軸は量子化されたピッチの bin です。Hz や半音そのものではありません。`Apply` は元のベースラインに対するシフトで、押すたびに累積する操作ではなく、手描き曲線も上書きします。**`Reset`** でベースラインに戻せます。

### 2-B. 音量を変える：Loudness

`Loudness shift (dB)` で変更量を選び、**`Apply`** を押します。範囲は **−20～+20 dB、1 dB 刻み**です。正の値で大きく、負の値で小さくなります。

曲線を直接編集する場合は **`Draw Loudness` → グラフをドラッグ → `Finish Loudness`** の順です。縦軸は loudness の bin で、dB の数値をそのまま表示しているわけではありません。

こちらも `Apply` は元のベースラインからの変更で、手描きや前回の適用結果を置き換えます。**`Reset`** で元に戻します。

### 2-C. 単語の長さを変える：Duration

1. **`Word`** で対象の単語を選びます。日本語ではフロントエンドが解析した形態素が単位です。
2. **`Ratio`** の `0.75×`・`1×`・`1.5×`・`2×`、または **`Target duration (s)`** で要求する秒数を選びます。
3. 微調整は **`−0.10 s`・`−0.05 s`・`+0.05 s`・`+0.10 s`** で行います。値は 0.01 秒の制御グリッドに合わせられます。
4. 画面の許容範囲と変更前後の秒数を確認し、**`Apply`** を押します。

対象単語内の音素を比例伸縮し、後ろの単語をまとめて移動します。単語間の休止時間は保持します。許容範囲外の倍率は表示されず、秒数も許容範囲内に制限されます。

**UI では一度に一単語だけの長さ変更です。** 次の `Apply` は元のベースラインからやり直し、前の長さ変更を置き換えます。複数単語への累積編集にはなりません。`Reset` はタイムライン全体を元に戻します。波形と単語トラックは確認用で、この画面では音素境界をドラッグして編集する操作ではありません。

### グラフの移動・拡大

- 描画モードでないときのドラッグは表示範囲の移動です。曲線が変わらずグラフが動く場合は、先に `Draw Pitch` / `Draw Loudness` を押してください。
- ツールバーのパン・ホイールズームで表示範囲を調整できます。Duration のトラックは横方向の移動・拡大です。
- **グラフのツールバーのリセットは表示範囲を戻す操作**です。編集値を破棄するには、制御パネル側の **`Reset`** を使います。

### 3. 再生成・再生・比較・保存

1. スライダー／秒数なら `Apply`、手描きなら `Finish Pitch` / `Finish Loudness` で反映・確定します。
2. **`Regenerate with current control`** を押します。UI の生成設定は ODE 32 ステップ、CFG 強度 1.5 です。
3. `Baseline` と **`Controlled · Second generation`** の音声プレーヤーで聴き比べます。
4. 比較グラフの `Baseline`（元）・`Requested`（要求）・`Achieved`（実測）を確認します。高さ・音量では実測曲線も表示します。

確認済み音素時刻をアップロードした Duration 編集では、出力の実測時刻を自動取得しないため、**`achieved duration unverified`** と表示されます。要求秒数の表示だけで達成を確認したことにはなりません。

UI には専用の WAV 保存ボタンや編集状態の保存・復元機能はありません。確実にファイルへ出力したい場合は、次の CLI の `--out` を使ってください。

<a id="cli"></a>
## CLI での生成・編集・保存

アセットの環境変数を設定し、各録音に対応する確認済み音素時刻を用意してから実行します。`reference.wav` などは手元のファイル名に置き換えてください。

```sh
uv run --extra cpu --extra japanese ctrlspeech --help
```

### 参照音声から一度合成する

`reference.txt` は参照音声の音素時刻、`target.txt` は対象文と要求タイムラインの 4 行ファイルです。

```sh
uv run --extra cpu --extra japanese ctrlspeech --prompt-wav reference.wav --prompt-annotation reference.txt --target-annotation target.txt --out generated.wav
```

この生成コマンドに高さ・音量・長さの編集オプションは併用できません。生成結果を編集するには、`generated.wav` 自身の正しい音素時刻を新たに用意し、次の録音編集の形式で実行します。

### 録音の高さ・音量を変える

```sh
# 声を 2 半音高くする
uv run --extra cpu --extra japanese ctrlspeech --audio recording.wav --annotation recording.txt --pitch-shift 2 --out higher.wav

# 音量の制御値を 3 dB 下げる
uv run --extra cpu --extra japanese ctrlspeech --audio recording.wav --annotation recording.txt --loudness-shift -3 --out quieter.wav

# 二つの制御を組み合わせ、編集前の音声も別に保存する
uv run --extra cpu --extra japanese ctrlspeech --audio recording.wav --annotation recording.txt --pitch-shift 2 --loudness-shift -3 --save-baseline baseline.wav --out edited.wav
```

### 単語を伸ばす・縮める

録音の解析結果に形態素ラベル `猫` が存在する場合の例です。

```sh
# 元の長さの 1.2 倍にする
uv run --extra cpu --extra japanese ctrlspeech --audio recording.wav --annotation recording.txt --stretch-word 猫 --stretch-ratio 1.2 --out stretched.wav

# 対象単語を 0.4 秒にする
uv run --extra cpu --extra japanese ctrlspeech --audio recording.wav --annotation recording.txt --stretch-word 猫 --stretch-seconds 0.4 --out duration.wav
```

`--stretch-ratio` と `--stretch-seconds` はどちらか一方だけ指定します。対象語が複数回ある場合は、正規化したラベルに最初に一致するものを編集します。見つからない場合は、エラーに候補ラベルが表示されます。

### オプション一覧

| オプション | 用途・条件 |
|---|---|
| `--model` | モデル名。既定は `japanese-lfm2-350m` |
| `--device` | 実行先。既定は `auto` |
| `--out` | 出力 WAV のパス。必須。親フォルダーは自動作成 |
| `--audio` | ベースラインにする録音 |
| `--annotation` | `--audio` の確認済み 4 行ファイル。MFA を省略 |
| `--transcript` / `--transcript-text` | 録音の書き起こしをファイル／文字列で指定。`--annotation` とは併用不可。自動アライナーが必要 |
| `--prompt-wav` | 合成時の声質参照音声。`--audio` とは併用不可 |
| `--prompt-annotation` / `--prompt-text` | 参照音声の 4 行ファイル／書き起こしファイル。どちらか一つ。後者は自動アライナーが必要 |
| `--target-annotation` | 対象文の 4 行ファイル。`--target-wav` / `--target-text` の代わりに指定 |
| `--target-wav` と `--target-text` | 対象の時刻参照音声と書き起こし。両方必要。自動アライナーが必要 |
| `--pitch-shift` | 半音単位の高さ変更。既定 0 |
| `--loudness-shift` | dB 単位の音量変更。既定 0 |
| `--stretch-word` | 長さを変える形態素ラベル |
| `--stretch-ratio` / `--stretch-seconds` | 正の倍率／秒数。`--stretch-word` と組で指定 |
| `--steps` | Flow Matching の ODE ステップ数。正の整数、既定 32 |
| `--cfg-strength` | CFG 強度。0 以上、既定 1.5 |
| `--save-baseline` | 編集前の音声も指定パスに保存 |

CLI の録音経路では、`--annotation`・`--transcript`・`--transcript-text` のうち一つだけを指定します。書き起こしだけを渡す自動アラインメント経路は、現状の日本語 MFA の不整合を回避する方法ではありません。

編集指定がなければベースラインをそのまま保存します。保存する WAV は 48 kHz / PCM 16-bit で、振幅は −1～1 にクリップされます。

<a id="python"></a>
## Python からの日本語解析

以下は TTS 重みを使わない日本語フロントエンドの確認です。同期済みの環境で `uv run --extra cpu --extra japanese python` を起動し、実行できます。

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

`feature.pitch` は言語的な Low / High アクセント情報で、音声から測った F0 や編集用のピッチ bin とは別です。

参照文と対象文から、モデル入力用の音素列・言語特徴量を作る例：

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

これは入力の前処理であり、音声生成や音素時刻の推定は行いません。ネイティブ LFM トークンを併用する例やモデル構造は [JAPANESE.md](JAPANESE.md) を参照してください。

<a id="training"></a>
## 学習する場合

詳細は日本語の [TRAINING.md](TRAINING.md) にまとめています。学習済みモデル、コーパス、MFA と OpenJTalk の音素時刻対応付けは付属しません。

### 日本語 TTS の学習

学習は事前計算した SARA 特徴量を使います。まず学習・検証データの manifest と特徴量を監査します。この操作はモデル重みをダウンロードしません。

```sh
uv run --extra cpu --extra japanese ctrlspeech-train --train-manifest data/train.jsonl --validation-manifest data/validation.jsonl --output-dir checkpoints/ja --validate-only
```

実学習には、同じ SARA ペアのメタ情報と、特徴量作成に使ったエンコーダー重みが必要です。特徴量の形式、話者ごとのデータ分割、5 → 15 → 30 → 60 秒の段階学習、再開方法は [学習ガイド](TRAINING.md#実行と再開)と [`configs/japanese-training.yaml`](configs/japanese-training.yaml) を参照してください。

### SARA コーデックだけを学習する

`packages/svae` は CtrlSpeech に依存しない独立パッケージです。書き起こしなしの WAV から学習し、エンコーダー／デコーダーのペアを出力できます。member 単独の環境では日本語フロントエンド・デモ・ONNX の依存は不要です。

```sh
uv sync --frozen --package svae --extra cpu
uv run --package svae --extra cpu svae-train --device cpu --train-dir data/svae/train --validation-dir data/svae/validation --output-dir checkpoints/svae
uv run --package svae --extra cpu svae-export --checkpoint checkpoints/svae/best.ckpt --output-dir assets/svae
```

出力先は新しいディレクトリにしてください。構造、SSL 枝、GAN の目的関数、再開、オフライン export は [SARA を WAV から学習する](TRAINING.md#sara-を-wav-から学習する) を参照してください。TTS の操作へ戻るときは、ルートのバックエンドと `japanese` を指定して再同期します。

<a id="controls"></a>
## 制御の仕様と上限

| 項目 | 仕様 |
|---|---|
| 音声 | 48 kHz、制御抽出の hop は 480 サンプル |
| 制御の時間解像度 | 100 Hz、1 フレーム 0.01 秒 |
| Pitch | mel 間隔の 128 bin。bin 0 は無声。スライダーによるシフトは無声フレームを維持 |
| Loudness | A 特性の dB を 64 bin に量子化。約 1.05 bin / dB |
| Duration | 音素ごとのフレーム数を対数圧縮した連続 MLP 条件。旧 192 エントリーの埋め込み表の上限は不使用 |
| 日本語アクセント | Low / High、アクセント句境界、アクセント核、句のモーラ数を独立した言語条件として入力 |
| 長文の安全上限 | 目標 60 秒、制御タイムライン最大 6001 点、AR 最大 750 ステップ |
| SARA 潜在特徴量 | 標準は 50 Hz / 128 次元。AR 1 ステップで 4 個、つまり 0.08 秒分 |

AR の生成ステップ数は要求する長さから算出します。単語を伸ばしたときに固定のステップ数で打ち切らない設計です。60 秒は安全上限・開発目標であり、日本語の長文音質を実証した値ではありません。また、Web UI のアップロード上限はこれとは別に 10 秒です。

<a id="environment"></a>
## 環境変数

| 変数 | 用途 |
|---|---|
| `CTRLSPEECH_ASSETS` | ローカルアセットの親ディレクトリ。指定時はダウンロードを省略 |
| `CTRLSPEECH_HF_REPO` | 公開後の日本語チェックポイントの Hub リポジトリ ID |
| `CTRLSPEECH_MFA_CACHE` | MFA の作業ファイル先。既定は `~/.cache/ctrlspeech/mfa` |
| `CTRLSPEECH_MFA_DICT` | 旧英語アライナー用の辞書指定。`JapaneseMFAAligner` では不使用 |
| `CTRLSPEECH_DEMO_MODEL` | デモのモデル名。既定は `japanese-lfm2-350m` |
| `CTRLSPEECH_DEMO_DEVICE` | デモの実行先。既定は `auto` |
| `CTRLSPEECH_TEST_ISOLATE_ONNX` | テスト時に `1` を指定してネイティブ ONNX Runtime を分離 |
| `CTRLSPEECH_TEST_DEVICE` | バックエンド実行テストで必須にするデバイス |
| `SVAE_TEST_DEVICE` | 独立コーデックの学習テストで必須にするデバイス |

日本語 MFA ラッパーは MFA の日本語 tokenizer・辞書・G2P を使い、呼び出しごとに作業ファイルを分離します。Python API の `JapaneseMFAAligner(timeout_seconds=600)` でタイムアウトを設定できます。旧英語用の小辞書最適化は日本語経路には適用しません。

本番コードは音声セッション作成前に ONNX Runtime 公式 API でテレメトリーを無効化します。ただしネイティブライブラリの import 後の処理であり、ネットワークの遮断機能ではありません。機微な録音を扱う場合は、実行環境のネットワークポリシーも管理してください。

<a id="development"></a>
## 開発時の確認

### CPU のテスト

この同期操作ではモデル重みをダウンロードしません。

```sh
uv sync --frozen --extra cpu --extra japanese --group dev
```

PowerShell：

```powershell
$env:CTRLSPEECH_TEST_ISOLATE_ONNX = "1"
uv run --no-sync pytest -q
```

Bash：

```bash
CTRLSPEECH_TEST_ISOLATE_ONNX=1 uv run --no-sync pytest -q
```

このモードは小型の実 Torch モデル、OpenJTalk のテキスト解析、スタブ化した話者／音声連携を確認するものです。テスト内では Hub ダウンロードと Hub テレメトリーを無効化しています。**ONNX 推論、公開チェックポイント互換性、日本語音質、GPU 性能は検証しません。** CI もこの制限付きモードです。

### GPU 実行を必須にする確認

Intel XPU の PowerShell 例：

```powershell
$env:CTRLSPEECH_TEST_DEVICE = "xpu"
$env:CTRLSPEECH_TEST_ISOLATE_ONNX = "1"
uv run --extra xpu --extra japanese pytest -q tests/test_backend_execution.py
```

Bash の場合：

```bash
CTRLSPEECH_TEST_DEVICE=xpu CTRLSPEECH_TEST_ISOLATE_ONNX=1 uv run --extra xpu --extra japanese pytest -q tests/test_backend_execution.py
```

AMD なら両方の `xpu` を `rocm`、NVIDIA なら `cuda` に置き換えます。明示した GPU が使えなければ失敗し、FP32 の学習・保存／再開・キャッシュ付き生成・WAV 復元がその GPU で実行される必要があります。BF16 のケースだけは、ハードウェア非対応時にスキップされます。

独立コーデックは、同じルート側プロファイルで `SVAE_TEST_DEVICE=xpu` などを設定し、`tests/test_svae_training.py` を実行します。これらは小型モデル・ランダム重み・合成音声などによる実行確認であり、学習済み音声の品質や大型 SSL モデルの性能測定ではありません。CPU 上の模擬 GPU テストも、GPU 実機での実行証拠にはなりません。

<a id="troubleshooting"></a>
## よくある問題

| 症状 | 確認すること |
|---|---|
| `Required uv version ...` | `uv --version` を確認し、0.12.1 以上を使う |
| バックエンドの依存関係が競合する | extra は `cpu` / `cuda` / `rocm` / `xpu` の一つだけ。`--all-extras` を使わない |
| `No public CtrlSpeech-JA checkpoint is configured yet` | 日本語重みは未公開。完全なローカルアセットを用意し `CTRLSPEECH_ASSETS` を設定する |
| `CtrlSpeech assets are incomplete` | アセットの親ディレクトリ指定と、エラーに列挙された不足ファイルを確認する |
| GPU を指定すると失敗する | 選んだ extra、Torch ビルド、ドライバー、対応 GPU、デバイス番号を確認する。明示指定時は CPU に自動退避しない |
| OpenJTalk と音素が一致しない | 生の MFA 音素列や名前だけ置き換えた列は使わず、音声と照合した正規音素時刻を指定する |
| アップロード時に書き起こしの不一致が出る | 入力欄と 4 行ファイルの書き起こしを一致させる。空欄ならファイルから補完される |
| UI に例が表示されない | `language: ja` と必要ファイルがある例のみ表示。同梱の `Dreams` は対象外 |
| 描こうとするとグラフ全体が動く | 先に `Draw Pitch` / `Draw Loudness` を押す。描いたら `Finish` で確定する |
| 単語が見つからない | `--stretch-word` は形態素ラベルを指定。エラーの候補一覧や `Word` の表示を確認する |
| 長さ変更が累積しない | UI の仕様。一回の `Apply` は一単語の変更で、前回分を置き換える |
| `achieved duration unverified` | 出力は生成済みでも実測長は未確認。出力に対する新しい正規音素時刻で評価する |

<a id="layout"></a>
## ディレクトリ構成

```text
ctrlspeech/
  pipeline.py       生成・録音取り込み・再生成の Python API
  assets.py         重みと補助アセットの取得・検証
  retime.py         単語単位の伸縮と長さの上限
  cli.py            ctrlspeech コマンド
  frontend/         OpenJTalk による日本語・アクセント解析
  data/             日本語のモデル入力列とバッチ作成
  align/            MFA ラッパーと 4 行音素時刻ファイル
  features/         高さ・音量・話者埋め込みの抽出
  models/           DiTar・LFM2・LocDiT
  training/         特徴量キャッシュ・段階学習・学習 CLI
packages/svae/
  src/svae/         独立 SARA コーデック・学習・export・デバイス選択
configs/            日本語モデル・学習設定・音素語彙
demo/
  app.py            Panel の Web UI
  interactive_plot.py  Bokeh の制御曲線・タイムライン編集
  assets/           デモ用音声・書き起こし・examples.json
scripts/
  generate.py       チェックアウトからの CLI 起動用
tests/              テスト群
```

<a id="license"></a>
## ライセンスと利用上の注意

- このリポジトリのコードは **MIT ライセンス**です。[LICENSE](LICENSE) を参照してください。
- オリジナル版の公開重みは **CC BY-NC 4.0（非商用）** です。コードの MIT ライセンスと混同しないでください。
- LFM2.5 の事前学習重みには別のモデルライセンスがあります。日本語の派生チェックポイントの利用・配布では、基盤モデルなど各構成要素の条件も確認してください。
- 声質を複製できるため、利用権限のある録音だけを使い、合成音声であることを明示してください。本人の同意のないなりすましには使わないでください。

<a id="citation"></a>
## 引用

```bibtex
@inproceedings{zheng2026ctrlspeech,
  title     = {CtrlSpeech: Coarse-to-Fine Control for Expressive Speech Synthesis},
  author    = {Zheng, Zhisheng and Sun, Xiaohang and Liu, Zhu and Chen, Caren and Kumar, Rohith and Aggarwal, Manoj and Medioni, Gerard and Harwath, David},
  year      = {2026},
  booktitle = {{Interspeech 2026}},
}
```

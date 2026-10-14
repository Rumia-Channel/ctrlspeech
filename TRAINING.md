# CtrlSpeech-JA と独立 SVAE の学習

事前計算した日本語の特徴量から、音声用モジュールの学習と LFM2 本体の
微調整を実行できます。設定は [japanese-training.yaml](configs/japanese-training.yaml)、
実行入口は `uv run --extra xpu --extra japanese ctrlspeech-train` です。

この実装は学習・保存・再開の処理を提供します。日本語の学習済みチェックポイント、
学習用コーパス、MFA 音素から OpenJTalk 音素への対応付けは付属しません。
音声品質や以下の学習率・エポック数は、実データでの評価がまだ必要です。

## SVAE を WAV から学習する

SVAE は `packages/svae` の独立 Python パッケージです。CtrlSpeech を import せず、
`svae-train` / `svae-export` を実行できます。両パッケージは uv workspace の
同じ lock を使用し、バックエンドのバイナリ指定は SVAE 側を正本にしています。
CtrlSpeech の backend extras は対応する SVAE extra を選択します。

```bash
uv sync --frozen --package svae --extra xpu
uv run --package svae --extra xpu svae-train --device xpu --train-dir data/svae/train --validation-dir data/svae/validation --output-dir checkpoints/svae
```

AMD は `xpu` を `rocm`、NVIDIA は `cuda`、CPU は `cpu` に置き換えます。
CPU の FP32 は `--precision none`、Intel Arc は BF16 を基本にします。
この workspace member だけの sync では、CtrlSpeech の日本語・ONNX・デモ依存は不要です。
CtrlSpeech へ戻る際は root で選択した backend と `japanese` extra を sync してください。

入力はそれぞれのディレクトリ以下の WAV だけです。書き起こし・音素時刻は使いません。
ステレオはモノラル化し、16 kHz へ resample した定長区間を抽出します。
初期区間長は 3 秒です。短いファイルは除外一覧へ出し、利用可能な例がゼロなら停止します。
短い発話を使う場合は `--clip-seconds` を 0.025 秒単位で指定してください。
学習と検証で同じパス／同じ WAV 内容を共有することは拒否します。SVAE の検証には
話者 ID は不要です。学習開始後は元 WAV を変更しないでください。

```bash
uv run --package svae --extra cpu svae-train --train-dir data/svae/train --validation-dir data/svae/validation --output-dir checkpoints/svae --validate-only
```

初期レシピは DAC 系エンコーダ・BigVGAN デコーダ、16 kHz・40 Hz・64 次元です。
凍結した実 `microsoft/wavlm-large` を取得し、最終 hidden features との意味整合を学習します。
別の教師は `--teacher-model` で指定でき、`distill_hidden_dim` と教師の hidden size を
一致させる必要があります。不足した重みをランダム初期化して続行しません。
`--semantic-weight 0` は明示的に意味整合を無効にした vanilla acoustic VAE です。

生成器の目的は多窓長メル復元、KL、hinge GAN、判別器の特徴一致、意味特徴整合です。
判別器は multi-period / multi-band STFT の両方を実際に更新します。
初期重みは復元 15、KL 0.01、GAN 1、特徴一致 2、意味整合 1。
Adam の初期 LR は 1e-4、エポックごとの指数減衰は 0.9996、EMA は 0.999 です。
検証は EMA の codec を使用し、学習乱数を変更しません。
原論文の数値に基づく開始レシピであり、音質や収束を保証する設定ではありません。
[Semantic-VAE 原論文](https://arxiv.org/html/2509.22167)も参照してください。

`--config` は `codec` / `training` の二つの YAML section を受け取ります。
`codec` は DAC の構造と、必要なら完全な `decoder_dict` を指定します。
`training` は `svae.training.engine.TrainingConfig` の学習率、各損失重み、
FFT 窓、判別器、教師、精度などを指定します。例:

```yaml
training:
  epochs: 10
  batch_size: 4
  clip_seconds: 3.0
  precision: bf16
  teacher_model: microsoft/wavlm-large
  semantic_weight: 1.0
```

既存 codec からの追加学習は `--init-from assets/svae` で開始します。
そのアーキテクチャの変更は拒否します。元のモデルと教師のライセンスを確認してください。
保存は `latest.ckpt` / `best.ckpt`、再開点は完了済みエポックです。再開は設定・WAV 内容・
教師 identity・backend・デバイス番号の一致を要求し、GPU カーネルの完全な決定性は
保証しません。目標エポック数だけは延長できます。

```bash
uv run --package svae --extra xpu svae-train --device xpu --train-dir data/svae/train --validation-dir data/svae/validation --output-dir checkpoints/svae --resume checkpoints/svae/latest.ckpt --epochs 20
uv run --package svae --extra xpu svae-export --checkpoint checkpoints/svae/best.ckpt --output-dir assets/svae
```

各エポックでは `exports/epoch-XXXXXX` にペアを出力し、JSON ログにパスを出します。
`svae-export` も同じ形式を出力します。出力先は新しいディレクトリが必要で、
既存のペアを上書きしません。一つのディレクトリ rename で完成したペアを公開します。
内容は `metainfo.json`、`dac/weights.pth`、`dac/ema_state_dict.pth`、
EMA エンコーダ単体の `encoder.safetensors` です。

前処理では `svae.loading.load_encoder()` と `encode_audio()`、
復元では `load_decoder()` を使います。`build_encoder()` は構造を作るだけです。
この出力を使って TTS 特徴量を作り、下記の CtrlSpeech 学習へ同じエンコーダを渡します。
SVAE を変更したら、エンコーダとデコーダを対で扱い、TTS 特徴量も作り直してください。

### SVAE の修正と評価上の注意

- KL と事後サンプリングの演算を FP32 化。FP16 の `exp(12)` オーバーフローを回避。
  KL は `expm1` を使用し、標準正規分布でゼロになる定義に合わせます。
- 意味投影は教師側の長さへ整合し、パディングを除いた各例を resample。
  潜在フレーム数より後ろの有効な教師特徴を捨てず、検証でも同じ意味損失を計算します。
- 符号化は一度だけサンプリングし、推論時に不要な KL を計算しません。
  decoder-only load は不要なエンコーダ／事後推定層を GPU に保持しません。
- DAC の `decoder_rates`、または BigVGAN の `upsample_rates` の積が
  エンコーダの hop と一致することを構築時に検査します。学習時の切り詰めに隠れた
  デコーダ時間軸の不一致を拒否し、書き出し後に音声が伸びることを防ぎます。

小型の実 codec と SSL モデル構造のテスト重みで、CPU FP32/BF16 の学習、
判別器と意味投影の更新、検証の乱数隔離、再開、厳密な重み読み込みと WAV 復元を確認します。
`SVAE_TEST_DEVICE=xpu` または `rocm` を設定して `tests/test_svae_training.py` を実行すると、
対象 GPU を必須にできます。この環境で AMD/Arc の実機実行、既定の大型教師での学習、
実音声の音質改善、必要 VRAM はまだ確認していません。
日本語／多言語教師や中間層の選択、KL 重み、潜在次元の変更は比較実験候補であり、
改善済みの音質として扱わないでください。


## 学習の進め方

初期設定では、5 秒以内の例で新しい音声用モジュールだけを学習し、次の段階から
LFM2 本体を解凍します。上限を 15、30、60 秒へ増やし、後半では学習率を下げます。
本体の学習率は音声用モジュールの 1/10 に設定しています。SVAE エンコーダは
全段階で固定します。期間は開始用の設定であり、最適な期間を保証するものではありません。

長さの上限は**参照音声と生成対象音声を合わせた長さ**です。潜在特徴量は 40 Hz、
音響制御は 100 Hz の別の時間軸を使います。長い例を途中で切ると音素や停止ラベルが
壊れるため、各段階では上限に収まる例だけを選択します。最終上限より長い例も除外されます。
各段階には上限以下の学習例が少なくとも一つ必要です。
エポック数、バッチサイズ、勾配蓄積回数などは整数で指定し、`train_backbone` などの
フラグには文字列ではなく YAML の `true` / `false` を使ってください。

長さの近い例を同じバッチにまとめます。初期値はバッチサイズ 2、勾配蓄積 8 回です。
最後の蓄積区間が 8 回未満でも、その区間の実際の回数で損失を割って更新します。
AdamW、段階ごとのウォームアップと cosine decay、勾配クリッピング、勾配チェックポイントを
使用します。勾配をクリップする前に AMP のスケールを解除します。
この順序は [PyTorch の AMP の説明](https://docs.pytorch.org/docs/2.14/notes/amp_examples.html)
に沿っています。

`precision: bf16` が初期値です。CPU・CUDA・ROCm・XPU で同じ学習器を使います。
導入は [README の backend profiles](README.md#install) を参照し、`uv run` にも
選択した extra を付けてください。以下の学習・再開例は Intel Arc の XPU です。
AMD では `--extra rocm --device rocm`、NVIDIA では `--extra cuda --device cuda` に
置き換えます。`--device auto` は CUDA/ROCm → XPU → CPU の順で選び、
明示した GPU が利用できなければエラーになります。

BF16 非対応の CUDA/ROCm GPU では `--precision fp16` を選択できます。
XPU の FP16 学習は GradScaler が必要とする FP64 もハードウェアで対応している場合だけ
許可します。FP64 非対応の Intel Arc では **BF16 または `none`** を使ってください。
CPU の動作確認では `--precision none --device cpu`、または BF16 を指定できます。
AMP と FP16 scaler は選択した backend に合わせます。
非有限の損失や勾配が発生した場合は、蓄積した勾配を破棄し、その更新を行わずエラーで終了します。

## データと品質確認

参照と対象は同じ話者の別発話を組にし、話者埋め込みは参照音声だけから抽出します。
学習と検証は話者単位で分割します。同じ話者 ID や同じ特徴量ファイルが両方に
含まれる場合、CLI はエラーにします。ID は実際の話者を正しく表す必要があります。

前処理では次を確認してください。

1. 16 kHz の音声、書き起こし、OpenJTalk の読みが一致していること。
2. 学習済みの SVAE エンコーダで潜在特徴量を抽出していること。
   `svae.loading.build_encoder()` は構造だけを作るため、`metainfo.json` だけでは重みがランダムです。
3. 参照音声の終端が 4 潜在フレーム、つまり 0.1 秒のパッチ境界にあること。
   音声の区間を先に決め、その区間に対する書き起こし・時刻を確定してから特徴量を作ります。
   音素情報を残したまま潜在特徴量だけを切らないでください。
4. 音響制御を使う場合、OpenJTalk の各音素に一致する 100 Hz の開始・終了時刻があること。
   MFA の日本語音素をそのまま OpenJTalk の列に重ねないでください。
5. 無音、クリッピング、未知の読み、時刻のずれを監査し、異常な例を修正・除外すること。

検証は全段階で同じ例・同じ最終長さ上限・同じ乱数を使います。検証の実行は
学習用の PyTorch・Python 乱数状態を変更しません。検証でエラーが起きた場合も復元します。条件ドロップは検証時に無効です。
記録される指標は各バッチの損失を例数で重み付けした平均であり、CER や聴感品質ではありません。
検証バッチサイズを変えると集計とノイズの割り当ても変わるので、比較時は固定してください。

実音声の評価では、同じ未知話者・同じ文章・同じサンプリング設定を使い、CER、
話者類似度、アクセント、F0 制御、音素長の誤差、30〜60 秒での発話の抜けや反復を
確認します。停止精度だけでは、中間クラスを出し続けるモデルを見逃すため、
停止位置の誤差も確認してください。LocDiT のステップ削減は品質基準を決めてから行います。

## 特徴量ファイル

`.pt` ファイルはテンソルと基本型だけを含む辞書として保存します。
ロードは `weights_only=True` を使います。潜在特徴量と話者埋め込みは、バッチ作成時に
float32 に揃えます。半精度や倍精度で保存した場合も、計算時の精度は AMP の設定に従います。

| キー | 形状・型 | 内容 |
|---|---|---|
| `vae_features` | `[T, 64]` 浮動小数点 | 参照→対象の順の潜在特徴量、40 Hz |
| `prompt_frames` | 正の Python `int` | 参照潜在フレーム数、4 の倍数、`T` 未満 |
| `speaker_embs` | `[192]` 浮動小数点 | 参照音声の話者埋め込み |
| `text_inputs` | `[L]` int64 | 参照の音素→`\|`→対象の音素、パディングなし |
| `linguistic_features` | 各 `[L]` の辞書 | `accent_pitch`・`phrase_boundary` は int64、`accent_nucleus`・`phrase_mora_count` は非負値、`valid` は bool |
| `native_text_inputs` | `[N]` int64、省略可能 | 同じ参照・対象文章を LFM2 tokenizer で符号化した列 |
| `pitch` / `loudness` | 各 `[F]` int64、省略可能 | 参照→対象の 100 Hz 制御、値は 0〜127 / 0〜63 |
| `duration_segments` | `[L, 2]` int64、省略可能 | 制御時間軸の半開区間 `[start, end)`、区切りは `(0, 0)` |

`pitch`、`loudness`、`duration_segments` はセットで指定します。任意フィールドの
有無は同じバッチ内で揃える必要があります。通常はコーパス全体で揃えてください。
音響制御なしのデータでは、pitch/loudness/duration の制御用モジュールは学習されません。
pitch と loudness のフレーム数は一致させ、非ゼロ長の duration_segments は音素順に
重複しない区間を指定してください。ゼロ長区間はこの順序チェックから除外します。
型や形状が不正なキャッシュは明示的なエラーにし、データセットのエラーにはファイル名を添えます。

テキスト列は既存の前処理を使って作れます。次は抽出済みの実特徴量を保存する例です。
`prompt_latents`、`target_latents` は `[T, 64]`、`speaker_embedding` は `[192]` の
抽出済みテンソルを渡します。

```python
import torch
from transformers import AutoTokenizer
from ctrlspeech.data import encode_japanese_pair, collate_japanese_sequences
from ctrlspeech.training import validate_example

tokenizer = AutoTokenizer.from_pretrained("LiquidAI/LFM2.5-350M-Base")
sequence = encode_japanese_pair(prompt_text, target_text, native_tokenizer=tokenizer)
text = collate_japanese_sequences([sequence])
example = {
    "vae_features": torch.cat([prompt_latents, target_latents]).float().cpu(),
    "prompt_frames": len(prompt_latents),
    "speaker_embs": speaker_embedding.float().cpu(),
    "text_inputs": text["text_inputs"][0],
    "native_text_inputs": text["native_text_inputs"][0],
    "linguistic_features": {
        name: tensor[0] for name, tensor in text["linguistic_features"].items()
    },
}
# 制御も学習する場合は、ここで音素に対応する pitch/loudness/duration_segments を追加。
validate_example(example)
torch.save(example, "features/pair-0001.pt")
```

ゼロ長の区間は「条件なし」を表し、フレーム 0 の値を取り込みません。
参照潜在特徴量は AR と LocDiT の履歴として使いますが、損失は対象音声のみに適用します。
パディングも損失から除外します。

各 split の JSONL manifest は、1 行につき一つのファイルを指定します。
パスは manifest の場所を基準に解決します。`frames` は参照と対象を合わせた `T` です。

```json
{"path": "features/pair-0001.pt", "frames": 180, "speaker_id": "speaker-001"}
```

## 実行と再開

最初に特徴量と分割を監査します。この操作はモデルをダウンロードしません。

```bash
uv run --extra cpu --extra japanese ctrlspeech-train --train-manifest data/train.jsonl --validation-manifest data/validation.jsonl --output-dir checkpoints/ja --validate-only
```

学習にはエンコーダ構造を記述した `metainfo.json` と、特徴量作成に使用した学習済みの
エンコーダ重みが必要です。重みはエンコーダ単体の state dict、または
`generator.*` / `model.generator.*` を含むチェックポイントから厳密に読み込みます。
SVAE 全体の別形式を指定する場合は、対応するエンコーダ state dict を事前に抽出してください。

```bash
uv run --extra xpu --extra japanese ctrlspeech-train --device xpu --train-manifest data/train.jsonl --validation-manifest data/validation.jsonl --svae-dir assets/svae --encoder-weights assets/svae/encoder.safetensors --output-dir checkpoints/ja
```

各エポック終了時に `latest.ckpt` と、検証損失が改善した場合の `best.ckpt` を保存します。
一時ファイルへ保存してから置換するため、保存中に中断しても既存のチェックポイントを
直接上書きしません。ログはエポックごとの JSON で標準出力へ出ます。

```bash
uv run --extra xpu --extra japanese ctrlspeech-train --device xpu --train-manifest data/train.jsonl --validation-manifest data/validation.jsonl --svae-dir assets/svae --resume checkpoints/ja/latest.ckpt --output-dir checkpoints/ja
```

再開点は完了済みエポックの境界です。モデル、optimizer、AMP scaler、学習段階内の
ステップ、乱数状態を復元します。元のモデル設定・学習設定・manifest を要求します。
manifest の内容は SHA-256 で確認しますが、参照先の特徴量ファイルの内容まではハッシュ
していません。学習開始後は特徴量を変更しないでください。同じデバイス・ソフトウェア・
設定での再開を想定しており、GPU カーネルの完全な決定性を保証するものではありません。

新しい学習チェックポイントは **format version 2** です。CPU・Python に加え、
選択した CUDA/ROCm/XPU デバイスの乱数状態と backend・デバイス番号を保存します。
検証でもそのデバイスの乱数状態を隔離し、正常終了・エラーのどちらでも復元します。
学習再開には元と同じ backend とデバイス番号が必要です。
旧 version 1 の学習再開は対応しません。

チェックポイントの `state_dict` は backend に依存せず、推論側のローダーでも読めます。
推論用 asset tree には生成された `config.yaml` に加え、SVAE、vocoder、
話者埋め込みモデル、音素語彙など既存の補助ファイルを用意する必要があります。

## 実装上の修正

可変長の日本語・音素 prefix を詰めて左にパディングし、全サンプルで最後の有効な
テキスト状態から最初の音声パッチを予測します。音素ごとの pitch/loudness 集約は
累積和で一括計算します。条件ドロップはバッチ全体ではなく例ごとに行い、
LocDiT ではパッチごとに行います。

曲線状の flow 経路を選ぶ場合は、その経路を微分した正解速度を使います。
新規 DiT の AdaLN-Zero と出力射影のゼロ初期化を維持し、共有 projector と
パッチごとのランダム時刻にも対応します。全パディングの音声パッチでは、
attention に有効キーがない場合の NaN を防ぎ、出力をゼロにします。

小型の実 LFM2/DiT を使ったテストで、forward/backward、bf16 と勾配チェックポイント、
可変長入力、全 CFG モード、蓄積の端数、検証乱数の保存、再開後の更新一致を確認します。
段階の途中と、本体を解凍する段階への切り替え直前の両方から再開し、連続実行との
チェックポイント・指標・乱数状態の一致も確認します。
`tests/test_backend_execution.py` は小型の実 LFM2/DiT・SVAE エンコーダ・DAC/BigVGAN
デコーダで、CLI 学習、再開後の更新、推論ローダーでの学習重み読み込み、参照波形の符号化、
キャッシュ付き生成と WAV 保存を確認します。話者埋め込みはキャッシュを使用します。
`CTRLSPEECH_TEST_DEVICE=xpu` または `rocm` で実機を必須にできます。
対象 GPU を持たない環境での CPU／乱数 API の模擬確認は、GPU 動作保証ではありません。
実コーパスでの音声品質と各 GPU での速度・メモリ使用量は別途評価が必要です。

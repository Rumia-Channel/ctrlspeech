# CtrlSpeech-JA と独立 SARA コーデックの学習

事前計算した日本語の特徴量から、音声用モジュールの学習と LFM2 本体の
微調整を実行できます。設定は [japanese-training.yaml](configs/japanese-training.yaml)、
実行入口は `uv run --extra xpu --extra japanese ctrlspeech-train` です。

この実装は学習・保存・再開の処理を提供します。日本語の学習済みチェックポイント、
学習用コーパス、MFA 音素から OpenJTalk 音素への対応付けは付属しません。
音声品質や以下の学習率・エポック数は、実データでの評価がまだ必要です。

## SARA を WAV から学習する

SARA ベースのコーデックは `packages/svae` の独立 Python パッケージです。
CtrlSpeech を import せず、`svae-train` / `svae-export` を実行できます。
両パッケージは uv workspace の同じ lock を使い、backend のバイナリ指定は
`svae` 側を正本にしています。CtrlSpeech の extras は対応する member extra を選びます。

```bash
uv sync --frozen --package svae --extra xpu
uv run --package svae --extra xpu svae-train --device xpu --train-dir data/svae/train --validation-dir data/svae/validation --output-dir checkpoints/svae
```

AMD は `xpu` を `rocm`、NVIDIA は `cuda`、CPU は `cpu` に置き換えます。
CPU の FP32 は `--precision none`、Intel Arc は BF16 を基本にします。
member-only sync では CtrlSpeech の日本語・ONNX・デモ依存は不要です。
CtrlSpeech へ戻る際は root の backend と `japanese` extra を sync してください。

入力はディレクトリ以下の WAV だけです。書き起こし・音素時刻は使いません。
モノラル化し、48 kHz へ resample した定長区間を抽出します。初期区間長は 1 秒です。
短い WAV は除外一覧へ出し、利用可能な例がゼロなら停止します。
`--clip-seconds` は 0.02 秒単位です。学習・検証で同じパス／同じ WAV 内容の共有は拒否します。
コーデックの検証には話者 ID は不要です。学習開始後は元 WAV を変更しないでください。
高域まで学習するには、元音声にもその情報が必要です。16／24 kHz の WAV を
48 kHz に変換しても、失われた高域は復元できません。48 kHz はサンプリング周波数で、
理論上の扱える帯域は 24 kHz 未満です。音質そのものを保証する数値ではありません。

```bash
uv run --package svae --extra cpu svae-train --train-dir data/svae/train --validation-dir data/svae/validation --output-dir checkpoints/svae --validate-only
```

### モデルと学習仕様

[SARA 論文](https://arxiv.org/html/2606.11611v1)の二枝構造をベースに、
**ネイティブ 48 kHz・50 Hz・128 次元**を初期設定とします。
原論文は 24 kHz・64 次元です。このリポジトリは帯域・容量を拡張した派生実装で、
最後にアップサンプリングするだけの方式ではありません。

- 意味枝: 凍結した実 [`facebook/w2v-bert-2.0`](https://huggingface.co/facebook/w2v-bert-2.0)。
  内部で 16 kHz へ resample して log-mel 特徴を作ります。
  学習時だけの教師ではなく、特徴量抽出・参照音声の符号化にも必要です。
- 音響枝: 48 kHz 波形を直接受ける Snake 活性化の残差 CNN。
  stride は `[4,3,4,4,5]` で、原論文の最初の stride を 2→4 にして 50 Hz を維持します。
  2 層の一方向 LSTM を通し、意味枝とチャネル結合して Gaussian 潜在表現へ投影します。
- 復元: native Torch の HiFi-GAN。stride `[5,4,4,3,4]` で
  960 サンプル／潜在フレームを直接復元します。
  生成された潜在表現の復元だけなら意味枝は不要です。
- 損失: 多窓長メル復元、KL、hinge GAN、判別器の特徴一致。
  multi-period / multi-band・multi-scale STFT 判別器を実際に更新します。
  旧 Semantic-VAE の意味蒸留損失はありません。
- 初期重み: 復元 15、KL 0.01、GAN 1、特徴一致 1。AdamW の初期 LR は 1e-4。
  LR と KL は最初の 10,000 更新で線形 warmup、その後の LR は更新ごとに 0.9999996 倍。
  EMA は 0.999。凍結 SSL の重みは optimizer・EMA の複製対象から外します。
  復元・STFT 判別器の既定 FFT 窓は `[4096,2048,1024]` で、48 kHz の帯域を扱います。

CNN の幅、HiFi-GAN の完全な設定など、論文だけでは確定できない値はこの実装の選択です。
既定値は CNN 初期幅 64、音響特徴幅／LSTM 幅 1024、decoder 初期幅 1024。
初期の 128 次元は容量を増やす設計判断で、64 次元より高音質だと実証済みではありません。
比較する場合は `codec.vae_dim` を変えて別のペアを学習し、特徴量も再抽出してください。
TTS の `model.audio_channels` と `model.loc_decoder.model.out_dim`、
特徴量 metadata の `latent_dim` を同じ値にします。不一致は拒否し、切り詰め・ゼロ埋めはしません。
KL はチャネル和なので、次元数の比較では KL 重みも含めて評価してください。
公式 SARA チェックポイントとの互換性や、論文の音質・数値を再現済みとは扱いません。
モデル・データのライセンスも確認してください。SSL 重みの不足をランダム初期化で補いません。

`--config` は `codec` / `training` の YAML section を受け取ります。例:

```yaml
codec:
  semantic_model: facebook/w2v-bert-2.0
  vae_dim: 128
training:
  epochs: 10
  batch_size: 4
  clip_seconds: 1.0
  precision: bf16
  warmup_steps: 10000
```

SSL の取得元は `--semantic-model` でも指定できます。同じ 50 Hz の
Wav2Vec2BertModel / SeamlessM4TFeatureExtractor の組が必要です。
追加学習は `--init-from assets/svae` で開始し、元の構造を変更する override は拒否します。
再開点は完了済みエポックです。設定・WAV 内容・backend・デバイス番号を検査し、
モデル、凍結 SSL、optimizer、更新単位の scheduler、scaler、EMA、乱数を復元します。
目標エポック数だけは延長できます。GPU カーネルの完全な決定性は保証しません。

```bash
uv run --package svae --extra xpu svae-train --device xpu --train-dir data/svae/train --validation-dir data/svae/validation --output-dir checkpoints/svae --resume checkpoints/svae/latest.ckpt --epochs 20
uv run --package svae --extra xpu svae-export --checkpoint checkpoints/svae/best.ckpt --output-dir assets/svae
```

保存は `latest.ckpt` / `best.ckpt`、コーデック checkpoint は format 2 です。
各エポックは `exports/epoch-XXXXXX` にペアを出力します。出力先は新しいディレクトリが必要で、
既存のペアを上書きせず、一つの directory rename で公開します。
ペアは `metainfo.json`、`weights.safetensors`、`ema.safetensors`、
`encoder.safetensors` です。凍結 SSL 重みは最後のファイルへ一度だけ格納します。
raw / EMA のどちらでも encoder / decoder の組をロードできます。
export 後の符号化・復元は元の SSL 取得先や Hub へ接続せず実行できます。

前処理は `svae.loading.load_encoder()` と `encode_audio()`、復元は `load_decoder()`。
`build_encoder()` は構造だけを作り、完全な重みを読み込む前の符号化は拒否します。
decoder-only load は SSL モデルを構築・ダウンロードしません。
KL と事後サンプリングは FP32 演算です。復元倍率と 960-sample hop の不一致も拒否します。
旧 DAC／BigVGAN ペア、旧 40 Hz／24 kHz 特徴量、旧 TTS 重みは使えません。
新しい SARA ペアを学習・出力し、その仕様で TTS 特徴量を作り直して TTS も再学習してください。

小型の実 W2v-BERT／CNN／LSTM／HiFi-GAN と合成テスト音声で、
CPU FP32/BF16 の学習、凍結 SSL の不変性、融合層・判別器の更新、
検証の乱数隔離、再開、offline export の厳密なロードと WAV 復元を確認します。
`SVAE_TEST_DEVICE=xpu` または `rocm` で `tests/test_svae_training.py` の実機を必須にできます。
この環境では AMD/Arc の実機、大型 pretrained SSL を使った学習・音質・VRAM は未確認です。



## 学習の進め方

初期設定では、5 秒以内の例で新しい音声用モジュールだけを学習し、次の段階から
LFM2 本体を解凍します。上限を 15、30、60 秒へ増やし、後半では学習率を下げます。
本体の学習率は音声用モジュールの 1/10 に設定しています。SARA エンコーダは
全段階で固定します。期間は開始用の設定であり、最適な期間を保証するものではありません。

長さの上限は**参照音声と生成対象音声を合わせた長さ**です。潜在特徴量は 50 Hz、
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

1. 48 kHz の音声、書き起こし、OpenJTalk の読みが一致していること。
2. 学習済み SARA エンコーダで潜在特徴量を抽出していること。
   `build_encoder()` の構造だけでは符号化できません。`load_encoder()` を使ってください。
3. 参照音声の終端が 4 潜在フレーム、つまり 0.08 秒のパッチ境界にあること。
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
| `codec` | 辞書、必須 | `{"name":"sara","sample_rate":48000,"latent_fps":50,"latent_dim":128}`。次元はモデルと一致させる |
| `vae_features` | `[T, D]` 浮動小数点 | 参照→対象の順、50 Hz。初期 `D=128` |
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
`prompt_latents`、`target_latents` は `[T, D]`（初期 `D=128`）、`speaker_embedding` は `[192]` の
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
    "codec": {"name": "sara", "sample_rate": 48000, "latent_fps": 50, "latent_dim": 128},
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

学習には同じ SARA ペアの `metainfo.json` と、特徴量作成に使用した
`encoder.safetensors` が必要です。エンコーダ全体を厳密に読み込みます。
別の TTS checkpoint の `generator.*` / `model.generator.*` からの抽出も可能ですが、
その特徴量と同じエンコーダ重みであることを確認してください。

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

新しい TTS 学習 checkpoint は **format version 3** です。CPU・Python に加え、
選択した CUDA/ROCm/XPU デバイスの乱数状態と backend・デバイス番号を保存します。
検証でもそのデバイスの乱数状態を隔離し、正常終了・エラーのどちらでも復元します。
学習再開には元と同じ backend とデバイス番号が必要です。
旧 format 1 / 2、および旧 codec profile のキャッシュ・モデル設定は対応しません。

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
`tests/test_backend_execution.py` は小型の実 LFM2/DiT・SARA エンコーダ・HiFi-GAN
デコーダで、CLI 学習、再開後の更新、推論ローダーでの学習重み読み込み、参照波形の符号化、
キャッシュ付き生成と WAV 保存を確認します。話者埋め込みはキャッシュを使用します。
`CTRLSPEECH_TEST_DEVICE=xpu` または `rocm` で実機を必須にできます。
対象 GPU を持たない環境での CPU／乱数 API の模擬確認は、GPU 動作保証ではありません。
実コーパスでの音声品質と各 GPU での速度・メモリ使用量は別途評価が必要です。

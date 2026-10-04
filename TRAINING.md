# CtrlSpeech-JA の学習

事前計算した日本語の特徴量から、音声用モジュールの学習と LFM2 本体の
微調整を実行できます。設定は [japanese-training.yaml](configs/japanese-training.yaml)、
実行入口は `uv run ctrlspeech-train` です。

この実装は学習・保存・再開の処理を提供します。日本語の学習済みチェックポイント、
学習用コーパス、MFA 音素から OpenJTalk 音素への対応付けは付属しません。
音声品質や以下の学習率・エポック数は、実データでの評価がまだ必要です。

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

`precision: bf16` が初期値です。対応しない CUDA GPU では `--precision fp16`、
CPU の動作確認では `--precision none --device cpu` を指定できます。
非有限の損失や勾配が発生した場合は、蓄積した勾配を破棄し、その更新を行わずエラーで終了します。

## データと品質確認

参照と対象は同じ話者の別発話を組にし、話者埋め込みは参照音声だけから抽出します。
学習と検証は話者単位で分割します。同じ話者 ID や同じ特徴量ファイルが両方に
含まれる場合、CLI はエラーにします。ID は実際の話者を正しく表す必要があります。

前処理では次を確認してください。

1. 16 kHz の音声、書き起こし、OpenJTalk の読みが一致していること。
2. 学習済みの SVAE エンコーダで潜在特徴量を抽出していること。
   `load_state()` は構造を作るだけなので、`metainfo.json` だけでは重みがランダムです。
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
uv run ctrlspeech-train --train-manifest data/train.jsonl --validation-manifest data/validation.jsonl --output-dir checkpoints/ja --validate-only
```

学習にはエンコーダ構造を記述した `metainfo.json` と、特徴量作成に使用した学習済みの
エンコーダ重みが必要です。重みはエンコーダ単体の state dict、または
`generator.*` / `model.generator.*` を含むチェックポイントから厳密に読み込みます。
SVAE 全体の別形式を指定する場合は、対応するエンコーダ state dict を事前に抽出してください。

```bash
uv run ctrlspeech-train --train-manifest data/train.jsonl --validation-manifest data/validation.jsonl --svae-dir assets/svae --encoder-weights assets/encoder.safetensors --output-dir checkpoints/ja
```

各エポック終了時に `latest.ckpt` と、検証損失が改善した場合の `best.ckpt` を保存します。
一時ファイルへ保存してから置換するため、保存中に中断しても既存のチェックポイントを
直接上書きしません。ログはエポックごとの JSON で標準出力へ出ます。

```bash
uv run ctrlspeech-train --train-manifest data/train.jsonl --validation-manifest data/validation.jsonl --svae-dir assets/svae --resume checkpoints/ja/latest.ckpt --output-dir checkpoints/ja
```

再開点は完了済みエポックの境界です。モデル、optimizer、AMP scaler、学習段階内の
ステップ、乱数状態を復元します。元のモデル設定・学習設定・manifest を要求します。
manifest の内容は SHA-256 で確認しますが、参照先の特徴量ファイルの内容まではハッシュ
していません。学習開始後は特徴量を変更しないでください。同じデバイス・ソフトウェア・
設定での再開を想定しており、GPU カーネルの完全な決定性を保証するものではありません。

チェックポイントの `state_dict` は推論側のローダーでも読める形式です。
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
実コーパスでの音声品質と CUDA での速度・メモリ使用量は別途評価が必要です。

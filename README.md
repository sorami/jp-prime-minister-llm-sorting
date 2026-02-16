# LLMで歴代首相をソート

LLMを使って歴代内閣総理大臣64人を「主観的・意味的」な基準でランキングする実験。名前のみをLLMに与え、Chain of Thought（CoT）で根拠を示させた上で比較させる。

参考: [LLMでソート - ｼﾞｮｲｼﾞｮｲｼﾞｮｲ](https://joisino.hatenablog.com/entry/llmsort)

## プロジェクト構成

```
├── pyproject.toml              # プロジェクト設定・依存関係
├── .env                        # APIキー・モデル名（.gitignore済み）
├── notebooks/
│   ├── 01_pointwise.py         # ポイントワイズ法（絶対評価, 0〜100点）
│   ├── 02_listwise.py          # リストワイズ法（全要素一括評価）
│   ├── 03a_pairwise_comparison.py  # ペアワイズ法: 全ペア比較 / ポジションバイアス / 推移律違反
│   ├── 03b_pairwise_sorting.py    # ペアワイズ法: 勝利数ソート / KwikSort / 手法間精度比較
│   └── 04_other_criteria.py        # 他の軸でのKwikSort（参考ランキング）
├── src/
│   └── pm_sort/
│       ├── __init__.py         # パッケージ公開API
│       ├── core/               # 共通基盤
│       │   ├── api.py          # OpenAI API基盤（Usage, リトライ, コスト計算）
│       │   ├── cache.py        # 結果キャッシュ（JSON保存・読み込み）
│       │   ├── config.py       # 設定（モデル名取得、リトライ、並列数等）
│       │   ├── criteria.py     # 評価軸の定義（6軸）
│       │   └── data.py         # データ読み込み（CSV）
│       └── methods/            # LLM比較・ソート手法
│           ├── listwise.py     # リストワイズ評価（一括ランキング）
│           ├── pointwise.py    # ポイントワイズ評価（0〜100点）
│           └── pairwise/       # ペアワイズ法
│               ├── compare.py  # ペアワイズ比較（双方向対応）
│               ├── sort.py     # ソートアルゴリズム（KwikSort cached/live）
│               └── analyze.py  # 分析関数（勝利数集計、推移律違反検出）
└── data/
    ├── prime_ministers.csv      # 首相データ（no, name, tenure）
    └── results/                # API結果のキャッシュ（自動生成）
        └── <model>/            # モデル別（例: gpt-5-mini/）
```

## データ

初代・伊藤博文から岸田文雄まで、**64人**のユニークな内閣総理大臣。

- ファイル: `data/prime_ministers.csv`
- カラム: `no`（連番1〜64）, `name`（氏名）, `tenure`（在任期間、参考用）
- 複数回就任した首相（吉田茂、安倍晋三など）は1人として扱う
- デフォルト使用モデル（gpt-5-mini）の knowledge cutoff（2024年5月）以降に就任した石破茂・高市早苗の両氏は対象外とした

LLMへの入力には**氏名のみ**を使用する。Wikipedia記事等の外部テキストは与えない。LLMが学習済みの知識だけで判断する設計。

## 評価軸

6つの軸で首相をソートする（`src/pm_sort/core/criteria.py` で定義）:

| 軸                            | 左           | 右           |
| :---------------------------- | :----------- | :----------- |
| **左派 ↔ 右派**（デフォルト） | 左派         | 右派         |
| トップダウン ↔ ボトムアップ   | トップダウン | ボトムアップ |
| ロマンチスト ↔ リアリスト     | ロマンチスト | リアリスト   |
| 犬っぽい ↔ 猫っぽい           | 犬っぽい     | 猫っぽい     |
| 肉食系 ↔ 草食系               | 肉食系       | 草食系       |
| 陰キャ ↔ 陽キャ               | 陰キャ       | 陽キャ       |

---

## セットアップ

```bash
uv sync
```

`.env` ファイルに OpenAI API キーと使用モデルを設定する（`.env.example` を参照）:

```
OPENAI_API_KEY=sk-...   # https://platform.openai.com/api-keys
LLM_SORT_MODEL=gpt-5-mini  # https://developers.openai.com/api/docs/models
```

## 実験の実行

marimo ノートブックでインタラクティブに実験を実行する:

```bash
uv run marimo edit notebooks/01_pointwise.py
```

| ノートブック                 | 内容                                                       |
| :--------------------------- | :--------------------------------------------------------- |
| `01_pointwise.py`            | ポイントワイズ法（各首相に0〜100点の絶対評価）             |
| `02_listwise.py`             | リストワイズ法（64人を一括で並べ替え）                     |
| `03a_pairwise_comparison.py` | ペアワイズ法: 全ペア比較 / ポジションバイアス / 推移律違反 |
| `03b_pairwise_sorting.py`    | ペアワイズ法: 勝利数ソート / KwikSort / 手法間精度比較     |
| `04_other_criteria.py`       | 他の軸でのKwikSort（参考ランキング）                       |

キャッシュ機構により、中断しても途中から再開可能。結果は `data/results/` 以下にJSON形式で保存される。

## 実験設計

ノートブックごとに異なるアプローチで首相をランキングし、手法間の精度とコストを比較する。

### `01_pointwise.py` — ポイントワイズ法

各首相に0〜100点の絶対スコアをつけさせる（64回のAPI呼び出し）。ナイーブなベースライン。

### `02_listwise.py` — リストワイズ法

64人全員を1プロンプトに入れ、一括で並べ替えさせる（1回のAPI呼び出し）。ハルシネーションの有無を検証。

### `03a_pairwise_comparison.py` — ペアワイズ比較

全 C(64,2) = 2,016ペアを双方向で比較（計4,032回のAPI呼び出し）。ポジションバイアスの検証、推移律違反（三すくみ）の検出を行う。デフォルト軸「左派 ↔ 右派」のみで実施。

### `03b_pairwise_sorting.py` — ソートアルゴリズム比較

`03a` の全ペア比較データを使って複数のソート手法を比較する（API呼び出しなし）:

- 勝利数ソート（ベースライン）
- KwikSort（seed 0〜99 の100パターンで安定性検証）
- 手法間の精度比較（リストワイズ・KwikSort の Kendall τ をベースラインと比較）

### `04_other_criteria.py` — 他の軸でのランキング

デフォルト軸「左派 ↔ 右派」以外の5つの軸で KwikSort を各1回実行し、参考ランキングを生成する。API を呼びながらソートする `kwiksort_live` を使用。seed=0 固定の1回実行。

### Chain of Thought（CoT）

2つの仕組みを併用している:

1. **プロンプトによるCoT**: 比較プロンプトで「考察を述べた上で回答せよ」と指示。LLMの判断根拠が `output_text` に含まれる
2. **OpenAI Reasoning API**: `reasoning={"effort": "medium"}` パラメータでモデル内部の推論プロセスを有効化。推論サマリーテキストも抽出・保存する

```
以下の2人の内閣総理大臣を「左派 ↔ 右派」の軸で比較してください。
伝統・秩序・現状維持を重視する右派的な政治姿勢か、改革・変革を志向する左派的な政治姿勢か

【A】安倍晋三
【B】村山富市

それぞれの人物についてこの軸に関する考察を簡潔に述べた上で、
最後の行に「回答: A」または「回答: B」と、より「左派」寄りの人物を回答してください。
```

### 設定パラメータ

`src/pm_sort/core/config.py` で以下を管理:

| パラメータ                 | デフォルト | 説明                                           |
| :------------------------- | :--------- | :--------------------------------------------- |
| `MAX_RETRIES`              | 5          | APIエラー時の最大リトライ回数                  |
| `BASE_DELAY`               | 1.0        | リトライ時の基本待機時間（秒、指数バックオフ） |
| `DEFAULT_REASONING_EFFORT` | `"medium"` | reasoning モデルのデフォルト思考量             |
| `MAX_CONCURRENCY`          | 50         | API同時並列リクエスト数の上限                  |

モデル名は `.env` の `LLM_SORT_MODEL` で指定する。

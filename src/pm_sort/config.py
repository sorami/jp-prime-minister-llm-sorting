import os

# APIエラー時の最大リトライ回数
MAX_RETRIES = 5

# リトライ時の基本待機時間（秒）。指数バックオフの基底値として使用される。
BASE_DELAY = 1.0

# reasoning モデルのデフォルト思考量。呼び出し側で個別にオーバーライド可能。
DEFAULT_REASONING_EFFORT = "medium"

# 推論サマリーの出力モード。"auto" / "concise" / "detailed" から選択。
DEFAULT_REASONING_SUMMARY = "detailed"

# API同時並列リクエスト数の上限。
MAX_CONCURRENCY = 20


# モデルごとのトークン単価（USD / 1M tokens）。
MODEL_PRICING: dict[str, dict[str, float]] = {
    # https://developers.openai.com/api/docs/models/gpt-5-mini
    "gpt-5-mini": {
        "input": 0.25,
        "cached_input": 0.025,
        "output": 2.00,
    },
    # https://developers.openai.com/api/docs/models/gpt-5-nano
    "gpt-5-nano": {
        "input": 0.05,
        "cached_input": 0.005,
        "output": 0.40,
    },
}


def get_model() -> str:
    """環境変数 LLM_SORT_MODEL からモデル名を取得する。"""
    model = os.environ.get("LLM_SORT_MODEL")
    if not model:
        raise RuntimeError(
            "環境変数 LLM_SORT_MODEL が設定されていません。.env ファイルを確認してください。"
        )
    return model

import os

# APIエラー時の最大リトライ回数
MAX_RETRIES = 5

# リトライ時の基本待機時間（秒）。指数バックオフの基底値として使用される。
BASE_DELAY = 1.0


def get_model() -> str:
    """環境変数 LLM_SORT_MODEL からモデル名を取得する。"""
    model = os.environ.get("LLM_SORT_MODEL")
    if not model:
        raise RuntimeError(
            "環境変数 LLM_SORT_MODEL が設定されていません。.env ファイルを確認してください。"
        )
    return model

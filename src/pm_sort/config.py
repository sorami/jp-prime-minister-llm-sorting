import os


def get_model() -> str:
    """環境変数 LLM_SORT_MODEL からモデル名を取得する。"""
    model = os.environ.get("LLM_SORT_MODEL")
    if not model:
        raise RuntimeError(
            "環境変数 LLM_SORT_MODEL が設定されていません。.env ファイルを確認してください。"
        )
    return model

import json
import logging
from pathlib import Path
from typing import Any

from .config import get_model

logger = logging.getLogger(__name__)

RESULTS_DIR = Path(__file__).parent.parent.parent.parent / "data" / "results"


def _cache_path(experiment: str, criterion_name: str, suffix: str = "") -> Path:
    """キャッシュファイルのパスを生成する。ディレクトリが無ければ作成する。"""
    d = RESULTS_DIR / get_model() / experiment
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{criterion_name}{suffix}.json"


def has_cache(experiment: str, criterion_name: str, suffix: str = "") -> bool:
    """指定された実験・基準のキャッシュが存在するか確認する。"""
    return _cache_path(experiment, criterion_name, suffix).exists()


def save_results(
    experiment: str, criterion_name: str, data: Any, suffix: str = ""
) -> Path:
    """結果をJSONファイルとして保存し、パスを返す。"""
    path = _cache_path(experiment, criterion_name, suffix)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    return path


def load_results(experiment: str, criterion_name: str, suffix: str = "") -> Any | None:
    """キャッシュされた結果を読み込む。存在しないか破損していれば None を返す。"""
    path = _cache_path(experiment, criterion_name, suffix)
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError:
        logger.warning(
            "キャッシュファイル %s が破損しています。キャッシュミスとして扱います", path
        )
        return None


def nested_int_keys(d: dict) -> dict:
    """2階層ネスト辞書のJSON文字列キーをintに変換する。

    ペアワイズ比較結果 {no_a: {no_b: result, ...}, ...} の読み込みに使用。
    """
    return {int(k): {int(k2): v2 for k2, v2 in v.items()} for k, v in d.items()}

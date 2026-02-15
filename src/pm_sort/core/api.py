import asyncio
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass

from openai import APIError, AsyncOpenAI, RateLimitError

from .config import (
    BASE_DELAY,
    DEFAULT_REASONING_EFFORT,
    DEFAULT_REASONING_SUMMARY,
    MAX_RETRIES,
    MODEL_PRICING,
)

# ---------------------------------------------------------------------------
# Usage データクラス
# ---------------------------------------------------------------------------


@dataclass
class Usage:
    """1回のAPI呼び出しにおけるトークン使用量。

    cached_input_tokens は input_tokens の内数（キャッシュヒットした入力トークン）。
    reasoning_tokens は output_tokens の内数（モデルの内部推論に使われた出力トークン）。
    total_tokens = input_tokens + output_tokens。
    """

    input_tokens: int = 0
    cached_input_tokens: int = 0
    output_tokens: int = 0
    reasoning_tokens: int = 0
    total_tokens: int = 0

    def to_dict(self) -> dict:
        return {
            "input_tokens": self.input_tokens,
            "cached_input_tokens": self.cached_input_tokens,
            "output_tokens": self.output_tokens,
            "reasoning_tokens": self.reasoning_tokens,
            "total_tokens": self.total_tokens,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Usage":
        return cls(
            input_tokens=d.get("input_tokens", 0),
            cached_input_tokens=d.get("cached_input_tokens", 0),
            output_tokens=d.get("output_tokens", 0),
            reasoning_tokens=d.get("reasoning_tokens", 0),
            total_tokens=d.get("total_tokens", 0),
        )

    def __add__(self, other: "Usage") -> "Usage":
        return Usage(
            input_tokens=self.input_tokens + other.input_tokens,
            cached_input_tokens=self.cached_input_tokens + other.cached_input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            reasoning_tokens=self.reasoning_tokens + other.reasoning_tokens,
            total_tokens=self.total_tokens + other.total_tokens,
        )


def extract_usage(response) -> Usage:
    """APIレスポンスからトークン使用量を抽出する。"""
    u = getattr(response, "usage", None)
    if u is None:
        return Usage()
    reasoning = 0
    details = getattr(u, "output_tokens_details", None)
    if details:
        reasoning = getattr(details, "reasoning_tokens", 0) or 0
    cached = 0
    input_details = getattr(u, "input_tokens_details", None)
    if input_details:
        cached = getattr(input_details, "cached_tokens", 0) or 0
    return Usage(
        input_tokens=u.input_tokens,
        cached_input_tokens=cached,
        output_tokens=u.output_tokens,
        total_tokens=getattr(u, "total_tokens", u.input_tokens + u.output_tokens),
        reasoning_tokens=reasoning,
    )


# ---------------------------------------------------------------------------
# レスポンス抽出
# ---------------------------------------------------------------------------


def extract_reasoning_summary(response) -> str:
    """APIレスポンスから推論サマリーテキストを抽出する。"""
    for item in getattr(response, "output", []):
        if getattr(item, "type", None) == "reasoning":
            for part in getattr(item, "summary", []):
                if getattr(part, "type", None) == "summary_text":
                    return getattr(part, "text", "")
    return ""


# ---------------------------------------------------------------------------
# API呼び出し
# ---------------------------------------------------------------------------


@asynccontextmanager
async def maybe_acquire(semaphore: asyncio.Semaphore | None):
    """セマフォがあれば取得し、なければそのまま通過する。"""
    if semaphore:
        async with semaphore:
            yield
    else:
        yield


async def call_with_retry(client: AsyncOpenAI, **kwargs) -> tuple:
    """リトライ付きでAPIを呼び出す。(response, elapsed_seconds, reasoning_effort) を返す。"""
    kwargs.setdefault(
        "reasoning",
        {"effort": DEFAULT_REASONING_EFFORT, "summary": DEFAULT_REASONING_SUMMARY},
    )
    reasoning_effort = kwargs.get("reasoning", {}).get("effort", "")
    for attempt in range(MAX_RETRIES):
        try:
            t0 = time.monotonic()
            r = await client.responses.create(**kwargs)
            elapsed = time.monotonic() - t0
            return r, elapsed, reasoning_effort
        except RateLimitError as e:
            if "insufficient_quota" in str(e):
                raise RuntimeError(
                    "OpenAI APIのクォータ（残高）が不足しています。"
                    " https://platform.openai.com/settings/organization/billing を確認してください。"
                ) from e
            delay = BASE_DELAY * (2**attempt)
            await asyncio.sleep(delay)
        except APIError:
            if attempt == MAX_RETRIES - 1:
                raise
            await asyncio.sleep(BASE_DELAY)
    raise RuntimeError("Max retries exceeded")


# ---------------------------------------------------------------------------
# 使用量・コスト集計
# ---------------------------------------------------------------------------


def _find_pricing(model: str) -> dict[str, float] | None:
    """モデル名から料金表を探す。完全一致→プレフィックス一致の順で検索。"""
    if model in MODEL_PRICING:
        return MODEL_PRICING[model]
    for key, pricing in MODEL_PRICING.items():
        if model.startswith(key):
            return pricing
    return None


def calculate_cost(
    results: list[dict],
    *,
    usage_key: str = "usage",
    model_key: str = "model",
) -> float | None:
    """結果dictのリストからAPIコスト（USD）を算出する。

    モデルの料金表が MODEL_PRICING に存在しない場合は None を返す。
    cached_input_tokens は input_tokens の内数として扱い、
    キャッシュ分は cached_input 単価、残りは input 単価で計算する。
    """
    usages = [r for r in results if r.get(usage_key) and r.get(model_key)]
    if not usages:
        return None

    model = usages[0].get(model_key, "")
    pricing = _find_pricing(model)
    if pricing is None:
        return None

    total_input = sum(r[usage_key]["input_tokens"] for r in usages)
    total_cached = sum(r[usage_key].get("cached_input_tokens", 0) for r in usages)
    total_output = sum(r[usage_key]["output_tokens"] for r in usages)

    uncached_input = total_input - total_cached
    cost = (
        uncached_input * pricing["input"] / 1_000_000
        + total_cached * pricing["cached_input"] / 1_000_000
        + total_output * pricing["output"] / 1_000_000
    )
    return cost


def format_usage_summary(
    results: list[dict],
    *,
    usage_key: str = "usage",
    elapsed_key: str = "elapsed_seconds",
    calls_label: str | None = None,
) -> str:
    """結果dictのリストからAPI使用量・コスト・処理時間のMarkdownサマリーを生成する。"""
    parts = []
    usages = [r[usage_key] for r in results if r.get(usage_key)]
    if usages:
        total_input = sum(u["input_tokens"] for u in usages)
        total_cached = sum(u.get("cached_input_tokens", 0) for u in usages)
        total_output = sum(u["output_tokens"] for u in usages)
        total_reasoning = sum(u.get("reasoning_tokens", 0) for u in usages)
        label = calls_label or f"{len(usages)}回呼び出し"
        cached_part = f", cached: {total_cached:,}" if total_cached else ""
        parts.append(
            f"**API使用量** ({label}): "
            f"input: {total_input:,}{cached_part}, output: {total_output:,} "
            f"(reasoning: {total_reasoning:,}), "
            f"合計: {total_input + total_output:,} tokens"
        )
    cost = calculate_cost(results, usage_key=usage_key)
    if cost is not None:
        parts.append(f"**APIコスト**: ${cost:.4f}")
    elapsed_items = [r.get(elapsed_key, 0) for r in results]
    total_elapsed = sum(elapsed_items)
    if total_elapsed > 0:
        n = len(elapsed_items)
        if total_elapsed >= 60:
            parts.append(
                f"**処理時間**: {total_elapsed:.1f}秒（{total_elapsed / 60:.1f}分）"
                f"、平均 {total_elapsed / n:.2f}秒/件"
            )
        else:
            parts.append(
                f"**処理時間**: {total_elapsed:.1f}秒"
                f"（平均 {total_elapsed / n:.2f}秒/件）"
            )
    return "\n\n".join(parts)

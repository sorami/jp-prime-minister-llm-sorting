from __future__ import annotations

import asyncio
import logging
import random
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from openai import AsyncOpenAI

from ...core.criteria import Criterion
from .compare import compare_pair

logger = logging.getLogger(__name__)


async def kwiksort(
    items: list[dict],
    client: AsyncOpenAI,
    criterion: Criterion,
    *,
    semaphore: asyncio.Semaphore | None = None,
    comparison_log: list | None = None,
    rng: random.Random | None = None,
) -> list[dict]:
    """KwikSort: QuickSort-based approximate sorting using LLM comparisons."""
    if len(items) <= 1:
        return items

    if rng is None:
        rng = random.Random()

    pivot = rng.choice(items)
    left, right, equal = [], [], [pivot]

    other_items = [x for x in items if x["no"] != pivot["no"]]
    tasks = [
        compare_pair(client, item, pivot, criterion, semaphore=semaphore)
        for item in other_items
    ]
    results = await asyncio.gather(*tasks)

    for item, result in zip(other_items, results):
        if comparison_log is not None:
            comparison_log.append(
                {
                    "no_a": result.no_a,
                    "no_b": result.no_b,
                    "winner": result.winner,
                    "usage": result.usage.to_dict(),
                    "elapsed_seconds": round(result.elapsed_seconds, 3),
                    "response_id": result.response_id,
                    "model": result.model,
                    "created_at": result.created_at,
                    "reasoning_effort": result.reasoning_effort,
                    "reasoning_summary": result.reasoning_summary,
                }
            )
        if result.winner == "A":
            right.append(item)
        elif result.winner == "B":
            left.append(item)
        else:
            if result.winner == "INVALID":
                logger.warning(
                    "INVALID response for pair (%s, %s), treating as equal to pivot",
                    item["name"],
                    pivot["name"],
                )
            equal.append(item)

    sorted_left = await kwiksort(
        left,
        client,
        criterion,
        semaphore=semaphore,
        comparison_log=comparison_log,
        rng=rng,
    )
    sorted_right = await kwiksort(
        right,
        client,
        criterion,
        semaphore=semaphore,
        comparison_log=comparison_log,
        rng=rng,
    )

    return sorted_left + equal + sorted_right


def _build_winner_lookup(pair_results: list[dict]) -> dict[tuple[int, int], str]:
    """全ペア比較結果から単方向ルックアップ辞書を構築する。

    (no_a, no_b) -> winner_ab（aを先出しした場合の勝者）
    """
    lookup = {}
    for r in pair_results:
        lookup[(r["no_a"], r["no_b"])] = r["winner_ab"]
    return lookup


def kwiksort_cached(
    items: list[dict],
    pair_results: list[dict],
    *,
    comparison_log: list | None = None,
    rng: random.Random | None = None,
) -> list[dict]:
    """事前計算済みの全ペア比較結果を使ったKwikSort（API呼び出しなし）。"""
    if len(items) <= 1:
        return items

    if rng is None:
        rng = random.Random()

    lookup = _build_winner_lookup(pair_results)
    return _kwiksort_cached_inner(items, lookup, comparison_log=comparison_log, rng=rng)


def _kwiksort_cached_inner(
    items: list[dict],
    lookup: dict[tuple[int, int], str],
    *,
    comparison_log: list | None = None,
    rng: random.Random,
) -> list[dict]:
    if len(items) <= 1:
        return items

    pivot = rng.choice(items)
    left, right, equal = [], [], [pivot]

    pivot_no = pivot["no"]
    for item in items:
        if item["no"] == pivot_no:
            continue

        item_no = item["no"]
        # pivot を A（左側）として参照する方向を優先
        if (pivot_no, item_no) in lookup:
            winner = lookup[(pivot_no, item_no)]
            if winner == "A":
                left.append(item)
            elif winner == "B":
                right.append(item)
            else:
                equal.append(item)
        elif (item_no, pivot_no) in lookup:
            winner = lookup[(item_no, pivot_no)]
            # item が A 側なので勝敗を反転
            if winner == "A":
                right.append(item)
            elif winner == "B":
                left.append(item)
            else:
                equal.append(item)
        else:
            logger.warning(
                "No cached result for pair (%s, %s), treating as equal",
                item["name"],
                pivot["name"],
            )
            equal.append(item)

        if comparison_log is not None:
            comparison_log.append({"no_a": pivot_no, "no_b": item_no})

    sorted_left = _kwiksort_cached_inner(
        left, lookup, comparison_log=comparison_log, rng=rng
    )
    sorted_right = _kwiksort_cached_inner(
        right, lookup, comparison_log=comparison_log, rng=rng
    )
    return sorted_left + equal + sorted_right

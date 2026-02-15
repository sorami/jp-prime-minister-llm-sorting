from __future__ import annotations

import asyncio
import logging
import random
from collections.abc import Callable

from openai import AsyncOpenAI

from ...core.criteria import Criterion
from .compare import PairwiseResult, compare_pair

logger = logging.getLogger(__name__)


def kwiksort_cached(
    items: list[dict],
    pair_results: dict,
    *,
    comparison_log: list | None = None,
    rng: random.Random | None = None,
) -> list[dict]:
    """事前計算済みの全ペア比較結果を使ったKwikSort（API呼び出しなし）。

    pair_results はネスト辞書 {no_a: {no_b: {"winner": ..., ...}, ...}, ...}。
    """
    if len(items) <= 1:
        return items

    if rng is None:
        rng = random.Random()

    return _kwiksort_cached_inner(
        items, pair_results, comparison_log=comparison_log, rng=rng
    )


def _kwiksort_cached_inner(
    items: list[dict],
    pair_results: dict,
    *,
    comparison_log: list | None = None,
    rng: random.Random,
) -> list[dict]:
    if len(items) <= 1:
        return items

    pivot = rng.choice(items)
    left, equal, right = [], [pivot], []

    pivot_no = pivot["no"]
    for item in items:
        if item["no"] == pivot_no:
            continue

        item_no = item["no"]
        # pivot を A（先出し）とした比較結果を参照
        entry = pair_results.get(pivot_no, {}).get(item_no)
        if entry is None:
            logger.warning(
                "ペア (%s, %s) のキャッシュ結果なし、同等として扱います",
                item["name"],
                pivot["name"],
            )
            equal.append(item)
        else:
            winner = entry["winner"]
            if winner == "A":
                left.append(item)
            elif winner == "B":
                right.append(item)
            else:
                # INVALID など — 同等として扱う
                logger.warning(
                    "不正な比較結果を同等として扱います: pivot=%s, item=%s, winner=%r",
                    pivot["name"],
                    item["name"],
                    winner,
                )
                equal.append(item)

        if comparison_log is not None:
            comparison_log.append({"no_a": pivot_no, "no_b": item_no})

    sorted_left = _kwiksort_cached_inner(
        left, pair_results, comparison_log=comparison_log, rng=rng
    )
    sorted_right = _kwiksort_cached_inner(
        right, pair_results, comparison_log=comparison_log, rng=rng
    )
    return sorted_left + equal + sorted_right


# ---------------------------------------------------------------------------
# Live KwikSort — API を呼びながらソート
# ---------------------------------------------------------------------------


async def kwiksort_live(
    items: list[dict],
    criterion: Criterion,
    client: AsyncOpenAI,
    *,
    semaphore: asyncio.Semaphore | None = None,
    rng: random.Random | None = None,
    on_compare: Callable[[PairwiseResult], None] | None = None,
) -> tuple[list[dict], list[PairwiseResult]]:
    """API を呼びながら KwikSort を実行する。

    Returns:
        (sorted_items, all_comparison_results)
    """
    if len(items) <= 1:
        return items, []

    if rng is None:
        rng = random.Random()

    results: list[PairwiseResult] = []
    sorted_items = await _kwiksort_live_inner(
        items,
        criterion=criterion,
        client=client,
        semaphore=semaphore,
        rng=rng,
        results=results,
        on_compare=on_compare,
    )
    return sorted_items, results


async def _kwiksort_live_inner(
    items: list[dict],
    *,
    criterion: Criterion,
    client: AsyncOpenAI,
    semaphore: asyncio.Semaphore | None,
    rng: random.Random,
    results: list[PairwiseResult],
    on_compare: Callable[[PairwiseResult], None] | None,
) -> list[dict]:
    if len(items) <= 1:
        return items

    pivot = rng.choice(items)
    left, equal, right = [], [pivot], []

    pivot_no = pivot["no"]
    others = [item for item in items if item["no"] != pivot_no]

    # ピボットと各要素の比較を並列実行
    coros = [
        compare_pair(client, pivot, item, criterion, semaphore=semaphore)
        for item in others
    ]
    pair_results = await asyncio.gather(*coros)

    for item, result in zip(others, pair_results):
        results.append(result)
        if on_compare is not None:
            on_compare(result)

        if result.winner == "A":
            left.append(item)
        elif result.winner == "B":
            right.append(item)
        else:
            equal.append(item)

    kwargs = dict(
        criterion=criterion,
        client=client,
        semaphore=semaphore,
        rng=rng,
        results=results,
        on_compare=on_compare,
    )
    sorted_left = await _kwiksort_live_inner(left, **kwargs)
    sorted_right = await _kwiksort_live_inner(right, **kwargs)
    return sorted_left + equal + sorted_right

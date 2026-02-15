from __future__ import annotations

import logging
import random

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

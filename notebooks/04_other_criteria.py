import marimo

__generated_with = "0.19.11"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _():
    import asyncio
    import random

    import marimo as mo
    import polars as pl
    from openai import AsyncOpenAI

    from pm_sort.core.api import format_usage_summary
    from pm_sort.core.cache import has_cache, load_results, save_results
    from pm_sort.core.config import MAX_CONCURRENCY
    from pm_sort.core.data import load_prime_ministers
    from pm_sort.methods.pairwise import kwiksort_live

    return (
        AsyncOpenAI,
        MAX_CONCURRENCY,
        asyncio,
        format_usage_summary,
        has_cache,
        kwiksort_live,
        load_prime_ministers,
        load_results,
        mo,
        random,
        save_results,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    # 他の軸でのランキング

    デフォルト軸「左派 ↔ 右派」以外の軸で KwikSort を1回実行し、参考ランキングを生成する。
    seed=0 固定の1回実行のため、ピボット選択によって結果は変わりうる点に注意。
    """)
    return


@app.cell(hide_code=True)
def _(load_prime_ministers):
    pms = load_prime_ministers()
    pms_by_no = {p["no"]: p for p in pms}
    return pms, pms_by_no


@app.cell(hide_code=True)
def _(mo):
    from pm_sort.core.criteria import CRITERIA, DEFAULT_CRITERION

    _non_default = {
        name: c for name, c in CRITERIA.items() if name != DEFAULT_CRITERION
    }
    criterion_selector = mo.ui.dropdown(
        options={c.label_ja: name for name, c in _non_default.items()},
        value=list(_non_default.values())[0].label_ja,
        label="評価軸",
    )
    criterion_selector
    return CRITERIA, criterion_selector


@app.cell
def _(CRITERIA, criterion_selector):
    criterion = CRITERIA[criterion_selector.value]
    return (criterion,)


@app.cell(hide_code=True)
def _(criterion, mo):
    mo.md(f"""
    ---

    ## {criterion.label_ja}

    > {criterion.description}
    """)
    return


@app.cell(hide_code=True)
def _(criterion, has_cache, mo):
    _cache_key = f"pairwise/kwiksort/{criterion.name}"
    _cached = has_cache(_cache_key, "seed_0")
    _label = (
        f"KwikSort 実行（{criterion.label_ja}）"
        if not _cached
        else f"KwikSort 実行（{criterion.label_ja}, キャッシュあり）"
    )
    run_btn = mo.ui.run_button(label=_label)
    run_btn
    return (run_btn,)


@app.cell(hide_code=True)
async def _(
    AsyncOpenAI,
    MAX_CONCURRENCY,
    asyncio,
    criterion,
    format_usage_summary,
    has_cache,
    kwiksort_live,
    load_results,
    mo,
    pms,
    pms_by_no,
    random,
    run_btn,
    save_results,
):
    mo.stop(not run_btn.value)

    _cache_key = f"pairwise/kwiksort/{criterion.name}"
    _seed = 0

    # キャッシュがあればそこから読み込む
    if has_cache(_cache_key, f"seed_{_seed}"):
        _cached = load_results(_cache_key, f"seed_{_seed}")
        ranking_nos = _cached["ranking"]
        _num_comparisons = _cached["num_comparisons"]
        comparison_dicts = _cached.get("comparisons", [])

        _parts = [
            mo.md(
                f"**キャッシュから読み込み** (seed={_seed}, {_num_comparisons}回比較)"
            ),
        ]
        if comparison_dicts:
            _parts.append(mo.md(format_usage_summary(comparison_dicts)))
        _parts.append(
            mo.ui.table(
                [
                    {"順位": _i + 1, "氏名": pms_by_no[_no]["name"]}
                    for _i, _no in enumerate(ranking_nos)
                ],
            ),
        )
        _output = mo.vstack(_parts)
    else:
        # API を呼びながら KwikSort 実行
        _client = AsyncOpenAI()
        _sem = asyncio.Semaphore(MAX_CONCURRENCY)
        _rng = random.Random(_seed)

        _compare_count = [0]

        def _on_compare(_r):
            _compare_count[0] += 1
            mo.output.replace(
                mo.md(
                    f"**KwikSort: {criterion.label_ja}** — 比較中... {_compare_count[0]}回完了"
                )
            )

        _sorted_pms, _results = await kwiksort_live(
            list(pms),
            criterion,
            _client,
            semaphore=_sem,
            rng=_rng,
            on_compare=_on_compare,
        )

        ranking_nos = [p["no"] for p in _sorted_pms]
        comparison_dicts = [r.to_dict() for r in _results]
        _num_comparisons = len(_results)

        # キャッシュに保存（03b と同じ形式）
        save_results(
            _cache_key,
            f"seed_{_seed}",
            {
                "ranking": ranking_nos,
                "comparisons": comparison_dicts,
                "seed": _seed,
                "num_comparisons": _num_comparisons,
            },
        )

        _output = mo.vstack(
            [
                mo.md(f"**完了** (seed={_seed}, {_num_comparisons}回比較)"),
                mo.md(format_usage_summary(comparison_dicts)),
                mo.ui.table(
                    [
                        {"順位": _i + 1, "氏名": pms_by_no[_no]["name"]}
                        for _i, _no in enumerate(ranking_nos)
                    ],
                ),
            ]
        )

    _output
    return comparison_dicts, ranking_nos


@app.cell(hide_code=True)
def _(comparison_dicts, criterion, mo, pms_by_no, ranking_nos):
    _top1_no = ranking_nos[0]
    _bottom1_no = ranking_nos[-1]
    _top1_name = pms_by_no[_top1_no]["name"]
    _bottom1_name = pms_by_no[_bottom1_no]["name"]

    def _find_response(target_no):
        for c in comparison_dicts:
            if c["no_a"] == target_no or c["no_b"] == target_no:
                _a = pms_by_no[c["no_a"]]["name"]
                _b = pms_by_no[c["no_b"]]["name"]
                return f"**{_a}** vs **{_b}**\n\n{c['raw_response']}"
        return "（該当する比較が見つかりません）"

    mo.md(
        f"### レスポンス例\n\n"
        f"#### {criterion.left}寄り 1位: {_top1_name}\n\n"
        f"{_find_response(_top1_no)}\n\n---\n\n"
        f"#### {criterion.right}寄り 1位: {_bottom1_name}\n\n"
        f"{_find_response(_bottom1_no)}"
    )
    return


if __name__ == "__main__":
    app.run()

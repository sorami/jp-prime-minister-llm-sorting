import marimo

__generated_with = "0.19.11"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _():
    import random

    import altair as alt
    import marimo as mo
    import polars as pl

    from pm_sort.core.api import calculate_cost
    from pm_sort.core.cache import (
        has_cache,
        load_results,
        nested_int_keys,
        save_results,
    )
    from pm_sort.core.data import load_prime_ministers
    from pm_sort.methods.pairwise import kwiksort_cached, win_count_sort

    return (
        alt,
        calculate_cost,
        has_cache,
        kwiksort_cached,
        load_prime_ministers,
        load_results,
        mo,
        nested_int_keys,
        pl,
        random,
        save_results,
        win_count_sort,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    # ソートアルゴリズムの比較

    矛盾の可能性があるLLM判定から、いかにして「妥当なランキング」を錬成するか。

    **前提**: `03a_pairwise_comparison.py` で取得した全ペア比較データ（2,016ペア）を使用。
    """)
    return


@app.cell(hide_code=True)
def _(load_prime_ministers):
    pms = load_prime_ministers()
    pms_by_no = {p["no"]: p for p in pms}
    return pms, pms_by_no


@app.cell(hide_code=True)
def _(mo):
    from pm_sort.core.criteria import CRITERIA
    from pm_sort.core.criteria import DEFAULT_CRITERION as _DEFAULT_CRITERION

    criterion_selector = mo.ui.dropdown(
        options={c.label_ja: name for name, c in CRITERIA.items()},
        value=CRITERIA[_DEFAULT_CRITERION].label_ja,
        label="評価基準",
    )
    criterion_selector
    return CRITERIA, criterion_selector


@app.cell
def _(CRITERIA, criterion_selector):
    criterion = CRITERIA[criterion_selector.value]
    return (criterion,)


@app.cell(hide_code=True)
def _(criterion, has_cache, load_results, mo, nested_int_keys):
    if not has_cache("pairwise", criterion.name):
        mo.stop(
            True,
            mo.md(
                "**ペアワイズ比較データが見つかりません。** 先に `03a_pairwise_comparison.py` を実行してください。"
            ),
        )

    pair_results = nested_int_keys(load_results("pairwise", criterion.name))
    _n_comparisons = sum(len(v) for v in pair_results.values())
    mo.md(f"ペアワイズデータ読み込み完了: **{_n_comparisons}件**")
    return (pair_results,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ---
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 勝利数ソート（ベースライン）

    総当たり2,016回の結果から、各首相の勝利数で順位を決定する。
    全ペア情報を使った「最も情報量の多いランキング」としてベースラインに設定。
    """)
    return


@app.cell
def _(pair_results, win_count_sort):
    wc_ranking = win_count_sort(pair_results)
    wc_rank_map = {no: rank for no, rank, _ in wc_ranking}
    return wc_rank_map, wc_ranking


@app.cell(hide_code=True)
def _(alt, mo, pl, pms_by_no, wc_ranking):
    _data = [
        {
            "pos": _i + 1,
            "rank": _rank,
            "no": _no,
            "wins": _wins,
            "name": pms_by_no[_no]["name"],
        }
        for _i, (_no, _rank, _wins) in enumerate(wc_ranking)
    ]
    _df = pl.DataFrame(_data)

    _chart = (
        alt.Chart(_df)
        .mark_bar()
        .encode(
            x=alt.X(
                "pos:O",
                title="順位",
                axis=alt.Axis(values=list(range(1, len(_data) + 1, 5))),
            ),
            y=alt.Y("wins:Q", title="勝利数"),
            tooltip=["rank", "no", "name", "wins"],
        )
        .properties(title="勝利数ソート: 勝利数分布", width=600, height=300)
    )

    # 同率順位の集計
    _n_unique_ranks = _df["rank"].n_unique()
    _n_total = len(_df)
    _tied_groups = (
        _df.group_by("rank")
        .agg(pl.col("name").count().alias("count"))
        .filter(pl.col("count") > 1)
        .sort("rank")
    )
    _n_tied_groups = len(_tied_groups)
    _n_tied_people = _tied_groups["count"].sum() if _n_tied_groups > 0 else 0

    _tied_details = ""
    if _n_tied_groups > 0:
        _tied_names = (
            _df.filter(pl.col("rank").is_in(_tied_groups["rank"]))
            .sort("rank", "no")
            .group_by("rank", maintain_order=True)
            .agg(
                pl.col("name"),
                pl.col("wins").first(),
            )
        )
        _tied_lines = []
        for _row in _tied_names.iter_rows(named=True):
            _names = "、".join(_row["name"])
            _tied_lines.append(f"  - {_row['rank']}位（{_row['wins']}勝）: {_names}")
        _tied_details = "\n".join(_tied_lines)

    mo.vstack(
        [
            mo.ui.altair_chart(_chart),
            mo.md(
                f"- 全{_n_total}人中、**{_n_unique_ranks}種類**の順位（{_n_tied_groups}組{_n_tied_people}人が同率）\n\n"
                + (f"**同率の組:**\n{_tied_details}" if _tied_details else "")
            ),
            mo.md("### 上位5人"),
            mo.ui.table(
                _df.head(5).select("rank", "no", "name", "wins"),
            ),
            mo.md("### 下位5人"),
            mo.ui.table(
                _df.tail(5).select("rank", "no", "name", "wins"),
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ---
    """)
    return


@app.cell(hide_code=True)
def _(criterion, mo):
    mo.md(f"""
    ## KwikSort（近似ソート）

    QuickSortベースの近似ソートアルゴリズム。
    全ペアの比較結果のうち O(N log N) 回分の参照だけでランキングを生成する。
    ピボット選択のランダム性により、実行のたびに結果が変わる。

    seed を変えて100回実行し、順位の安定性を検証する。

    評価基準: **{criterion.label_ja}**
    """)
    return


@app.cell(hide_code=True)
def _(criterion, mo):
    stability_run_btn = mo.ui.run_button(
        label=f"KwikSort 100回実行（{criterion.label_ja}, seed 0-99）"
    )
    stability_run_btn
    return (stability_run_btn,)


@app.cell(hide_code=True)
def _(
    criterion,
    has_cache,
    kwiksort_cached,
    load_results,
    mo,
    pair_results,
    pms,
    random,
    save_results,
    stability_run_btn,
):
    mo.stop(not stability_run_btn.value)

    _n_runs = 100
    _ks_experiment = f"pairwise/kwiksort/{criterion.name}"
    ks_multi_rankings = {}
    ks_multi_logs = {}

    with mo.status.progress_bar(total=_n_runs, title="KwikSort 100回実行中...") as _bar:
        for _seed in range(_n_runs):
            _ks_cache_name = f"seed_{_seed}"
            if has_cache(_ks_experiment, _ks_cache_name):
                _cached = load_results(_ks_experiment, _ks_cache_name)
                ks_multi_rankings[_seed] = _cached["ranking"]
                ks_multi_logs[_seed] = _cached["comparisons"]
            else:
                _comparison_log = []
                _rng = random.Random(_seed)
                _sorted_pms = kwiksort_cached(
                    list(pms),
                    pair_results,
                    comparison_log=_comparison_log,
                    rng=_rng,
                )
                _result = {
                    "ranking": [p["no"] for p in _sorted_pms],
                    "comparisons": _comparison_log,
                    "seed": _seed,
                    "num_comparisons": len(_comparison_log),
                }
                save_results(_ks_experiment, _ks_cache_name, _result)
                ks_multi_rankings[_seed] = _result["ranking"]
                ks_multi_logs[_seed] = _comparison_log
            _bar.update()
    return ks_multi_logs, ks_multi_rankings


@app.cell(hide_code=True)
def _(
    alt,
    calculate_cost,
    ks_multi_logs,
    ks_multi_rankings,
    mo,
    pair_results,
    pl,
    pms_by_no,
    wc_rank_map,
):
    from scipy.stats import kendalltau as _kendalltau

    _all_nos = sorted(pms_by_no.keys())
    _total_pairs = len(_all_nos) * (len(_all_nos) - 1) // 2

    _rank_data = []
    for _seed, _ranking in ks_multi_rankings.items():
        for _rank, _no in enumerate(_ranking):
            _rank_data.append({"seed": _seed, "no": _no, "rank": _rank + 1})

    _df = pl.DataFrame(_rank_data)
    _stats = (
        _df.group_by("no")
        .agg(
            pl.col("rank").mean().alias("mean_rank"),
            pl.col("rank").std().alias("std_rank"),
            pl.col("rank").min().alias("min_rank"),
            pl.col("rank").max().alias("max_rank"),
        )
        .sort("mean_rank")
        .with_columns(
            pl.col("no")
            .map_elements(lambda no: pms_by_no[no]["name"], return_dtype=pl.Utf8)
            .alias("name"),
            pl.col("no")
            .map_elements(lambda no: wc_rank_map[no], return_dtype=pl.Int64)
            .alias("wc_rank"),
        )
    )

    # 標準偏差のヒストグラム
    _hist_chart = (
        alt.Chart(_stats)
        .mark_bar()
        .encode(
            x=alt.X("std_rank:Q", bin=alt.Bin(step=1), title="順位の標準偏差"),
            y=alt.Y("count():Q", title="人数"),
        )
        .properties(
            title="KwikSort: 順位の標準偏差の分布（100回実行）",
            width=400,
            height=300,
        )
    )

    # 勝利数ソート順位 vs KwikSort順位の分布（箱ひげ図）
    # wc_rank を KwikSort と同じ方向（1=左派、64=右派）に変換
    _df_with_wc = _df.join(
        _stats.select("no", "wc_rank", "name"), on="no"
    ).with_columns((65 - pl.col("wc_rank")).alias("wc_rank_lr"))
    _boxplot = (
        alt.Chart(_df_with_wc)
        .mark_boxplot(size=8)
        .encode(
            x=alt.X(
                "wc_rank_lr:O",
                title="勝利数ソート順位（← 左派 ｜ 右派 →）",
                axis=alt.Axis(values=list(range(0, 65, 5))),
            ),
            y=alt.Y(
                "rank:Q",
                title="KwikSort順位（← 左派 ｜ 右派 →）",
            ),
            tooltip=["wc_rank_lr:O", "name:N"],
        )
    )
    _diag_data = pl.DataFrame(
        {"wc_rank_lr": list(range(1, 65)), "rank": list(range(1, 65))}
    )
    _diag_line = (
        alt.Chart(_diag_data)
        .mark_line(strokeDash=[4, 4], color="gray", opacity=0.5)
        .encode(
            x=alt.X("wc_rank_lr:O"),
            y=alt.Y("rank:Q"),
        )
    )
    _boxplot_chart = (_diag_line + _boxplot).properties(
        title="勝利数ソート順位 vs KwikSort順位のブレ幅（100回実行）",
        width=800,
        height=400,
    )

    # 各実行のコスト・比較回数・Kendall τ を計算
    _costs = []
    _n_comparisons = []
    _taus = []
    _wc_ranks = [wc_rank_map[_no] for _no in _all_nos]
    for _seed, _log in ks_multi_logs.items():
        _referenced = [pair_results[_e["no_a"]][_e["no_b"]] for _e in _log]
        _cost = calculate_cost(_referenced)
        _costs.append(_cost if _cost is not None else 0.0)
        _n_comparisons.append(len(_log))
        _ks_rank_map = {_no: _r + 1 for _r, _no in enumerate(ks_multi_rankings[_seed])}
        _ks_ranks = [_ks_rank_map.get(_no, -1) for _no in _all_nos]
        _tau, _ = _kendalltau(_wc_ranks, _ks_ranks)
        _taus.append(_tau)
    _avg_cost = sum(_costs) / len(_costs)
    _avg_comparisons = sum(_n_comparisons) / len(_n_comparisons)
    _avg_tau = sum(_taus) / len(_taus)

    _mean_std = _stats["std_rank"].mean()
    _max_std = _stats["std_rank"].max()
    _n_runs = len(ks_multi_rankings)

    mo.vstack(
        [
            mo.ui.altair_chart(_boxplot_chart),
            mo.ui.altair_chart(_hist_chart),
            mo.md(
                f"- 実行回数: **{_n_runs}回**\n"
                f"- 平均比較参照回数: **{_avg_comparisons:.0f}回**"
                f"（総当たり {_total_pairs:,} ペアの {_avg_comparisons / _total_pairs * 100:.1f}%）\n"
                f"- 平均コスト（相当）: **${_avg_cost:.4f}**\n"
                f"- 勝利数ソートとの平均 Kendall τ: **{_avg_tau:.2f}**\n"
                f"- 順位の標準偏差の平均: **{_mean_std:.1f}**\n"
                f"- 順位の標準偏差の最大: **{_max_std:.1f}**"
            ),
            mo.md("### ブレが大きい首相 Top 10"),
            mo.ui.table(
                _stats.sort("std_rank", descending=True)
                .head(10)
                .select(
                    "no",
                    "name",
                    "wc_rank",
                    "mean_rank",
                    "std_rank",
                    "min_rank",
                    "max_rank",
                ),
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ---
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 手法間の精度比較

    勝利数ソート（ベースライン）に対するリストワイズ法の Kendall τ を比較する。
    リストワイズの結果は `02_listwise.py` で取得済みのキャッシュを参照。

    ※ ポイントワイズ法は同率スコアが大量に発生し（64人中92%が同率）、順位付けが困難なため比較対象外とした。
    """)
    return


@app.cell(hide_code=True)
def _(criterion, has_cache, load_results, mo, pl, pms_by_no, wc_rank_map):
    import re as _re

    from scipy.stats import kendalltau as _kendalltau

    _all_nos = sorted(pms_by_no.keys())
    _wc_ranks = [wc_rank_map[no] for no in _all_nos]

    _rows = []

    # --- リストワイズ ---
    if has_cache("listwise", criterion.name):
        _lw_result = load_results("listwise", criterion.name)
        _lw_numbers = [int(x) for x in _re.findall(r"\d+", _lw_result["raw_response"])]
        _valid_nos = set(_all_nos)
        _lw_unique = []
        _seen = set()
        for n in _lw_numbers:
            if n in _valid_nos and n not in _seen:
                _lw_unique.append(n)
                _seen.add(n)
        if len(_lw_unique) == len(_all_nos):
            _lw_rank_map = {no: rank + 1 for rank, no in enumerate(_lw_unique)}
            _lw_ranks = [_lw_rank_map[no] for no in _all_nos]
            _tau, _ = _kendalltau(_wc_ranks, _lw_ranks)
            _rows.append(
                {
                    "手法": "リストワイズ（一括ランキング）",
                    "API呼出数": 1,
                    "Kendall τ": round(_tau, 2),
                }
            )

    # --- ペアワイズ総当たり（ベースライン）---
    _rows.append(
        {
            "手法": "ペアワイズ総当たり（ベースライン）",
            "API呼出数": len(_all_nos) * (len(_all_nos) - 1),
            "Kendall τ": 1.00,
        }
    )

    if len(_rows) <= 1:
        _output = mo.md(
            "リストワイズのキャッシュが見つかりません。"
            "先に `02_listwise.py` を実行してください。"
        )
    else:
        _df = pl.DataFrame(_rows)
        _output = mo.vstack(
            [
                mo.ui.table(_df),
                mo.md("※ KwikSort の Kendall τ は上の100回実行の結果を参照。"),
            ]
        )
    _output
    return


if __name__ == "__main__":
    app.run()

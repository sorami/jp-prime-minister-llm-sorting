import marimo

__generated_with = "0.19.11"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _():
    import asyncio

    import altair as alt
    import marimo as mo
    import polars as pl
    from dotenv import load_dotenv as _load_dotenv
    from openai import AsyncOpenAI

    from pm_sort.core.api import format_usage_summary
    from pm_sort.core.cache import has_cache, load_results, save_results
    from pm_sort.core.config import MAX_CONCURRENCY
    from pm_sort.core.data import load_prime_ministers
    from pm_sort.methods import score_pointwise

    _load_dotenv()
    return (
        AsyncOpenAI,
        MAX_CONCURRENCY,
        alt,
        asyncio,
        format_usage_summary,
        has_cache,
        load_prime_ministers,
        load_results,
        mo,
        pl,
        save_results,
        score_pointwise,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    # ポイントワイズ法（絶対評価）

    各首相に「左端（0点）↔ 右端（100点）」でスコアをつけさせる。

    10点満点では分解能が低く同率が大量発生するため、100点満点で実施する。
    64人中、同点がどれくらい発生するかを観察する。
    """)
    return


@app.cell
def _(load_prime_ministers):
    pms = load_prime_ministers()
    return (pms,)


@app.cell
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
def _(criterion, mo):
    mo.md(f"""
    ## スコアリング

    各首相に「{criterion.left}（0点）↔ {criterion.right}（100点）」でスコアをつけさせる。
    """)
    return


@app.cell
def _(criterion, mo):
    pointwise_run_btn = mo.ui.run_button(
        label=f"ポイントワイズ法を実行（{criterion.label_ja}）"
    )
    pointwise_run_btn
    return (pointwise_run_btn,)


@app.cell
async def _(
    AsyncOpenAI,
    MAX_CONCURRENCY,
    asyncio,
    criterion,
    has_cache,
    load_results,
    mo,
    pms,
    pointwise_run_btn,
    save_results,
    score_pointwise,
):
    mo.stop(not pointwise_run_btn.value)

    if has_cache("pointwise", criterion.name):
        pointwise_results = load_results("pointwise", criterion.name)
        mo.output.replace(mo.md(f"キャッシュから読み込み: {len(pointwise_results)}件"))
    else:
        _client = AsyncOpenAI()
        _sem = asyncio.Semaphore(MAX_CONCURRENCY)

        pointwise_results = [None] * len(pms)
        _completed = {"count": 0}
        with mo.status.progress_bar(
            total=len(pms), title="ポイントワイズ評価中..."
        ) as _bar:

            async def _run_one(_idx, _pm):
                _r = await score_pointwise(_client, _pm, criterion, semaphore=_sem)
                _result = {
                    "no": _r.no,
                    "score": _r.score,
                    "raw_response": _r.raw_response,
                    "usage": _r.usage.to_dict(),
                    "elapsed_seconds": round(_r.elapsed_seconds, 3),
                    "response_id": _r.response_id,
                    "model": _r.model,
                    "created_at": _r.created_at,
                    "reasoning_effort": _r.reasoning_effort,
                    "reasoning_summary": _r.reasoning_summary,
                }
                pointwise_results[_idx] = _result
                _completed["count"] += 1
                _bar.update()
                if _completed["count"] % 10 == 0:
                    _snapshot = [r for r in pointwise_results if r is not None]
                    save_results("pointwise", criterion.name, _snapshot)

            await asyncio.gather(*[_run_one(i, p) for i, p in enumerate(pms)])

        save_results("pointwise", criterion.name, pointwise_results)
        mo.output.replace(mo.md(f"完了: {len(pointwise_results)}件を保存"))
    return (pointwise_results,)


@app.cell
def _(format_usage_summary, mo, pl, pointwise_results):
    _n_total = len(pointwise_results)
    _df = pl.DataFrame(pointwise_results)
    _n_valid = _df.filter(pl.col("score").is_between(0, 100)).height
    _n_invalid = _n_total - _n_valid

    _usage_text = format_usage_summary(pointwise_results)

    mo.md(
        f"## 処理結果\n\n"
        f"- 総数: {_n_total}件（パース成功: {_n_valid}件、パース失敗: {_n_invalid}件）\n\n"
        + _usage_text
    )
    return


@app.cell
def _(alt, criterion, mo, pl, pointwise_results):
    _df = pl.DataFrame(pointwise_results).filter(pl.col("score").is_between(0, 100))

    _step = 5

    _chart = (
        alt.Chart(_df)
        .mark_bar()
        .encode(
            x=alt.X("score:Q", bin=alt.Bin(step=_step), title="スコア"),
            y=alt.Y("count():Q", title="人数"),
        )
        .properties(
            title=f"{criterion.label_ja}: スコア分布（{_step}点刻み）",
            width=500,
            height=300,
        )
    )

    mo.vstack([mo.md("## スコア分布"), mo.ui.altair_chart(_chart)])
    return


@app.cell(hide_code=True)
def _(criterion, mo, pl, pms, pointwise_results):
    _df = pl.DataFrame(pointwise_results).filter(pl.col("score").is_between(0, 100))
    _pms_lookup = {p["no"]: p["name"] for p in pms}

    _bottom5 = _df.sort("score", "no").head(5)
    _top5 = (
        _df.sort("score", descending=True)
        .sort("no", descending=False, maintain_order=True)
        .head(5)
        .sort("score", descending=True)
    )

    def _fmt_rows(rows):
        return "\n".join(
            f"| {i + 1} | {_pms_lookup.get(r['no'], str(r['no']))} | {r['score']}点 |"
            for i, r in enumerate(rows.iter_rows(named=True))
        )

    _header = "| # | 首相 | スコア |\n|---|------|--------|"

    mo.md(
        f"## {criterion.label_ja} スコア上位・下位\n\n"
        f"### 上位5人\n\n{_header}\n{_fmt_rows(_top5)}\n\n"
        f"### 下位5人\n\n{_header}\n{_fmt_rows(_bottom5)}"
    )
    return


@app.cell(hide_code=True)
def _(mo, pl, pms, pointwise_results):
    _df = pl.DataFrame(pointwise_results).filter(pl.col("score").is_between(0, 100))
    _pms_lookup = {p["no"]: p["name"] for p in pms}

    _top1 = _df.sort("score", descending=True).row(0, named=True)
    _bottom1 = _df.sort("score").row(0, named=True)

    def _fmt_response(row):
        name = _pms_lookup.get(row["no"], str(row["no"]))
        return f"**{name}**（{row['score']}点）\n\n{row['raw_response']}"

    mo.md(
        f"### レスポンス例\n\n"
        f"#### 最高スコア\n\n{_fmt_response(_top1)}\n\n---\n\n"
        f"#### 最低スコア\n\n{_fmt_response(_bottom1)}"
    )
    return


@app.cell
def _(mo, pl, pms, pointwise_results):
    _df = pl.DataFrame(pointwise_results).filter(pl.col("score").is_between(0, 100))
    _score_counts = _df.group_by("score").len().sort("score")
    _tied = _score_counts.filter(pl.col("len") > 1)
    _n_tied_scores = _tied.height

    _n_tied_people = _tied["len"].sum()
    _n_total = _df.height
    _tied_ratio = _n_tied_people / _n_total * 100 if _n_total > 0 else 0

    _pms_lookup = {p["no"]: p["name"] for p in pms}

    _lines = [
        f"## 同率スコア\n\n"
        f"同率が存在するスコア値: **{_n_tied_scores}個** / "
        f"単独でない人数: **{_n_tied_people}人**（全体{_n_total}人中 **{_tied_ratio:.1f}%**）\n"
    ]
    for row in _tied.iter_rows(named=True):
        _score = row["score"]
        _names = _df.filter(pl.col("score") == _score).sort("no")
        _name_list = ", ".join(
            _pms_lookup.get(r["no"], f"#{r['no']}")
            for r in _names.iter_rows(named=True)
        )
        _lines.append(f"- **{_score}点**（{row['len']}人）: {_name_list}")

    _unique = _score_counts.filter(pl.col("len") == 1)
    if _unique.height > 0:
        _lines.append(f"\n### 単独スコア（{_unique.height}人）\n")
        for row in _unique.iter_rows(named=True):
            _score = row["score"]
            _r = _df.filter(pl.col("score") == _score).row(0, named=True)
            _name = _pms_lookup.get(_r["no"], str(_r["no"]))
            _lines.append(f"- **{_score}点**: {_name}")

    mo.md("\n".join(_lines))
    return


if __name__ == "__main__":
    app.run()

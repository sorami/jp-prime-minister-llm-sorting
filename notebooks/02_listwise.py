import marimo

__generated_with = "0.19.11"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _():
    import re

    import marimo as mo
    from dotenv import load_dotenv as _load_dotenv
    from openai import AsyncOpenAI

    from pm_sort.api import format_usage_summary
    from pm_sort.cache import has_cache, load_results, save_results
    from pm_sort.compare import rank_listwise
    from pm_sort.data import load_prime_ministers

    _load_dotenv()
    return (
        AsyncOpenAI,
        format_usage_summary,
        has_cache,
        load_prime_ministers,
        load_results,
        mo,
        rank_listwise,
        re,
        save_results,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    # リストワイズ法（全要素一括評価）

    64人すべてを1プロンプトに入れ、「左端」→「右端」の順に並べ替えさせる。

    **仮説**: 出力が途中で切れる、番号が重複・欠落するなどのハルシネーションが発生する。
    """)
    return


@app.cell
def _(load_prime_ministers):
    pms = load_prime_ministers()
    return (pms,)


@app.cell
def _(mo):
    from pm_sort.criteria import CRITERIA
    from pm_sort.criteria import DEFAULT_CRITERION as _DEFAULT_CRITERION

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
    ## 一括ランキング

    64人すべてを1プロンプトに入れ、「{criterion.left}」→「{criterion.right}」の順に並べ替えさせる。
    """)
    return


@app.cell
def _(criterion, mo):
    listwise_run_btn = mo.ui.run_button(
        label=f"リストワイズ法を実行（{criterion.label_ja}）"
    )
    listwise_run_btn
    return (listwise_run_btn,)


@app.cell
async def _(
    AsyncOpenAI,
    criterion,
    has_cache,
    listwise_run_btn,
    load_results,
    mo,
    pms,
    rank_listwise,
    save_results,
):
    mo.stop(not listwise_run_btn.value)

    if has_cache("listwise", criterion.name):
        listwise_result = load_results("listwise", criterion.name)
        mo.output.replace(mo.md("キャッシュから読み込み"))
    else:
        _client = AsyncOpenAI()
        with mo.status.spinner(title="リストワイズ評価中..."):
            listwise_result = await rank_listwise(_client, pms, criterion)
        save_results("listwise", criterion.name, listwise_result)
    return (listwise_result,)


@app.cell
def _(format_usage_summary, listwise_result, mo, pms, re):
    _raw = listwise_result["raw_response"]

    ranked_numbers = [int(x) for x in re.findall(r"\d+", _raw)]
    _expected = {p["no"] for p in pms}
    _got = set(ranked_numbers)
    _missing = sorted(_expected - _got)
    _seen = set()
    _duplicates_unique = sorted(
        {n for n in ranked_numbers if n in _seen or _seen.add(n)}  # type: ignore[func-returns-value]
    )

    mo.md(
        f"## 分析結果\n\n"
        f"- 出力に含まれる番号の数: **{len(ranked_numbers)}**（期待値: {len(pms)}）\n"
        f"- ユニーク数: **{len(_got)}**\n"
        f"- 欠落した番号: **{len(_missing)}件** {_missing[:20]}{'...' if len(_missing) > 20 else ''}\n"
        f"- 重複した番号: **{len(_duplicates_unique)}件** {_duplicates_unique[:20]}{'...' if len(_duplicates_unique) > 20 else ''}\n\n"
        + format_usage_summary([listwise_result], calls_label="1回呼び出し")
    )
    return (ranked_numbers,)


@app.cell
def _(criterion, mo, pms, ranked_numbers):
    _no_to_name = {p["no"]: p["name"] for p in pms}
    _ranked = [
        {"rank": i + 1, "no": n, "name": _no_to_name.get(n, "?")}
        for i, n in enumerate(ranked_numbers)
        if n in _no_to_name
    ]
    _top5 = _ranked[:5]
    _bottom5 = _ranked[-5:]

    _top_rows = "\n".join(f"| {r['rank']} | {r['name']} |" for r in _top5)
    _bottom_rows = "\n".join(f"| {r['rank']} | {r['name']} |" for r in _bottom5)

    mo.md(
        f"### {criterion.left}寄り Top 5\n\n"
        f"| 順位 | 名前 |\n| ---: | :--- |\n{_top_rows}\n\n"
        f"### {criterion.right}寄り Top 5\n\n"
        f"| 順位 | 名前 |\n| ---: | :--- |\n{_bottom_rows}"
    )
    return


@app.cell(hide_code=True)
def _(listwise_result, mo):
    _summary = listwise_result.get("reasoning_summary", "")
    if _summary:
        _result = mo.md(f"### 推論サマリー\n\n{_summary}")
    else:
        _result = mo.md("### 推論サマリー\n\n（サマリーなし）")
    _result
    return


@app.cell(hide_code=True)
def _(listwise_result, mo):
    _raw = listwise_result["raw_response"]
    mo.md(f"### 生レスポンス（先頭500文字）\n\n```\n{_raw[:500]}\n```")
    return


if __name__ == "__main__":
    app.run()

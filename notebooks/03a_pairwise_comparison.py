import marimo

__generated_with = "0.19.11"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _():
    import asyncio
    from itertools import combinations
    from math import comb

    import altair as alt
    import marimo as mo
    import polars as pl
    from dotenv import load_dotenv as _load_dotenv
    from openai import AsyncOpenAI

    from pm_sort.core.api import format_usage_summary
    from pm_sort.core.cache import (
        has_cache,
        load_results,
        nested_int_keys,
        save_results,
    )
    from pm_sort.core.config import MAX_CONCURRENCY
    from pm_sort.core.data import load_prime_ministers
    from pm_sort.methods.pairwise import (
        compare_pair,
        find_transitivity_violations,
        resolve_winner,
    )

    _load_dotenv()
    return (
        AsyncOpenAI,
        MAX_CONCURRENCY,
        asyncio,
        comb,
        combinations,
        compare_pair,
        find_transitivity_violations,
        format_usage_summary,
        has_cache,
        load_prime_ministers,
        load_results,
        mo,
        nested_int_keys,
        resolve_winner,
        save_results,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    # ペアワイズ法（二項比較）と「矛盾」の観察

    人間と同じように「2つを比較する」のがLLMにとって最も安定する手法。
    ただし、LLM特有の「論理破綻」が存在しうる。

    - 64人の組み合わせ C(64,2) = **2,016ペア** を総当たりで比較
    - ポジションバイアス検証のため、各ペアを**両方向**（計4,032回）で実行
    """)
    return


@app.cell
def _(load_prime_ministers):
    pms = load_prime_ministers()
    pms_by_no = {p["no"]: p for p in pms}
    return (pms_by_no,)


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
def _(mo):
    mo.md("""
    ---
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 全ペア総当たり比較
    """)
    return


@app.cell
def _(criterion, mo):
    pairwise_run_btn = mo.ui.run_button(
        label=f"全ペア比較を実行（{criterion.label_ja}）"
    )
    pairwise_run_btn
    return (pairwise_run_btn,)


@app.cell(hide_code=True)
async def _(
    AsyncOpenAI,
    MAX_CONCURRENCY,
    asyncio,
    combinations,
    compare_pair,
    criterion,
    has_cache,
    load_results,
    mo,
    nested_int_keys,
    pairwise_run_btn,
    pms_by_no,
    save_results,
):
    mo.stop(not pairwise_run_btn.value)

    _all_nos = sorted(pms_by_no.keys())

    if has_cache("pairwise", criterion.name):
        pair_results = nested_int_keys(load_results("pairwise", criterion.name))
    else:
        pair_results = {}

    # 未取得のペアを抽出（両方向とも存在するか確認）
    _all_pairs = list(combinations(_all_nos, 2))
    _remaining = [
        (_a, _b)
        for _a, _b in _all_pairs
        if _a not in pair_results
        or _b not in pair_results.get(_a, {})
        or _b not in pair_results
        or _a not in pair_results.get(_b, {})
    ]

    if not _remaining:
        _n_comparisons = sum(len(v) for v in pair_results.values())
        mo.output.replace(
            mo.md(
                f"キャッシュから読み込み: **{_n_comparisons}件**（{len(_all_pairs)}ペア × 両方向）"
            )
        )
    else:
        _client = AsyncOpenAI()
        _sem = asyncio.Semaphore(MAX_CONCURRENCY)
        _batch_size = 100

        with mo.status.progress_bar(
            total=len(_all_pairs),
            title="全ペア比較中...",
        ) as _bar:
            _bar.update(increment=len(_all_pairs) - len(_remaining))
            for _batch_start in range(0, len(_remaining), _batch_size):
                _batch = _remaining[_batch_start : _batch_start + _batch_size]
                _tasks = []
                for _a, _b in _batch:
                    _tasks.append(
                        compare_pair(
                            _client,
                            pms_by_no[_a],
                            pms_by_no[_b],
                            criterion,
                            semaphore=_sem,
                        )
                    )
                    _tasks.append(
                        compare_pair(
                            _client,
                            pms_by_no[_b],
                            pms_by_no[_a],
                            criterion,
                            semaphore=_sem,
                        )
                    )
                _batch_results = await asyncio.gather(*_tasks)
                for _result in _batch_results:
                    pair_results.setdefault(_result.no_a, {})[_result.no_b] = (
                        _result.to_dict()
                    )
                _bar.update(increment=len(_batch))
                save_results("pairwise", criterion.name, pair_results)

        _n_comparisons = sum(len(v) for v in pair_results.values())
        mo.output.replace(mo.md(f"完了: **{_n_comparisons}件**を保存"))
    return (pair_results,)


@app.cell(hide_code=True)
def _(combinations, format_usage_summary, mo, pair_results, resolve_winner):
    _all_nos = sorted(pair_results.keys())
    _all_pairs = list(combinations(_all_nos, 2))
    _total = len(_all_pairs)

    _a_wins = 0
    _b_wins = 0
    _ties = 0
    for _a, _b in _all_pairs:
        _result = resolve_winner(pair_results, _a, _b)
        if _result == "A":
            _a_wins += 1
        elif _result == "B":
            _b_wins += 1
        else:
            _ties += 1

    # 全比較結果をフラットにして format_usage_summary に渡す
    _all_results = [r for inner in pair_results.values() for r in inner.values()]

    mo.md(
        f"### 集計\n\n"
        f"- 総ペア数: **{_total}**\n"
        f"- A（先出し）勝ち: **{_a_wins}** ({_a_wins / _total * 100:.1f}%)\n"
        f"- B（後出し）勝ち: **{_b_wins}** ({_b_wins / _total * 100:.1f}%)\n"
        f"- TIE（不一致）: **{_ties}** ({_ties / _total * 100:.1f}%)\n\n"
        + format_usage_summary(
            _all_results,
            calls_label=f"{_total}ペア × 両方向 = {_total * 2}回呼び出し",
        )
    )
    return


@app.cell(hide_code=True)
def _(mo, pair_results, pms_by_no):
    _invalid_entries = []
    for _a in sorted(pair_results.keys()):
        for _b in sorted(pair_results[_a].keys()):
            if pair_results[_a][_b]["winner"] not in ("A", "B"):
                _invalid_entries.append((_a, _b, pair_results[_a][_b]))

    _n_total = sum(len(v) for v in pair_results.values())

    if _invalid_entries:
        _lines = []
        for _a, _b, _entry in _invalid_entries:
            _name_a = pms_by_no[_a]["name"]
            _name_b = pms_by_no[_b]["name"]
            # raw_response の末尾3行を表示
            _tail = "\n".join(_entry["raw_response"].strip().splitlines()[-3:])
            _lines.append(
                f"- **{_name_a}** vs **{_name_b}** — winner=`{_entry['winner']}`\n"
                f"  > ```\n  > {_tail}\n  > ```"
            )
        mo.md(
            f"### INVALID な比較結果\n\n"
            f"AでもBでもない回答が **{len(_invalid_entries)}件** / {_n_total}件 検出された。\n"
            f"パーサーが「回答: A」形式を見つけられなかったケース。\n"
            f"勝利数ソートではTIE扱い、KwikSortでは同等扱いとなる。\n\n"
            + "\n".join(_lines)
        )
    else:
        mo.md(f"INVALID な比較結果: **0件** / {_n_total}件")
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
    ## ポジションバイアスの検証

    同じペアでも提示順を入れ替えると結果が変わる（先に提示された方を選びやすい等）割合を検証する。
    """)
    return


@app.cell(hide_code=True)
def _(combinations, mo, pair_results, pms_by_no):
    _all_nos = sorted(pair_results.keys())
    _all_pairs = list(combinations(_all_nos, 2))
    _total = len(_all_pairs)

    # 各ペアの llm(a,b) と llm(b,a) の結果を集計
    _counts = {"AA": 0, "AB": 0, "BA": 0, "BB": 0}
    _person_inconsistent: dict[int, int] = {}
    for _a, _b in _all_pairs:
        _w_ab = pair_results[_a][_b]["winner"]  # llm(a,b)
        _w_ba = pair_results[_b][_a]["winner"]  # llm(b,a)
        _key = _w_ab + _w_ba
        if _key in _counts:
            _counts[_key] += 1
        if _w_ab == _w_ba:  # 不一致
            _person_inconsistent[_a] = _person_inconsistent.get(_a, 0) + 1
            _person_inconsistent[_b] = _person_inconsistent.get(_b, 0) + 1

    _consistent = _counts["BA"] + _counts["AB"]
    _inconsistent = _counts["AA"] + _counts["BB"]

    # 先出し勝率: llm(a,b) でA勝ち + llm(b,a) でA勝ちの合計を全4,032回中で
    _first_win_count = (_counts["AA"] + _counts["AB"]) + (_counts["AA"] + _counts["BA"])
    _first_win_pct = _first_win_count / (_total * 2) * 100

    # 不一致登場回数 Top 10
    _top_inconsistent = sorted(_person_inconsistent.items(), key=lambda x: -x[1])[:10]
    _top_lines = "\n".join(
        f"| {pms_by_no[_no]['name']} | {_cnt} |" for _no, _cnt in _top_inconsistent
    )

    mo.md(
        f"### 結果の一致・不一致\n\n"
        f"各ペアについて llm(a,b) と llm(b,a) の結果が一致するか（同じ人物を勝者とするか）を確認する。\n\n"
        f"- **両方向で一致**: **{_consistent}件** ({_consistent / _total * 100:.1f}%)\n"
        f"  - llm(a,b)=後出し勝ち, llm(b,a)=先出し勝ち → aの勝ち: **{_counts['BA']}件**\n"
        f"  - llm(a,b)=先出し勝ち, llm(b,a)=後出し勝ち → bの勝ち: **{_counts['AB']}件**\n"
        f"- **両方向で不一致（TIE扱い）**: **{_inconsistent}件** ({_inconsistent / _total * 100:.1f}%)\n"
        f"  - 両方とも先出し勝ち: **{_counts['AA']}件** — 提示順に引きずられている\n"
        f"  - 両方とも後出し勝ち: **{_counts['BB']}件** — 提示順に引きずられている\n\n"
        f"全 **{_total * 2:,}回** の比較中、先出しが勝った割合: **{_first_win_pct:.1f}%**\n\n"
        f"### 不一致に多く登場する首相 Top 10\n\n"
        f"| 首相名 | 不一致回数 |\n"
        f"| :--- | ---: |\n" + _top_lines
    )
    return


@app.cell(hide_code=True)
def _(combinations, criterion, mo, pair_results, pms_by_no, save_results):
    _all_nos = sorted(pair_results.keys())
    _all_pairs = list(combinations(_all_nos, 2))

    _inconsistent_details = []
    for _a, _b in _all_pairs:
        _w_ab = pair_results[_a][_b]["winner"]
        _w_ba = pair_results[_b][_a]["winner"]
        if _w_ab == _w_ba:  # AA or BB → 不一致
            _inconsistent_details.append(
                {
                    "no_a": _a,
                    "name_a": pms_by_no[_a]["name"],
                    "no_b": _b,
                    "name_b": pms_by_no[_b]["name"],
                    "pattern": _w_ab + _w_ba,
                    "llm_a_b": {
                        "winner": _w_ab,
                        "raw_response": pair_results[_a][_b]["raw_response"],
                    },
                    "llm_b_a": {
                        "winner": _w_ba,
                        "raw_response": pair_results[_b][_a]["raw_response"],
                    },
                }
            )

    _path = save_results(
        "pairwise", criterion.name, _inconsistent_details, "_inconsistent"
    )
    mo.md(
        f"不一致ペアの詳細（両方向の raw_response 付き）を保存: **{len(_inconsistent_details)}件** → `{_path.name}`"
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
    ## 推移律の崩壊（三すくみ）

    「AよりBが右派」「BよりCが右派」なのに「CよりAが右派」となる矛盾サイクルの検出。

    ここでは**長さ3のサイクル（三すくみ）** を検出する。
    A>B>C>D>A のような長いサイクルも理論上存在しうるが、
    長いサイクルは必ず長さ3のサイクルを含む
    （例: A>B>C>D>A で A vs C の結果がどちらでも長さ3のサイクルが出現する）。
    つまり三すくみが0件なら、どんな長さの矛盾サイクルも存在しない。
    """)
    return


@app.cell
def _(find_transitivity_violations, pair_results):
    violations = find_transitivity_violations(pair_results)
    return (violations,)


@app.cell(hide_code=True)
def _(comb, mo, pms_by_no, violations):
    _all_nos = sorted(pms_by_no.keys())
    _total_triples = comb(len(_all_nos), 3)
    _num_violations = len(violations)
    _pct = _num_violations / _total_triples * 100 if _total_triples > 0 else 0

    _examples = violations[:5]
    _example_texts = []
    for _a, _b, _c in _examples:
        _pa = pms_by_no[_a]
        _pb = pms_by_no[_b]
        _pc = pms_by_no[_c]
        _example_texts.append(
            f"- {_pa['name']} > {_pb['name']} > {_pc['name']} > {_pa['name']}"
        )

    # 人物別の三すくみ登場回数
    _person_violation: dict[int, int] = {}
    for _a, _b, _c in violations:
        for _no in (_a, _b, _c):
            _person_violation[_no] = _person_violation.get(_no, 0) + 1
    _top_violation = sorted(_person_violation.items(), key=lambda x: -x[1])
    _top_violation_lines = "\n".join(
        f"| {pms_by_no[_no]['name']} | {_cnt} |" for _no, _cnt in _top_violation
    )

    mo.md(
        f"### 三すくみの検出結果\n\n"
        f"- 全3つ組の数: **{_total_triples:,}**\n"
        f"- 三すくみ（ユニークな矛盾サイクル）の数: **{_num_violations:,}**\n"
        f"- 矛盾サイクル率: **{_pct:.2f}%**\n\n"
        f"### 具体例\n\n"
        + "\n".join(_example_texts)
        + f"\n\n### 三すくみに登場する首相\n\n"
        f"| 首相名 | 登場回数 |\n"
        f"| :--- | ---: |\n" + _top_violation_lines
    )
    return


@app.cell(hide_code=True)
def _(
    criterion,
    mo,
    pair_results,
    pms_by_no,
    resolve_winner,
    save_results,
    violations,
):
    _violation_details = []
    for _a, _b, _c in violations:
        # サイクル: a>b>c>a
        _violation_details.append(
            {
                "cycle": [_a, _b, _c],
                "names": [
                    pms_by_no[_a]["name"],
                    pms_by_no[_b]["name"],
                    pms_by_no[_c]["name"],
                ],
                "a_vs_b": {
                    "resolve_winner": resolve_winner(pair_results, _a, _b),
                    "llm_a_b": {
                        "winner": pair_results[_a][_b]["winner"],
                        "raw_response": pair_results[_a][_b]["raw_response"],
                    },
                    "llm_b_a": {
                        "winner": pair_results[_b][_a]["winner"],
                        "raw_response": pair_results[_b][_a]["raw_response"],
                    },
                },
                "b_vs_c": {
                    "resolve_winner": resolve_winner(pair_results, _b, _c),
                    "llm_b_c": {
                        "winner": pair_results[_b][_c]["winner"],
                        "raw_response": pair_results[_b][_c]["raw_response"],
                    },
                    "llm_c_b": {
                        "winner": pair_results[_c][_b]["winner"],
                        "raw_response": pair_results[_c][_b]["raw_response"],
                    },
                },
                "c_vs_a": {
                    "resolve_winner": resolve_winner(pair_results, _c, _a),
                    "llm_c_a": {
                        "winner": pair_results[_c][_a]["winner"],
                        "raw_response": pair_results[_c][_a]["raw_response"],
                    },
                    "llm_a_c": {
                        "winner": pair_results[_a][_c]["winner"],
                        "raw_response": pair_results[_a][_c]["raw_response"],
                    },
                },
            }
        )

    _path = save_results("pairwise", criterion.name, _violation_details, "_violations")
    mo.md(
        f"三すくみの詳細（各ペアの raw_response 付き）を保存: **{len(_violation_details)}件** → `{_path.name}`"
    )
    return


if __name__ == "__main__":
    app.run()

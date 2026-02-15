from itertools import combinations


def resolve_winner(pair_results: dict, a: int, b: int) -> str:
    """両方向の比較結果から最終勝者を導出する。

    pair_results[a][b] と pair_results[b][a] の winner を照合し、
    一致すれば勝者を、不一致なら "TIE" を返す。

    返り値:
        "A" — a が勝ち（より右寄り）
        "B" — b が勝ち（より右寄り）
        "TIE" — 両方向で不一致
    """
    winner_ab = pair_results[a][b]["winner"]
    winner_ba = pair_results[b][a]["winner"]
    if winner_ab == "A" and winner_ba == "B":
        return "A"
    elif winner_ab == "B" and winner_ba == "A":
        return "B"
    return "TIE"


def win_count_sort(pair_results: dict) -> list[tuple[int, int, float]]:
    """全ペア比較データから勝利数でソートする。

    同じ勝利数の人物には同じ順位を付与する（標準競技順位方式: 1, 2, 2, 4, ...）。

    Returns:
        [(no, rank, wins), ...] を勝利数降順で返す。
    """
    all_nos = sorted(pair_results.keys())
    wins: dict[int, float] = {no: 0.0 for no in all_nos}
    for a, b in combinations(all_nos, 2):
        result = resolve_winner(pair_results, a, b)
        if result == "A":
            wins[a] += 1
        elif result == "B":
            wins[b] += 1
        else:
            wins[a] += 0.5
            wins[b] += 0.5
    sorted_items = sorted(wins.items(), key=lambda x: (-x[1], x[0]))

    # 標準競技順位: 同じ勝利数には同じ順位、次は飛ばす
    result_list: list[tuple[int, int, float]] = []
    for i, (no, w) in enumerate(sorted_items):
        if i == 0 or w != sorted_items[i - 1][1]:
            current_rank = i + 1
        result_list.append((no, current_rank, w))
    return result_list


def find_transitivity_violations(
    pair_results: dict,
) -> list[tuple[int, int, int]]:
    """a>b, b>c, c>a となる三すくみサイクルを検出する。

    各サイクルは1回だけカウントされ、最小要素が先頭になるよう正規化される。
    """
    # 勝敗グラフを構築
    wins: dict[int, set[int]] = {}
    all_nos = sorted(pair_results.keys())
    for a, b in combinations(all_nos, 2):
        result = resolve_winner(pair_results, a, b)
        if result == "A":
            wins.setdefault(a, set()).add(b)
        elif result == "B":
            wins.setdefault(b, set()).add(a)

    violations: set[tuple[int, int, int]] = set()
    for a, b, c in combinations(all_nos, 3):
        a_wins = wins.get(a, set())
        b_wins = wins.get(b, set())
        c_wins = wins.get(c, set())
        # a > b > c > a のサイクルをチェック
        if b in a_wins and c in b_wins and a in c_wins:
            cycle = (a, b, c)
            min_val = min(cycle)
            idx = cycle.index(min_val)
            normalized = cycle[idx:] + cycle[:idx]
            violations.add(normalized)
        # a > c > b > a のサイクルをチェック
        if c in a_wins and b in c_wins and a in b_wins:
            cycle = (a, c, b)
            min_val = min(cycle)
            idx = cycle.index(min_val)
            normalized = cycle[idx:] + cycle[:idx]
            violations.add(normalized)

    return sorted(violations)

from itertools import combinations


def win_count_sort(pair_results: list[dict]) -> list[tuple[int, float]]:
    """全ペア比較データから勝利数でソートする。[(no, wins), ...] を降順で返す。"""
    all_nos = set()
    for r in pair_results:
        all_nos.add(r["no_a"])
        all_nos.add(r["no_b"])
    wins: dict[int, float] = {i: 0.0 for i in all_nos}
    for r in pair_results:
        if r["final_winner"] == "A":
            wins[r["no_a"]] += 1
        elif r["final_winner"] == "B":
            wins[r["no_b"]] += 1
        else:  # TIE
            wins[r["no_a"]] += 0.5
            wins[r["no_b"]] += 0.5
    return sorted(wins.items(), key=lambda x: (-x[1], x[0]))


def find_transitivity_violations(
    pair_results: list[dict],
) -> list[tuple[int, int, int]]:
    """a>b, b>c, c>a となる三すくみサイクルを検出する。

    各サイクルは1回だけカウントされ、最小要素が先頭になるよう正規化される。
    """
    wins: dict[int, set[int]] = {}
    for r in pair_results:
        a, b = r["no_a"], r["no_b"]
        if r["final_winner"] == "A":
            wins.setdefault(a, set()).add(b)
        elif r["final_winner"] == "B":
            wins.setdefault(b, set()).add(a)

    all_nos = sorted(
        set(r["no_a"] for r in pair_results) | set(r["no_b"] for r in pair_results)
    )

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

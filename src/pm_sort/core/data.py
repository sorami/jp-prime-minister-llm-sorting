import csv
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent.parent.parent / "data"


def load_prime_ministers() -> list[dict]:
    """歴代首相のデータを読み込み"""
    with open(DATA_DIR / "prime_ministers.csv") as f:
        pms = list(csv.DictReader(f))
        for p in pms:
            p["no"] = int(p["no"])
    return pms

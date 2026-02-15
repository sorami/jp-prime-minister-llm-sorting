import asyncio
import re
from dataclasses import dataclass, field

from openai import AsyncOpenAI

from ...core.api import (
    Usage,
    call_with_retry,
    extract_reasoning_summary,
    extract_usage,
    maybe_acquire,
)
from ...core.config import get_model
from ...core.criteria import Criterion


@dataclass
class PairwiseResult:
    """ペアワイズ比較の結果。"""

    no_a: int
    no_b: int
    winner: str  # "A", "B", or "INVALID"
    raw_response: str
    prompt: str
    usage: Usage = field(default_factory=Usage)
    elapsed_seconds: float = 0.0
    response_id: str = ""
    model: str = ""
    created_at: str = ""
    reasoning_effort: str = ""
    reasoning_summary: str = ""


def _parse_winner(text: str) -> str:
    """CoTレスポンスから勝者（AまたはB）をパースする。

    「回答: A」「回答: B」パターンを優先し、
    見つからなければ最終行をフォールバックとして確認する。
    """
    match = re.search(r"回答[：:]\s*([ABab])", text)
    if match:
        return match.group(1).upper()
    for line in reversed(text.strip().splitlines()):
        line = line.strip()
        if line in ("A", "B", "a", "b"):
            return line.upper()
    return "INVALID"


async def compare_pair(
    client: AsyncOpenAI,
    pm_a: dict,
    pm_b: dict,
    criterion: Criterion,
    *,
    semaphore: asyncio.Semaphore | None = None,
) -> PairwiseResult:
    """2人の首相を指定基準でChain of Thoughtにより比較する。"""
    prompt = (
        f"以下の2人の内閣総理大臣を「{criterion.left} ↔ {criterion.right}」の軸で比較してください。\n"
        f"{criterion.description}\n\n"
        f"【A】{pm_a['name']}\n"
        f"【B】{pm_b['name']}\n\n"
        f"それぞれの人物についてこの軸に関する考察を簡潔に述べた上で、\n"
        f"最後の行に「回答: A」または「回答: B」と、より「{criterion.right}」寄りの人物を回答してください。"
    )

    async with maybe_acquire(semaphore):
        r, elapsed, effort = await call_with_retry(
            client, model=get_model(), input=prompt
        )

    raw = r.output_text or ""
    winner = _parse_winner(raw)
    return PairwiseResult(
        no_a=pm_a["no"],
        no_b=pm_b["no"],
        winner=winner,
        raw_response=raw,
        prompt=prompt,
        usage=extract_usage(r),
        elapsed_seconds=elapsed,
        response_id=r.id,
        model=r.model,
        created_at=str(r.created_at),
        reasoning_effort=effort,
        reasoning_summary=extract_reasoning_summary(r),
    )


async def compare_pair_bidirectional(
    client: AsyncOpenAI,
    pm_a: dict,
    pm_b: dict,
    criterion: Criterion,
    *,
    semaphore: asyncio.Semaphore | None = None,
) -> dict:
    """llm(a,b) と llm(b,a) の両方向で比較し、不一致ならTIEとする。"""
    r_ab, r_ba = await asyncio.gather(
        compare_pair(client, pm_a, pm_b, criterion, semaphore=semaphore),
        compare_pair(client, pm_b, pm_a, criterion, semaphore=semaphore),
    )

    if r_ab.winner == "A" and r_ba.winner == "B":
        final_winner = "A"
    elif r_ab.winner == "B" and r_ba.winner == "A":
        final_winner = "B"
    else:
        final_winner = "TIE"

    combined_usage = r_ab.usage + r_ba.usage
    combined_elapsed = r_ab.elapsed_seconds + r_ba.elapsed_seconds
    return {
        "no_a": pm_a["no"],
        "no_b": pm_b["no"],
        "winner_ab": r_ab.winner,
        "winner_ba": r_ba.winner,
        "final_winner": final_winner,
        "response_ab": r_ab.raw_response,
        "response_ba": r_ba.raw_response,
        "usage": combined_usage.to_dict(),
        "elapsed_seconds": round(combined_elapsed, 3),
        "response_id_ab": r_ab.response_id,
        "response_id_ba": r_ba.response_id,
        "model_ab": r_ab.model,
        "model_ba": r_ba.model,
        "created_at_ab": str(r_ab.created_at),
        "created_at_ba": str(r_ba.created_at),
        "reasoning_effort": r_ab.reasoning_effort,
        "reasoning_summary_ab": r_ab.reasoning_summary,
        "reasoning_summary_ba": r_ba.reasoning_summary,
    }

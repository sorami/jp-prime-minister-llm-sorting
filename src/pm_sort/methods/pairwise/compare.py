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

    def to_dict(self) -> dict:
        return {
            "no_a": self.no_a,
            "no_b": self.no_b,
            "winner": self.winner,
            "raw_response": self.raw_response,
            "prompt": self.prompt,
            "usage": self.usage.to_dict(),
            "elapsed_seconds": round(self.elapsed_seconds, 3),
            "response_id": self.response_id,
            "model": self.model,
            "created_at": self.created_at,
            "reasoning_effort": self.reasoning_effort,
            "reasoning_summary": self.reasoning_summary,
        }


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

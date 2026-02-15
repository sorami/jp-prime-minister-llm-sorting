import asyncio
import re
from dataclasses import dataclass, field

from openai import AsyncOpenAI

from ..api import (
    Usage,
    call_with_retry,
    extract_reasoning_summary,
    extract_usage,
    maybe_acquire,
)
from ..config import get_model
from ..criteria import Criterion


@dataclass
class PointwiseResult:
    """ポイントワイズ評価の結果。"""

    no: int
    score: int  # 0-100、パース失敗時は -1
    raw_response: str
    usage: Usage = field(default_factory=Usage)
    elapsed_seconds: float = 0.0
    response_id: str = ""
    model: str = ""
    created_at: str = ""
    reasoning_effort: str = ""
    reasoning_summary: str = ""


async def score_pointwise(
    client: AsyncOpenAI,
    pm: dict,
    criterion: Criterion,
    *,
    semaphore: asyncio.Semaphore | None = None,
) -> PointwiseResult:
    """1人の首相を指定基準で0〜100点のスコアで評価する。"""
    prompt = (
        f"以下の内閣総理大臣を「{criterion.left} ↔ {criterion.right}」の軸で0〜100点で評価してください。\n"
        f"{criterion.description}\n\n"
        f"{pm['name']}\n\n"
        f"この人物についてこの軸に関する考察を簡潔に述べた上で、\n"
        f"最後の行に「スコア: X」と、{criterion.left}寄りなら0点、{criterion.right}寄りなら100点として数字で回答してください。"
    )

    async with maybe_acquire(semaphore):
        r, elapsed, effort = await call_with_retry(
            client, model=get_model(), input=prompt
        )

    raw = (r.output_text or "").strip()
    match = re.search(r"スコア[：:]\s*(\d+)", raw)
    if match:
        score = int(match.group(1))
    else:
        try:
            score = int(raw.splitlines()[-1].strip())
        except (ValueError, IndexError):
            score = -1
    return PointwiseResult(
        no=pm["no"],
        score=score,
        raw_response=raw,
        usage=extract_usage(r),
        elapsed_seconds=elapsed,
        response_id=r.id,
        model=r.model,
        created_at=str(r.created_at),
        reasoning_effort=effort,
        reasoning_summary=extract_reasoning_summary(r),
    )

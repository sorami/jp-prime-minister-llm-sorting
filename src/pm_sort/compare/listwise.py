from openai import AsyncOpenAI

from ..api import call_with_retry, extract_reasoning_summary, extract_usage
from ..config import get_model
from ..criteria import Criterion


async def rank_listwise(
    client: AsyncOpenAI,
    pms: list[dict],
    criterion: Criterion,
) -> dict:
    """全員を1プロンプトに入れてソートさせる。"""
    pms_text = "\n".join(f"{p['no']}. {p['name']}" for p in pms)
    prompt = (
        f"以下の{len(pms)}人の内閣総理大臣を「{criterion.left} ↔ {criterion.right}」の軸で並べ替えてください。\n"
        f"{criterion.description}\n\n"
        f"{pms_text}\n\n"
        f"{criterion.left}寄りの人物から{criterion.right}寄りの人物の順に、番号のみをカンマ区切りで出力してください。"
    )

    r, elapsed, effort = await call_with_retry(client, model=get_model(), input=prompt)
    return {
        "raw_response": r.output_text,
        "prompt": prompt,
        "usage": extract_usage(r).to_dict(),
        "elapsed_seconds": round(elapsed, 3),
        "response_id": r.id,
        "model": r.model,
        "created_at": str(r.created_at),
        "reasoning_effort": effort,
        "reasoning_summary": extract_reasoning_summary(r),
    }

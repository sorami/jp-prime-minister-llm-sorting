import asyncio
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass

from openai import APIError, AsyncOpenAI, RateLimitError

from ..config import BASE_DELAY, DEFAULT_REASONING_EFFORT, MAX_RETRIES


@dataclass
class Usage:
    """1回のAPI呼び出しにおけるトークン使用量。

    reasoning_tokens は output_tokens の内数。
    """

    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    reasoning_tokens: int = 0

    def to_dict(self) -> dict:
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "reasoning_tokens": self.reasoning_tokens,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Usage":
        return cls(
            input_tokens=d.get("input_tokens", 0),
            output_tokens=d.get("output_tokens", 0),
            total_tokens=d.get("total_tokens", 0),
            reasoning_tokens=d.get("reasoning_tokens", 0),
        )

    def __add__(self, other: "Usage") -> "Usage":
        return Usage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            total_tokens=self.total_tokens + other.total_tokens,
            reasoning_tokens=self.reasoning_tokens + other.reasoning_tokens,
        )


def extract_usage(response) -> Usage:
    """APIレスポンスからトークン使用量を抽出する。"""
    u = getattr(response, "usage", None)
    if u is None:
        return Usage()
    reasoning = 0
    details = getattr(u, "output_tokens_details", None)
    if details:
        reasoning = getattr(details, "reasoning_tokens", 0) or 0
    return Usage(
        input_tokens=u.input_tokens,
        output_tokens=u.output_tokens,
        total_tokens=getattr(u, "total_tokens", u.input_tokens + u.output_tokens),
        reasoning_tokens=reasoning,
    )


@asynccontextmanager
async def maybe_acquire(semaphore: asyncio.Semaphore | None):
    """セマフォがあれば取得し、なければそのまま通過する。"""
    if semaphore:
        async with semaphore:
            yield
    else:
        yield


async def call_with_retry(client: AsyncOpenAI, **kwargs) -> tuple:
    """リトライ付きでAPIを呼び出す。(response, elapsed_seconds, reasoning_effort) を返す。"""
    kwargs.setdefault("reasoning", {"effort": DEFAULT_REASONING_EFFORT})
    reasoning_effort = kwargs.get("reasoning", {}).get("effort", "")
    for attempt in range(MAX_RETRIES):
        try:
            t0 = time.monotonic()
            r = await client.responses.create(**kwargs)
            elapsed = time.monotonic() - t0
            return r, elapsed, reasoning_effort
        except RateLimitError as e:
            if "insufficient_quota" in str(e):
                raise RuntimeError(
                    "OpenAI APIのクォータ（残高）が不足しています。"
                    " https://platform.openai.com/settings/organization/billing を確認してください。"
                ) from e
            delay = BASE_DELAY * (2**attempt)
            await asyncio.sleep(delay)
        except APIError:
            if attempt == MAX_RETRIES - 1:
                raise
            await asyncio.sleep(BASE_DELAY)
    raise RuntimeError("Max retries exceeded")

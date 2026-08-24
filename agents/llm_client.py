"""Shared Groq client, transient-error retry, and structured-output
parsing used by both the drafting and guardrail agents.

The SDK's own retry (`max_retries`, default backoff) is turned off on this
client — retry is owned here instead, so the 2s/4s/8s, max-3-attempt
backoff the spec asks for is exact and doesn't stack with another retry
layer underneath it.
"""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Callable, Type, TypeVar

import groq
from dotenv import load_dotenv
from pydantic import BaseModel, ValidationError

load_dotenv()

logger = logging.getLogger("agents.llm_client")

# openai/gpt-oss-20b: fast, cheap, and (with openai/gpt-oss-120b) one of only
# two Groq-hosted models that support strict json_schema mode -- the other
# models only offer best-effort JSON, which defeats the point of validating
# structured output at the API level rather than just hoping the prompt worked.
MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")
MAX_TRANSIENT_RETRIES = 3
BACKOFF_SECONDS = (2, 4, 8)

_client: groq.Groq | None = None


def get_client() -> groq.Groq:
    global _client
    if _client is None:
        _client = groq.Groq(max_retries=0)
    return _client


class LLMCallError(Exception):
    """Raised when an LLM call fails after exhausting transient retries, or
    keeps returning malformed/empty structured output after one retry.
    Never swallowed — callers see this, not a silent fallback.
    """


T = TypeVar("T", bound=BaseModel)


def _call_with_transient_retry(fn: Callable[[], "groq.types.chat.ChatCompletion"]):
    last_exc: Exception | None = None
    reason = "unknown"
    for attempt in range(MAX_TRANSIENT_RETRIES):
        try:
            return fn()
        except groq.RateLimitError as exc:
            last_exc, reason = exc, "rate_limited"
        except groq.APITimeoutError as exc:
            last_exc, reason = exc, "timeout"
        except groq.APIConnectionError as exc:
            last_exc, reason = exc, "connection_error"

        if attempt < MAX_TRANSIENT_RETRIES - 1:
            delay = BACKOFF_SECONDS[attempt]
            logger.warning("llm_call_retry attempt=%d reason=%s delay=%ds", attempt + 1, reason, delay)
            time.sleep(delay)

    logger.error("llm_call_exhausted_retries reason=%s", reason)
    raise LLMCallError(f"LLM call failed after {MAX_TRANSIENT_RETRIES} attempts: {last_exc!r}") from last_exc


def _strict_schema(model: Type[BaseModel]) -> dict:
    """Groq's strict json_schema mode requires every property to be listed
    in `required` (optional fields must be modeled as present-but-nullable,
    never absent) and `additionalProperties: false` at the top level.
    Pydantic's own model_json_schema() doesn't produce that shape by
    default — Optional fields are simply left out of `required`.
    """
    schema = model.model_json_schema()
    schema["additionalProperties"] = False
    schema["required"] = list(schema.get("properties", {}).keys())
    return schema


def _extract_parsed(response, output_format: Type[T]) -> T | None:
    if not response.choices:
        return None
    content = response.choices[0].message.content
    if not content:
        return None
    try:
        data = json.loads(content)
        return output_format.model_validate(data)
    except (json.JSONDecodeError, ValidationError):
        return None


def parse_structured(
    *,
    system: str,
    user_content: str,
    output_format: Type[T],
    max_tokens: int = 1024,
    context: str = "",
) -> T:
    """One structured-output LLM call.

    Transient errors (timeout / rate limit / connection) retry with
    2s/4s/8s backoff, up to 3 attempts. A malformed or empty result (bad
    JSON, or JSON that doesn't validate against `output_format` even after
    requesting strict mode) retries once more, then raises `LLMCallError`
    — never silently passed through.
    """
    client = get_client()
    schema = _strict_schema(output_format)

    def _call():
        return client.chat.completions.create(
            model=MODEL,
            max_completion_tokens=max_tokens,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user_content},
            ],
            response_format={
                "type": "json_schema",
                "json_schema": {"name": output_format.__name__, "strict": True, "schema": schema},
            },
        )

    response = _call_with_transient_retry(_call)
    parsed = _extract_parsed(response, output_format)
    if parsed is None:
        logger.warning("llm_malformed_output_retry context=%s", context)
        response = _call_with_transient_retry(_call)
        parsed = _extract_parsed(response, output_format)
        if parsed is None:
            raise LLMCallError(f"LLM returned malformed/empty structured output ({context})")
    return parsed

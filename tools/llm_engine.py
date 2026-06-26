"""Structured-output LLM istemcisi (OpenRouter/OpenAI uyumlu)."""

from __future__ import annotations

import os
from typing import Type, TypeVar

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel
from tenacity import (
    AsyncRetrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from core.config import load_oracle_config

SchemaT = TypeVar("SchemaT", bound=BaseModel)


class LlmEngine:
    """Pydantic şema zorlamalı LLM çağrıları için tek giriş noktası."""

    def __init__(self) -> None:
        self._client: ChatOpenAI | None = None
        self._max_retry_attempts: int = 4

    async def _ensure_client(self) -> ChatOpenAI:
        if self._client is not None:
            return self._client

        conf = await load_oracle_config()
        llm_conf = conf.llm

        base_url = os.getenv(llm_conf.base_url_env, "https://openrouter.ai/api/v1")
        api_key = os.getenv(llm_conf.api_key_env)
        if not api_key:
            raise RuntimeError(
                f"LLM API key bulunamadi. Beklenen env: {llm_conf.api_key_env}"
            )

        self._max_retry_attempts = llm_conf.max_retries + 1
        self._client = ChatOpenAI(
            model=llm_conf.model,
            base_url=base_url,
            api_key=api_key,
            temperature=llm_conf.temperature,
            timeout=llm_conf.timeout_seconds,
        )
        return self._client

    async def invoke_structured(
        self,
        *,
        schema: Type[SchemaT],
        system_prompt: str,
        user_prompt: str,
    ) -> SchemaT:
        client = await self._ensure_client()
        runner = client.with_structured_output(schema)
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ]

        async for attempt in AsyncRetrying(
            stop=stop_after_attempt(self._max_retry_attempts),
            wait=wait_exponential(multiplier=2, min=2, max=8),
            retry=retry_if_exception_type(Exception),
            reraise=True,
        ):
            with attempt:
                response = await runner.ainvoke(messages)
                if not isinstance(response, schema):
                    # Bazı sürümlerde dict dönebilir; Pydantic'e zorla.
                    return schema.model_validate(response)
                return response

        raise RuntimeError("LLM structured output üretimi başarısız oldu.")

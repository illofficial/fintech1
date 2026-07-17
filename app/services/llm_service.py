import logging
from collections.abc import AsyncIterator
from typing import cast

from openai import NOT_GIVEN, APIError, AsyncOpenAI, AsyncStream, NotGiven
from openai.types.chat import (
    ChatCompletion,
    ChatCompletionChunk,
    ChatCompletionMessageParam,
)

from app.models.llm import LLMRequest, LLMResponse
from app.services.retry import retry_on_rate_limit

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "gpt-4o-mini"
DEFAULT_TEMPERATURE = 0.7


class LLMService:
    """Thin, decoupled wrapper around the OpenAI chat completions API."""

    def __init__(self, client: AsyncOpenAI) -> None:
        self._client = client

    @staticmethod
    def _to_message_params(request: LLMRequest) -> list[ChatCompletionMessageParam]:
        return [
            cast(ChatCompletionMessageParam, {"role": message.role, "content": message.content})
            for message in request.messages
        ]

    @retry_on_rate_limit
    async def _create_completion(self, request: LLMRequest) -> ChatCompletion:
        max_tokens: int | NotGiven = NOT_GIVEN if request.max_tokens is None else request.max_tokens
        return await self._client.chat.completions.create(
            model=request.model,
            messages=self._to_message_params(request),
            temperature=request.temperature,
            max_tokens=max_tokens,
        )

    @retry_on_rate_limit
    async def _open_stream(
        self,
        *,
        model: str,
        messages: list[ChatCompletionMessageParam],
        temperature: float,
        max_tokens: int | NotGiven,
    ) -> AsyncStream[ChatCompletionChunk]:
        return await self._client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,
        )

    @staticmethod
    async def _iter_deltas(stream: AsyncStream[ChatCompletionChunk]) -> AsyncIterator[str]:
        async for chunk in stream:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta

    async def generate_response(self, request: LLMRequest) -> LLMResponse:
        try:
            response = await self._create_completion(request)
        except APIError:
            logger.exception("OpenAI API error while generating a response")
            raise

        choice = response.choices[0]
        return LLMResponse(
            content=choice.message.content or "",
            model=response.model,
            finish_reason=choice.finish_reason,
        )

    async def stream_response(self, request: LLMRequest) -> AsyncIterator[str]:
        """Stream token deltas for an :class:`LLMRequest` of typed chat messages."""
        max_tokens: int | NotGiven = NOT_GIVEN if request.max_tokens is None else request.max_tokens
        try:
            stream = await self._open_stream(
                model=request.model,
                messages=self._to_message_params(request),
                temperature=request.temperature,
                max_tokens=max_tokens,
            )
            async for delta in self._iter_deltas(stream):
                yield delta
        except APIError:
            logger.exception("OpenAI API error while streaming a response")
            raise

    async def stream_completion(
        self,
        messages: list[ChatCompletionMessageParam],
        *,
        model: str = DEFAULT_MODEL,
        temperature: float = DEFAULT_TEMPERATURE,
    ) -> AsyncIterator[str]:
        """Stream token deltas for a pre-built list of chat message params.

        Used by the agent flow to stream the final answer after the orchestrator
        has resolved any tool calls into a message context.
        """
        try:
            stream = await self._open_stream(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=NOT_GIVEN,
            )
            async for delta in self._iter_deltas(stream):
                yield delta
        except APIError:
            logger.exception("OpenAI API error while streaming completion")
            raise

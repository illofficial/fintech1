import os
from collections.abc import AsyncIterator, Iterator
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest
from openai import APIConnectionError

# The application settings require an API key at import/startup time. A dummy value is
# enough because every outbound OpenAI call is mocked in the test suite.
os.environ.setdefault("OPENAI_API_KEY", "test-key")


def make_api_error(message: str = "boom") -> APIConnectionError:
    """Build a concrete OpenAI ``APIError`` subclass usable in ``pytest.raises``."""
    return APIConnectionError(request=httpx.Request("POST", "https://api.openai.com/v1"))


def build_completion(
    *,
    content: str | None = None,
    tool_calls: list[SimpleNamespace] | None = None,
    model: str = "gpt-4o-mini",
    finish_reason: str = "stop",
) -> SimpleNamespace:
    """Build a stand-in for an OpenAI ``ChatCompletion`` response object."""
    message = SimpleNamespace(content=content, tool_calls=tool_calls)
    choice = SimpleNamespace(message=message, finish_reason=finish_reason)
    return SimpleNamespace(choices=[choice], model=model)


def build_tool_call(call_id: str, name: str, arguments: str) -> SimpleNamespace:
    """Build a stand-in for an OpenAI tool-call object."""
    return SimpleNamespace(id=call_id, function=SimpleNamespace(name=name, arguments=arguments))


class FakeStream:
    """Async-iterable stand-in for an OpenAI streaming response."""

    def __init__(self, deltas: list[str | None]) -> None:
        self._chunks = [
            SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content=delta))])
            for delta in deltas
        ]

    def __aiter__(self) -> AsyncIterator[SimpleNamespace]:
        return self._iterate()

    async def _iterate(self) -> AsyncIterator[SimpleNamespace]:
        for chunk in self._chunks:
            yield chunk


def build_openai_client(create_result: object) -> AsyncMock:
    """Build a mock ``AsyncOpenAI`` whose ``chat.completions.create`` is awaitable."""
    client = AsyncMock()
    client.chat.completions.create = AsyncMock(return_value=create_result)
    return client


@pytest.fixture
def fastapi_app() -> object:
    from app.main import create_app

    return create_app()


@pytest.fixture
def client(fastapi_app: object) -> Iterator[object]:
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    assert isinstance(fastapi_app, FastAPI)
    with TestClient(fastapi_app) as test_client:
        yield test_client

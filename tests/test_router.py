from collections.abc import AsyncIterator
from unittest.mock import AsyncMock

import pytest
from app.dependencies import get_agent_orchestrator, get_llm_service
from app.models.agent import MaxIterationsExceededError
from fastapi import FastAPI


def _make_agent(context: list[dict[str, str]] | None = None) -> AsyncMock:
    agent = AsyncMock()
    agent.model = "gpt-4o-mini"
    agent.build_context = AsyncMock(return_value=context or [{"role": "user", "content": "hi"}])
    return agent


def _make_llm(tokens: list[str]) -> AsyncMock:
    llm = AsyncMock()

    async def _stream(*_args: object, **_kwargs: object) -> AsyncIterator[str]:
        for token in tokens:
            yield token

    # stream_completion is a sync method returning an async generator, not a coroutine.
    llm.stream_completion = _stream
    return llm


def test_health(client: object) -> None:
    response = client.get("/health")  # type: ignore[attr-defined]
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_chat_streams_tokens(fastapi_app: object, client: object) -> None:
    assert isinstance(fastapi_app, FastAPI)
    agent = _make_agent()
    llm = _make_llm(["Hello", ", ", "world"])
    fastapi_app.dependency_overrides[get_agent_orchestrator] = lambda: agent
    fastapi_app.dependency_overrides[get_llm_service] = lambda: llm

    response = client.post("/v1/chat", json={"message": "hi"})  # type: ignore[attr-defined]

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert response.text == "Hello, world"
    agent.build_context.assert_awaited_once_with("hi")


def test_chat_rejects_empty_message(fastapi_app: object, client: object) -> None:
    assert isinstance(fastapi_app, FastAPI)
    fastapi_app.dependency_overrides[get_agent_orchestrator] = lambda: _make_agent()
    fastapi_app.dependency_overrides[get_llm_service] = lambda: _make_llm(["x"])

    response = client.post("/v1/chat", json={"message": ""})  # type: ignore[attr-defined]
    assert response.status_code == 422


def test_chat_missing_field(client: object) -> None:
    response = client.post("/v1/chat", json={})  # type: ignore[attr-defined]
    assert response.status_code == 422


def test_chat_iteration_limit_returns_504(fastapi_app: object, client: object) -> None:
    assert isinstance(fastapi_app, FastAPI)
    agent = AsyncMock()
    agent.model = "gpt-4o-mini"
    agent.build_context = AsyncMock(side_effect=MaxIterationsExceededError(5))
    fastapi_app.dependency_overrides[get_agent_orchestrator] = lambda: agent
    fastapi_app.dependency_overrides[get_llm_service] = lambda: _make_llm(["x"])

    response = client.post("/v1/chat", json={"message": "loop"})  # type: ignore[attr-defined]
    assert response.status_code == 504


def test_chat_upstream_error_returns_502(fastapi_app: object, client: object) -> None:
    from tests.conftest import make_api_error

    assert isinstance(fastapi_app, FastAPI)
    agent = AsyncMock()
    agent.model = "gpt-4o-mini"
    agent.build_context = AsyncMock(side_effect=make_api_error())
    fastapi_app.dependency_overrides[get_agent_orchestrator] = lambda: agent
    fastapi_app.dependency_overrides[get_llm_service] = lambda: _make_llm(["x"])

    response = client.post("/v1/chat", json={"message": "hi"})  # type: ignore[attr-defined]
    assert response.status_code == 502


@pytest.fixture(autouse=True)
def _clear_overrides(fastapi_app: object) -> object:
    assert isinstance(fastapi_app, FastAPI)
    yield
    fastapi_app.dependency_overrides.clear()

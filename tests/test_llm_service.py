import pytest
from app.models.llm import ChatMessage, LLMRequest
from app.services.llm_service import LLMService
from openai import APIError
from tests.conftest import FakeStream, build_completion, build_openai_client, make_api_error


def _request() -> LLMRequest:
    return LLMRequest(messages=[ChatMessage(role="user", content="hello")])


async def test_generate_response_returns_content() -> None:
    client = build_openai_client(build_completion(content="hi there", model="gpt-4o-mini"))
    service = LLMService(client)

    result = await service.generate_response(_request())

    assert result.content == "hi there"
    assert result.model == "gpt-4o-mini"
    assert result.finish_reason == "stop"
    client.chat.completions.create.assert_awaited_once()


async def test_generate_response_handles_none_content() -> None:
    client = build_openai_client(build_completion(content=None))
    service = LLMService(client)

    result = await service.generate_response(_request())
    assert result.content == ""


async def test_generate_response_raises_api_error() -> None:
    client = build_openai_client(build_completion(content="unused"))
    client.chat.completions.create.side_effect = make_api_error()
    service = LLMService(client)

    with pytest.raises(APIError):
        await service.generate_response(_request())


async def test_stream_response_yields_non_empty_deltas() -> None:
    client = build_openai_client(FakeStream(["He", "llo", None, "!"]))
    service = LLMService(client)

    tokens = [token async for token in service.stream_response(_request())]
    assert tokens == ["He", "llo", "!"]


async def test_stream_completion_yields_tokens() -> None:
    client = build_openai_client(FakeStream(["a", "b"]))
    service = LLMService(client)

    tokens = [
        token
        async for token in service.stream_completion(
            [{"role": "user", "content": "hi"}], model="gpt-4o-mini"
        )
    ]
    assert tokens == ["a", "b"]


async def test_stream_response_raises_api_error() -> None:
    client = build_openai_client(FakeStream([]))
    client.chat.completions.create.side_effect = make_api_error()
    service = LLMService(client)

    with pytest.raises(APIError):
        async for _ in service.stream_response(_request()):
            pass

import json
from datetime import date

import pytest
from app.models.agent import (
    AgentResponse,
    FintechTransactionQuery,
    MaxIterationsExceededError,
)
from app.services.agent_core import AgentOrchestrator, default_fintech_db_lookup
from openai import APIError
from tests.conftest import build_completion, build_openai_client, build_tool_call, make_api_error

VALID_ARGS = json.dumps(
    {
        "account_id": "ACC1",
        "start_date": "2026-07-01",
        "end_date": "2026-07-31",
        "category": "all",
        "limit": 10,
    }
)


async def test_default_lookup_filters_by_date_and_category() -> None:
    query = FintechTransactionQuery(
        account_id="ACC9",
        start_date=date(2026, 7, 1),
        end_date=date(2026, 7, 6),
        category="transfer",
    )
    records = await default_fintech_db_lookup(query)
    assert [record.id for record in records] == ["TXN-1002"]
    assert records[0].account_id == "ACC9"


async def test_default_lookup_respects_limit() -> None:
    query = FintechTransactionQuery(
        account_id="ACC1",
        start_date=date(2026, 7, 1),
        end_date=date(2026, 7, 31),
        limit=1,
    )
    records = await default_fintech_db_lookup(query)
    assert len(records) == 1


async def test_execute_tool_returns_transactions() -> None:
    orchestrator = AgentOrchestrator(build_openai_client(build_completion()))
    raw = await orchestrator._execute_tool("get_fintech_transactions", VALID_ARGS)
    payload = json.loads(raw)
    assert "transactions" in payload
    assert payload["transactions"][0]["id"] == "TXN-1001"


async def test_execute_tool_unknown_tool() -> None:
    orchestrator = AgentOrchestrator(build_openai_client(build_completion()))
    raw = await orchestrator._execute_tool("nope", VALID_ARGS)
    assert json.loads(raw)["error"].startswith("Unknown tool")


async def test_execute_tool_invalid_arguments() -> None:
    orchestrator = AgentOrchestrator(build_openai_client(build_completion()))
    raw = await orchestrator._execute_tool("get_fintech_transactions", '{"account_id": "BAD"}')
    assert json.loads(raw)["error"] == "Invalid tool arguments"


async def test_build_context_direct_answer_has_no_tool_turns() -> None:
    client = build_openai_client(build_completion(content="Here is your summary"))
    orchestrator = AgentOrchestrator(client)

    context = await orchestrator.build_context("what is my balance?")

    assert [message["role"] for message in context] == ["system", "user"]
    assert context[1]["content"] == "what is my balance?"


async def test_build_context_runs_tool_then_returns_context() -> None:
    tool_call = build_tool_call("call-1", "get_fintech_transactions", VALID_ARGS)
    client = build_openai_client(build_completion())
    client.chat.completions.create.side_effect = [
        build_completion(content=None, tool_calls=[tool_call]),
        build_completion(content="done"),
    ]
    orchestrator = AgentOrchestrator(client)

    context = await orchestrator.build_context("show transfers")

    roles = [message["role"] for message in context]
    assert roles == ["system", "user", "assistant", "tool"]
    assert client.chat.completions.create.await_count == 2


async def test_build_context_enforces_iteration_limit() -> None:
    tool_call = build_tool_call("call-1", "get_fintech_transactions", VALID_ARGS)
    client = build_openai_client(build_completion(content=None, tool_calls=[tool_call]))
    orchestrator = AgentOrchestrator(client, max_iterations=2)

    with pytest.raises(MaxIterationsExceededError) as exc_info:
        await orchestrator.build_context("loop forever")

    assert exc_info.value.max_iterations == 2
    assert client.chat.completions.create.await_count == 2


async def test_build_context_propagates_api_error() -> None:
    client = build_openai_client(build_completion())
    client.chat.completions.create.side_effect = make_api_error()
    orchestrator = AgentOrchestrator(client)

    with pytest.raises(APIError):
        await orchestrator.build_context("hi")


async def test_run_returns_agent_response() -> None:
    client = build_openai_client(build_completion(content="final answer"))
    orchestrator = AgentOrchestrator(client)

    response = await orchestrator.run("summarize my spending")

    assert isinstance(response, AgentResponse)
    assert response.content == "final answer"
    assert response.iterations == 0

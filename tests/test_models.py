from datetime import date

import pytest
from app.models.agent import (
    AgentRequest,
    FintechTransactionQuery,
    TransactionRecord,
    UserRequest,
)
from app.models.llm import ChatMessage, LLMRequest
from app.models.rag import RAGConfig, RetrieveContextRequest, ScoredDocument
from pydantic import ValidationError


def test_user_request_requires_non_empty_message() -> None:
    assert UserRequest(message="hello").message == "hello"
    with pytest.raises(ValidationError):
        UserRequest(message="")


def test_user_request_rejects_oversized_message() -> None:
    with pytest.raises(ValidationError):
        UserRequest(message="x" * 8_001)


def test_fintech_query_account_pattern() -> None:
    query = FintechTransactionQuery(
        account_id="ACC123",
        start_date=date(2026, 7, 1),
        end_date=date(2026, 7, 31),
    )
    assert query.category == "all"
    assert query.limit == 10

    with pytest.raises(ValidationError):
        FintechTransactionQuery(
            account_id="BAD1",
            start_date=date(2026, 7, 1),
            end_date=date(2026, 7, 31),
        )


def test_fintech_query_rejects_reversed_date_range() -> None:
    with pytest.raises(ValidationError, match="on or after"):
        FintechTransactionQuery(
            account_id="ACC1",
            start_date=date(2026, 7, 31),
            end_date=date(2026, 7, 1),
        )


def test_fintech_query_limit_bounds() -> None:
    with pytest.raises(ValidationError):
        FintechTransactionQuery(
            account_id="ACC1",
            start_date=date(2026, 7, 1),
            end_date=date(2026, 7, 2),
            limit=0,
        )


def test_agent_request_defaults() -> None:
    request = AgentRequest(query="show my transfers")
    assert request.model == "gpt-4o-mini"
    assert request.system_prompt


def test_transaction_record_json_roundtrip() -> None:
    record = TransactionRecord(
        id="TXN-1",
        account_id="ACC1",
        date=date(2026, 7, 1),
        amount=-10.5,
        currency="USD",
        category="payment",
        merchant="Shop",
    )
    dumped = record.model_dump(mode="json")
    assert dumped["date"] == "2026-07-01"
    assert TransactionRecord.model_validate(dumped) == record


def test_chat_message_role_validation() -> None:
    assert ChatMessage(role="user", content="hi").role == "user"
    with pytest.raises(ValidationError):
        ChatMessage(role="robot", content="hi")


def test_llm_request_requires_messages() -> None:
    with pytest.raises(ValidationError):
        LLMRequest(messages=[])

    request = LLMRequest(messages=[ChatMessage(role="user", content="hi")])
    assert request.temperature == 0.7


def test_llm_request_temperature_bounds() -> None:
    with pytest.raises(ValidationError):
        LLMRequest(messages=[ChatMessage(role="user", content="hi")], temperature=5.0)


def test_scored_document_source_default() -> None:
    doc = ScoredDocument(id="d1", content="c", score=0.5)
    assert doc.source == "hybrid"


def test_rag_config_and_retrieve_defaults() -> None:
    assert RAGConfig().vector_size == 1536
    assert RetrieveContextRequest(query="q").limit == 3
    with pytest.raises(ValidationError):
        RetrieveContextRequest(query="q", limit=0)

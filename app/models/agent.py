from datetime import date
from typing import Literal, Self

from pydantic import BaseModel, Field, model_validator


class FintechTransactionQuery(BaseModel):
    """Validated arguments for the fintech database lookup tool."""

    account_id: str = Field(..., min_length=1, pattern=r"^ACC\d+$")
    start_date: date
    end_date: date
    category: Literal["all", "transfer", "payment", "withdrawal"] = "all"
    limit: int = Field(default=10, ge=1, le=100)

    @model_validator(mode="after")
    def validate_date_range(self) -> Self:
        if self.end_date < self.start_date:
            raise ValueError("end_date must be on or after start_date")
        return self


class TransactionRecord(BaseModel):
    """A single transaction row returned by the fintech lookup tool."""

    id: str
    account_id: str
    date: date
    amount: float
    currency: str
    category: str
    merchant: str


class UserRequest(BaseModel):
    """Inbound payload for the public ``/chat`` endpoint."""

    message: str = Field(..., min_length=1, max_length=8_000)


class AgentRequest(BaseModel):
    query: str = Field(..., min_length=1)
    model: str = Field(default="gpt-4o-mini", min_length=1)
    system_prompt: str = Field(
        default=(
            "You are a fintech assistant. Use available tools to fetch "
            "transaction data when needed, then summarize findings clearly."
        ),
        min_length=1,
    )


class AgentResponse(BaseModel):
    content: str
    model: str
    iterations: int
    finish_reason: str | None = None


class MaxIterationsExceededError(Exception):
    """Raised when the agent tool loop exceeds the allowed iteration limit."""

    def __init__(self, max_iterations: int) -> None:
        self.max_iterations = max_iterations
        super().__init__(f"Agent exceeded maximum iterations ({max_iterations})")

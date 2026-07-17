from typing import Literal

from pydantic import BaseModel, Field, field_validator


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str = Field(..., min_length=1)


class LLMRequest(BaseModel):
    messages: list[ChatMessage] = Field(..., min_length=1)
    model: str = Field(default="gpt-4o-mini", min_length=1)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: int | None = Field(default=None, gt=0)

    @field_validator("messages")
    @classmethod
    def validate_messages_not_empty(cls, messages: list[ChatMessage]) -> list[ChatMessage]:
        if not messages:
            raise ValueError("messages must contain at least one message")
        return messages


class LLMResponse(BaseModel):
    content: str
    model: str
    finish_reason: str | None = None

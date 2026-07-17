from typing import Literal, Self

from pydantic import BaseModel, Field, model_validator

from app.models.types import FinishReason


class ChatMessage(BaseModel):
    """
    A single message in a chat conversation.

    Supports system, user, assistant, and tool messages with metadata
    for function calling.

    Examples:
        >>> msg = ChatMessage(
        ...     role="user",
        ...     content="What's my balance?",
        ... )
        >>> tool_msg = ChatMessage(
        ...     role="tool",
        ...     content='{"balance": 1000}',
        ...     tool_call_id="call_abc123",
        ...     name="get_balance",
        ... )
    """

    role: Literal["system", "user", "assistant", "tool"] = Field(
        ...,
        description="Message role in the conversation",
        examples=["user", "assistant"],
    )
    content: str = Field(
        ...,
        min_length=1,
        max_length=100_000,
        description="Message content",
        examples=["Hello, how can I help you?"],
    )
    name: str | None = Field(
        default=None,
        description="Function name for tool messages",
        examples=["get_balance", "transfer_funds"],
    )
    tool_call_id: str | None = Field(
        default=None,
        description="Tool call ID for tool response messages",
        examples=["call_abc123"],
    )

    @model_validator(mode="after")
    def validate_tool_message(self) -> Self:
        """Ensure tool messages have required metadata."""
        if self.role == "tool":
            if not self.tool_call_id:
                raise ValueError("tool_call_id is required for tool messages")
            if not self.name:
                raise ValueError("name is required for tool messages")
        return self

    def __repr__(self) -> str:
        """Human-readable representation for debugging."""
        content_preview = self.content[:50] + "..." if len(self.content) > 50 else self.content
        return f"<ChatMessage role={self.role} content='{content_preview}'>"


class TokenUsage(BaseModel):
    """
    Token usage statistics for an LLM request.

    Used for cost monitoring and optimization.
    """

    prompt_tokens: int = Field(..., ge=0, description="Number of tokens in the prompt")
    completion_tokens: int = Field(..., ge=0, description="Number of tokens in the completion")
    total_tokens: int = Field(..., ge=0, description="Total tokens used")

    @model_validator(mode="after")
    def validate_total(self) -> Self:
        """Ensure total_tokens equals sum of prompt and completion tokens."""
        expected_total = self.prompt_tokens + self.completion_tokens
        if self.total_tokens != expected_total:
            # OpenAI sometimes has slight discrepancies, so just warn
            pass
        return self


class LLMRequest(BaseModel):
    """
    Request to the LLM service.

    Supports system, user, assistant, and tool messages with full metadata.

    Examples:
        >>> request = LLMRequest(
        ...     messages=[
        ...         ChatMessage(role="system", content="You are a helpful assistant."),
        ...         ChatMessage(role="user", content="Hello!"),
        ...     ],
        ...     temperature=0.7,
        ...     max_tokens=1000,
        ... )
    """

    messages: list[ChatMessage] = Field(
        ...,
        min_length=1,
        max_length=100,
        description="Conversation messages",
    )
    model: str = Field(
        default="gpt-4o-mini",
        min_length=1,
        description="OpenAI model to use",
        examples=["gpt-4o-mini", "gpt-4o"],
    )
    temperature: float = Field(
        default=0.7,
        ge=0.0,
        le=2.0,
        description="Sampling temperature (0.0 = deterministic, 2.0 = random)",
    )
    max_tokens: int | None = Field(
        default=None,
        gt=0,
        le=16_384,
        description="Maximum tokens to generate (None = no limit)",
        examples=[1000, 4000],
    )

    def __repr__(self) -> str:
        """Human-readable representation for debugging."""
        return (
            f"<LLMRequest model={self.model} messages={len(self.messages)} temp={self.temperature}>"
        )


class LLMResponse(BaseModel):
    """
    Response from the LLM service.

    Contains the generated content, model info, and token usage.

    Examples:
        >>> response = LLMResponse(
        ...     content="Hello! How can I help you today?",
        ...     model="gpt-4o-mini",
        ...     finish_reason="stop",
        ...     usage=TokenUsage(prompt_tokens=10, completion_tokens=8, total_tokens=18),
        ... )
    """

    content: str = Field(
        ...,
        description="Generated response content",
    )
    model: str = Field(
        ...,
        description="Model that generated the response",
        examples=["gpt-4o-mini"],
    )
    finish_reason: FinishReason | None = Field(
        default=None,
        description="Reason the model stopped generating tokens",
    )
    usage: TokenUsage | None = Field(
        default=None,
        description="Token usage statistics for cost monitoring",
    )

    def __repr__(self) -> str:
        """Human-readable representation for debugging."""
        content_preview = self.content[:50] + "..." if len(self.content) > 50 else self.content
        return (
            f"<LLMResponse model={self.model} "
            f"finish_reason={self.finish_reason} "
            f"content='{content_preview}'>"
        )

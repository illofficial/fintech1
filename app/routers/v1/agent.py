import logging
import time
import uuid
from collections.abc import AsyncIterator
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import StreamingResponse
from openai import APIError, BadRequestError

from app.dependencies import get_agent_orchestrator, get_llm_service
from app.models.agent import MaxIterationsExceededError, UserRequest
from app.services.agent_core import AgentOrchestrator
from app.services.llm_service import LLMService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1", tags=["agent"])

AgentDep = Annotated[AgentOrchestrator, Depends(get_agent_orchestrator)]
LLMServiceDep = Annotated[LLMService, Depends(get_llm_service)]


@router.post("/chat")
async def chat(
    request: Request,
    payload: UserRequest,
    agent: AgentDep,
    llm_service: LLMServiceDep,
) -> StreamingResponse:
    """Resolve the request through the agent's tool loop and stream the answer back.

    The agent performs any necessary tool calls and returns a message context,
    then the LLM service streams the final natural-language answer to the client.

    ## Error Responses:
    - 400: Context window exceeded (conversation history too large)
    - 502: Upstream gateway error (OpenAI infrastructure down)
    - 504: Agent step budget exhausted (infinite tool-calling loop protection)

    ## Example:
        ```bash
        curl -N -X POST http://localhost:8000/v1/chat \\
          -H 'Content-Type: application/json' \\
          -d '{"message": "Summarize my transfers on account ACC123 in July 2026."}'
        ```
    """
    request_id = str(uuid.uuid4())
    start_time = time.monotonic()

    logger.info(
        "Initiating chat resolution pipeline",
        extra={
            "request_id": request_id,
            "message_preview": payload.message[:100],
        },
    )

    try:
        # Step 1: Resolve dependencies and build contextual historical graph
        context = await agent.build_context(payload.message)
    except MaxIterationsExceededError as exc:
        logger.error(
            "Agent execution step budget exhausted",
            extra={"request_id": request_id},
        )
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail=(
                "The agent failed to resolve the request within its allowed "
                "reasoning iteration budget."
            ),
        ) from exc
    except BadRequestError as exc:
        if "context_length_exceeded" in str(exc):
            logger.warning(
                "Context bounds constraint violated during context construction",
                extra={"request_id": request_id},
            )
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=(
                    "The conversation context exceeds limits. "
                    "Please clear state or shorten your input."
                ),
            ) from exc
        raise
    except APIError as exc:
        logger.exception(
            "Fatal upstream OpenAI infrastructure failure during context build",
            extra={"request_id": request_id},
        )
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Upstream language model gateway is temporarily unavailable.",
        ) from exc

    # Step 2: Safe generator wrapper to capture real execution metrics and mid-stream disconnects
    async def stream_wrapper() -> AsyncIterator[str]:
        try:
            async for chunk in llm_service.stream_completion(context, model=agent.model):
                # Ensure the client is still connected before yielding downstream token buffers
                if await request.is_disconnected():
                    logger.warning(
                        "Client disconnected early during token stream delivery loop",
                        extra={"request_id": request_id},
                    )
                    break
                yield chunk
        except APIError:
            logger.exception(
                "Infrastructure failure mid-stream during token generation",
                extra={"request_id": request_id},
            )
            raise
        finally:
            duration = time.monotonic() - start_time
            logger.info(
                "Streaming response context closed and finalized",
                extra={
                    "request_id": request_id,
                    "total_pipeline_duration_seconds": round(duration, 3),
                },
            )

    return StreamingResponse(
        stream_wrapper(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "Connection": "keep-alive",
            "X-Request-ID": request_id,
        },
    )

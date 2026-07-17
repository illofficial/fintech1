import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from openai import AsyncOpenAI

from app.config import get_settings
from app.models.rag import RAGConfig
from app.routers.v1 import agent as agent_router
from app.services.agent_core import AgentOrchestrator
from app.services.llm_service import LLMService
from app.services.vector_db import RAGService

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Build shared, decoupled service singletons and store them on ``app.state``."""
    settings = get_settings()
    client = AsyncOpenAI(
        api_key=settings.openai_api_key,
        timeout=settings.request_timeout_seconds,
    )
    rag_config = RAGConfig(
        collection_name=settings.qdrant_collection,
        embedding_model=settings.embedding_model,
        qdrant_url=settings.qdrant_url,
        qdrant_api_key=settings.qdrant_api_key,
    )

    app.state.settings = settings
    app.state.openai_client = client
    app.state.llm_service = LLMService(client)
    app.state.agent_orchestrator = AgentOrchestrator(
        client,
        model=settings.openai_model,
        max_iterations=settings.agent_max_iterations,
    )
    app.state.rag_service = RAGService(client=client, config=rag_config)

    logger.info("Application services initialized")
    try:
        yield
    finally:
        await app.state.rag_service.close()
        await client.close()
        logger.info("Application services shut down")


def create_app() -> FastAPI:
    app = FastAPI(title="Fintech Agent API", version="1.0.0", lifespan=lifespan)
    app.include_router(agent_router.router)

    @app.get("/health", tags=["health"])
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    return app


app = create_app()

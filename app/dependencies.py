from fastapi import Request

from app.config import Settings
from app.services.agent_core import AgentOrchestrator
from app.services.llm_service import LLMService
from app.services.vector_db import RAGService


def get_settings(request: Request) -> Settings:
    """Return the application settings created during startup."""
    return request.app.state.settings


def get_llm_service(request: Request) -> LLMService:
    """Return the process-wide :class:`LLMService` built during startup."""
    return request.app.state.llm_service


def get_agent_orchestrator(request: Request) -> AgentOrchestrator:
    """Return the process-wide :class:`AgentOrchestrator` built during startup."""
    return request.app.state.agent_orchestrator


def get_rag_service(request: Request) -> RAGService:
    """Return the process-wide :class:`RAGService` built during startup."""
    return request.app.state.rag_service

from typing import Literal

from pydantic import BaseModel, Field


class ScoredDocument(BaseModel):
    id: str
    content: str
    score: float
    source: Literal["vector", "keyword", "hybrid"] = "hybrid"


class RAGConfig(BaseModel):
    collection_name: str = Field(default="documents", min_length=1)
    embedding_model: str = Field(default="text-embedding-3-small", min_length=1)
    vector_size: int = Field(default=1536, gt=0)
    qdrant_url: str | None = None
    qdrant_api_key: str | None = None
    use_mock: bool | None = Field(
        default=None,
        description="Force mock store when True. None auto-detects from Qdrant availability.",
    )
    hybrid_vector_weight: float = Field(default=0.7, ge=0.0, le=1.0)
    rrf_k: int = Field(default=60, gt=0)


class RetrieveContextRequest(BaseModel):
    query: str = Field(..., min_length=1)
    limit: int = Field(default=3, ge=1, le=50)

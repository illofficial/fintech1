import asyncio
import logging
import math
import os
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import httpx
from openai import APIError, AsyncOpenAI
from qdrant_client import AsyncQdrantClient
from qdrant_client.http.exceptions import ResponseHandlingException, UnexpectedResponse

from app.models.rag import RAGConfig, RetrieveContextRequest, ScoredDocument
from app.services.retry import retry_on_rate_limit

logger = logging.getLogger(__name__)

# Errors that indicate Qdrant is unreachable or misconfigured; any of these triggers a
# graceful fallback to the in-memory store rather than crashing the service.
QDRANT_CONNECTION_ERRORS = (
    ResponseHandlingException,
    UnexpectedResponse,
    httpx.HTTPError,
    OSError,
)

_TOKEN_PATTERN = re.compile(r"[a-z0-9]+")

_MOCK_SEED_DOCUMENTS: list[dict[str, str]] = [
    {
        "id": "doc-001",
        "content": (
            "Wire transfer limits for retail accounts: daily outbound limit is "
            "USD 25,000 and monthly limit is USD 100,000."
        ),
    },
    {
        "id": "doc-002",
        "content": (
            "KYC refresh policy requires identity verification every 24 months "
            "for high-risk customer segments."
        ),
    },
    {
        "id": "doc-003",
        "content": (
            "Chargeback handling SLA: merchant disputes must be acknowledged "
            "within 2 business days and resolved within 15 business days."
        ),
    },
    {
        "id": "doc-004",
        "content": (
            "AML monitoring rules flag transactions above USD 10,000 and "
            "unusual cross-border payment patterns."
        ),
    },
    {
        "id": "doc-005",
        "content": (
            "API rate limits for fintech partners: 600 requests per minute "
            "with burst capacity up to 1200 requests per minute."
        ),
    },
]


def _tokenize(text: str) -> set[str]:
    return set(_TOKEN_PATTERN.findall(text.lower()))


def _keyword_score(query: str, content: str) -> float:
    query_tokens = _tokenize(query)
    if not query_tokens:
        return 0.0

    content_tokens = _tokenize(content)
    if not content_tokens:
        return 0.0

    overlap = query_tokens & content_tokens
    if not overlap:
        return 0.0

    # Lightweight BM25-inspired score without corpus statistics.
    return len(overlap) / math.sqrt(len(content_tokens))


def _cosine_similarity(left: list[float], right: list[float]) -> float:
    if len(left) != len(right) or not left:
        return 0.0

    dot = sum(a * b for a, b in zip(left, right, strict=True))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return dot / (left_norm * right_norm)


@dataclass(slots=True)
class StoredDocument:
    """A seed document plus its cached embedding held by the in-memory store."""

    id: str
    content: str
    embedding: list[float] = field(default_factory=list)


class VectorStoreBackend(ABC):
    @abstractmethod
    async def vector_search(self, vector: list[float], limit: int) -> list[ScoredDocument]:
        raise NotImplementedError

    @abstractmethod
    async def keyword_search(self, query: str, limit: int) -> list[ScoredDocument]:
        raise NotImplementedError

    @abstractmethod
    async def close(self) -> None:
        raise NotImplementedError


class EmbeddingProvider:
    def __init__(
        self,
        client: AsyncOpenAI,
        *,
        model: str,
    ) -> None:
        self._client = client
        self._model = model

    @retry_on_rate_limit
    async def embed(self, text: str) -> list[float]:
        response = await self._client.embeddings.create(
            model=self._model,
            input=text,
        )
        return response.data[0].embedding


class MockVectorStore(VectorStoreBackend):
    """In-memory vector store used when Qdrant is unavailable."""

    def __init__(
        self,
        *,
        embedder: EmbeddingProvider,
        seed_documents: list[dict[str, str]] | None = None,
    ) -> None:
        self._embedder = embedder
        self._documents: list[StoredDocument] = []
        self._seed_documents = seed_documents or _MOCK_SEED_DOCUMENTS
        self._initialized = False
        self._init_lock = asyncio.Lock()

    async def _ensure_initialized(self) -> None:
        if self._initialized:
            return

        async with self._init_lock:
            if self._initialized:
                return

            for document in self._seed_documents:
                embedding = await self._embedder.embed(document["content"])
                self._documents.append(
                    StoredDocument(
                        id=document["id"],
                        content=document["content"],
                        embedding=embedding,
                    )
                )

            self._initialized = True
            logger.info("MockVectorStore initialized with %s documents", len(self._documents))

    async def vector_search(self, vector: list[float], limit: int) -> list[ScoredDocument]:
        await self._ensure_initialized()

        ranked = sorted(
            self._documents,
            key=lambda item: _cosine_similarity(vector, item.embedding),
            reverse=True,
        )

        results: list[ScoredDocument] = []
        for item in ranked[:limit]:
            score = _cosine_similarity(vector, item.embedding)
            if score <= 0.0:
                continue
            results.append(
                ScoredDocument(
                    id=item.id,
                    content=item.content,
                    score=score,
                    source="vector",
                )
            )
        return results

    async def keyword_search(self, query: str, limit: int) -> list[ScoredDocument]:
        await self._ensure_initialized()

        ranked = sorted(
            self._documents,
            key=lambda item: _keyword_score(query, item.content),
            reverse=True,
        )

        results: list[ScoredDocument] = []
        for item in ranked[:limit]:
            score = _keyword_score(query, item.content)
            if score <= 0.0:
                continue
            results.append(
                ScoredDocument(
                    id=item.id,
                    content=item.content,
                    score=score,
                    source="keyword",
                )
            )
        return results

    async def close(self) -> None:
        return None


class QdrantVectorStore(VectorStoreBackend):
    """Async Qdrant integration for vector and keyword retrieval."""

    def __init__(
        self,
        client: AsyncQdrantClient,
        *,
        collection_name: str,
        content_field: str = "content",
    ) -> None:
        self._client = client
        self._collection_name = collection_name
        self._content_field = content_field

    async def vector_search(self, vector: list[float], limit: int) -> list[ScoredDocument]:
        response = await self._client.query_points(
            collection_name=self._collection_name,
            query=vector,
            limit=limit,
            with_payload=True,
        )

        points = response.points if hasattr(response, "points") else response
        results: list[ScoredDocument] = []
        for point in points:
            payload = point.payload or {}
            content = payload.get(self._content_field)
            if not isinstance(content, str):
                continue
            results.append(
                ScoredDocument(
                    id=str(point.id),
                    content=content,
                    score=float(point.score or 0.0),
                    source="vector",
                )
            )
        return results

    async def keyword_search(self, query: str, limit: int) -> list[ScoredDocument]:
        from qdrant_client.models import FieldCondition, Filter, MatchText

        records, _ = await self._client.scroll(
            collection_name=self._collection_name,
            scroll_filter=Filter(
                must=[
                    FieldCondition(
                        key=self._content_field,
                        match=MatchText(text=query),
                    )
                ]
            ),
            limit=limit,
            with_payload=True,
        )

        results: list[ScoredDocument] = []
        for point in records:
            payload = point.payload or {}
            content = payload.get(self._content_field)
            if not isinstance(content, str):
                continue
            results.append(
                ScoredDocument(
                    id=str(point.id),
                    content=content,
                    score=_keyword_score(query, content),
                    source="keyword",
                )
            )

        results.sort(key=lambda item: item.score, reverse=True)
        return results[:limit]

    async def close(self) -> None:
        await self._client.close()


async def create_vector_store(
    config: RAGConfig,
    embedder: EmbeddingProvider,
) -> VectorStoreBackend:
    if config.use_mock is True:
        logger.info("RAGService configured to use MockVectorStore")
        return MockVectorStore(embedder=embedder)

    qdrant_url = config.qdrant_url or os.getenv("QDRANT_URL")
    qdrant_api_key = config.qdrant_api_key or os.getenv("QDRANT_API_KEY")

    if not qdrant_url:
        logger.warning("QDRANT_URL is not set; using MockVectorStore")
        return MockVectorStore(embedder=embedder)

    try:
        client = AsyncQdrantClient(url=qdrant_url, api_key=qdrant_api_key)
        await client.get_collections()
    except QDRANT_CONNECTION_ERRORS:
        logger.exception("Failed to connect to Qdrant; falling back to MockVectorStore")
        return MockVectorStore(embedder=embedder)

    logger.info("Connected to Qdrant at %s", qdrant_url)
    return QdrantVectorStore(client, collection_name=config.collection_name)


class RAGService:
    def __init__(
        self,
        *,
        api_key: str | None = None,
        client: AsyncOpenAI | None = None,
        config: RAGConfig | None = None,
        vector_store: VectorStoreBackend | None = None,
    ) -> None:
        self._config = config or RAGConfig()
        self._client = client or AsyncOpenAI(api_key=api_key)
        self._embedder = EmbeddingProvider(self._client, model=self._config.embedding_model)
        self._vector_store = vector_store
        self._store_lock = asyncio.Lock()

    async def _get_store(self) -> VectorStoreBackend:
        if self._vector_store is not None:
            return self._vector_store

        async with self._store_lock:
            if self._vector_store is None:
                self._vector_store = await create_vector_store(self._config, self._embedder)
            return self._vector_store

    @staticmethod
    def _reciprocal_rank_fusion(
        vector_results: list[ScoredDocument],
        keyword_results: list[ScoredDocument],
        *,
        limit: int,
        rrf_k: int,
        vector_weight: float,
    ) -> list[ScoredDocument]:
        keyword_weight = 1.0 - vector_weight
        fused_scores: dict[str, float] = {}
        documents: dict[str, ScoredDocument] = {}

        for rank, document in enumerate(vector_results, start=1):
            fused_scores[document.id] = fused_scores.get(document.id, 0.0) + (
                vector_weight / (rrf_k + rank)
            )
            documents[document.id] = document

        for rank, document in enumerate(keyword_results, start=1):
            fused_scores[document.id] = fused_scores.get(document.id, 0.0) + (
                keyword_weight / (rrf_k + rank)
            )
            documents.setdefault(document.id, document)

        ranked_ids = sorted(
            fused_scores,
            key=lambda document_id: fused_scores[document_id],
            reverse=True,
        )

        hybrid_results: list[ScoredDocument] = []
        for document_id in ranked_ids[:limit]:
            base = documents[document_id]
            hybrid_results.append(
                ScoredDocument(
                    id=base.id,
                    content=base.content,
                    score=fused_scores[document_id],
                    source="hybrid",
                )
            )
        return hybrid_results

    async def hybrid_search(self, query: str, limit: int = 10) -> list[ScoredDocument]:
        validated = RetrieveContextRequest.model_validate({"query": query, "limit": limit})
        store = await self._get_store()

        try:
            query_vector = await self._embedder.embed(validated.query)
            candidate_limit = max(validated.limit * 2, validated.limit)
            vector_results, keyword_results = await asyncio.gather(
                store.vector_search(query_vector, candidate_limit),
                store.keyword_search(validated.query, candidate_limit),
            )

            return self._reciprocal_rank_fusion(
                vector_results,
                keyword_results,
                limit=validated.limit,
                rrf_k=self._config.rrf_k,
                vector_weight=self._config.hybrid_vector_weight,
            )
        except APIError:
            logger.exception("OpenAI API error while embedding query: %s", validated.query)
            raise
        except QDRANT_CONNECTION_ERRORS:
            logger.exception("Vector store error during hybrid search for: %s", validated.query)
            raise

    async def retrieve_context(self, query: str, limit: int = 3) -> list[str]:
        validated = RetrieveContextRequest.model_validate({"query": query, "limit": limit})
        results = await self.hybrid_search(validated.query, limit=validated.limit)
        return [result.content for result in results]

    async def close(self) -> None:
        if self._vector_store is not None:
            await self._vector_store.close()

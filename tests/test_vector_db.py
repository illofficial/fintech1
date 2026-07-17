from types import SimpleNamespace
from unittest.mock import AsyncMock

from app.models.rag import RAGConfig, ScoredDocument
from app.services.vector_db import (
    MockVectorStore,
    RAGService,
    _cosine_similarity,
    _keyword_score,
    create_vector_store,
)


class FakeEmbedder:
    """Deterministic embedder used to make similarity ranking predictable in tests."""

    def __init__(self, mapping: dict[str, list[float]]) -> None:
        self._mapping = mapping

    async def embed(self, text: str) -> list[float]:
        return self._mapping.get(text, [0.0, 0.0, 1.0])


SEED = [
    {"id": "wire", "content": "wire transfer daily limit"},
    {"id": "kyc", "content": "kyc identity verification refresh"},
]
EMBEDDINGS = {
    "wire transfer daily limit": [1.0, 0.0, 0.0],
    "kyc identity verification refresh": [0.0, 1.0, 0.0],
}


def test_keyword_score_rewards_overlap() -> None:
    assert _keyword_score("wire transfer", "wire transfer limit") > 0.0
    assert _keyword_score("unrelated", "wire transfer limit") == 0.0
    assert _keyword_score("", "anything") == 0.0


def test_cosine_similarity_bounds() -> None:
    assert _cosine_similarity([1.0, 0.0], [1.0, 0.0]) == 1.0
    assert _cosine_similarity([1.0, 0.0], [0.0, 1.0]) == 0.0
    assert _cosine_similarity([0.0, 0.0], [1.0, 1.0]) == 0.0


async def test_mock_store_vector_search_ranks_by_similarity() -> None:
    store = MockVectorStore(embedder=FakeEmbedder(EMBEDDINGS), seed_documents=SEED)
    results = await store.vector_search([1.0, 0.0, 0.0], limit=2)
    assert results[0].id == "wire"
    assert results[0].source == "vector"


async def test_mock_store_keyword_search_matches_tokens() -> None:
    store = MockVectorStore(embedder=FakeEmbedder(EMBEDDINGS), seed_documents=SEED)
    results = await store.keyword_search("kyc verification", limit=2)
    assert results[0].id == "kyc"
    assert results[0].source == "keyword"


def test_reciprocal_rank_fusion_merges_sources() -> None:
    vector = [ScoredDocument(id="a", content="a", score=1.0, source="vector")]
    keyword = [ScoredDocument(id="a", content="a", score=1.0, source="keyword")]
    fused = RAGService._reciprocal_rank_fusion(
        vector, keyword, limit=5, rrf_k=60, vector_weight=0.7
    )
    assert len(fused) == 1
    assert fused[0].source == "hybrid"


async def test_create_vector_store_uses_mock_when_forced() -> None:
    embedder = FakeEmbedder(EMBEDDINGS)
    store = await create_vector_store(RAGConfig(use_mock=True), embedder)  # type: ignore[arg-type]
    assert isinstance(store, MockVectorStore)


async def test_create_vector_store_falls_back_without_url() -> None:
    embedder = FakeEmbedder(EMBEDDINGS)
    store = await create_vector_store(RAGConfig(qdrant_url=None), embedder)  # type: ignore[arg-type]
    assert isinstance(store, MockVectorStore)


async def test_rag_service_retrieve_context_end_to_end() -> None:
    client = AsyncMock()
    client.embeddings.create = AsyncMock(
        return_value=SimpleNamespace(data=[SimpleNamespace(embedding=[1.0, 0.0, 0.0])])
    )
    store = MockVectorStore(embedder=FakeEmbedder(EMBEDDINGS), seed_documents=SEED)
    service = RAGService(client=client, config=RAGConfig(use_mock=True), vector_store=store)

    contexts = await service.retrieve_context("wire transfer daily limit", limit=1)
    assert contexts == ["wire transfer daily limit"]

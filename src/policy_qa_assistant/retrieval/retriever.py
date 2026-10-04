from __future__ import annotations

from policy_qa_assistant.config.settings import get_settings
from policy_qa_assistant.retrieval.embedding_service import EmbeddingService
from policy_qa_assistant.retrieval.vector_store import VectorStore


class Retriever:
    def __init__(self) -> None:
        self.embedding_service = EmbeddingService()
        self.vector_store = VectorStore()
        self.settings = get_settings()

    def retrieve(
        self, query: str, top_k: int | None = None, document_ids: list[str] | None = None
    ) -> list[dict]:
        if not query.strip():
            return []
        scope = set(document_ids) if document_ids is not None else None
        return self.vector_store.query(
            self.embedding_service.embed_text(query), top_k or self.settings.top_k, scope
        )

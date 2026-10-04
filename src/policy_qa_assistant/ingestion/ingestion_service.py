"""
Document ingestion service.

Coordinates loading, metadata extraction, duplicate detection,
and document chunking.
"""

from __future__ import annotations
import hashlib
from policy_qa_assistant.retrieval.embedding_service import EmbeddingService
from policy_qa_assistant.retrieval.vector_store import VectorStore

import uuid
from pathlib import Path
from typing import Dict, List

from policy_qa_assistant.ingestion.chunker import DocumentChunker
from policy_qa_assistant.ingestion.document_loader import (
    DocumentLoader,
)
from policy_qa_assistant.ingestion.duplicate_detector import DuplicateDetector
from policy_qa_assistant.ingestion.metadata import MetadataExtractor
from policy_qa_assistant.models.document_models import (
    DocumentChunk,
    DocumentMetadata,
    IngestionResponse,
)


class IngestionService:
    """Service responsible for document ingestion."""

    def __init__(self) -> None:
        self.loader = DocumentLoader()
        self.chunker = DocumentChunker()
        self.metadata_extractor = MetadataExtractor()
        self.duplicate_detector = DuplicateDetector()
        self.embedding_service = EmbeddingService()
        self.vector_store = VectorStore()


    def ingest_document(
        self,
        file_path: Path,
        original_file_name: str | None = None,
    ) -> Dict:
        """
        Ingest a single document.
        """

        file_hash = self.duplicate_detector.calculate_hash(file_path)

        if self.vector_store.has_hash(file_hash):
            return {
                "response": IngestionResponse(
                    success=False,
                    message="Duplicate document detected.",
                ),
                "chunks": [],
            }

        loaded = self.loader.load_document(file_path)

        extracted = self.metadata_extractor.extract(
            loaded["text"],
            loaded["title"],
        )

        document_metadata = DocumentMetadata(
            document_id=str(uuid.uuid4()),
            file_name=original_file_name or loaded["file_name"],
            title=extracted["title"],
            version=extracted["version"],
            effective_date=extracted["effective_date"],
            section_name=None,
            page_number=None,
            file_hash=file_hash,
        )

        chunks: List[DocumentChunk] = self.chunker.chunk_document(
            loaded["text"],
            document_metadata,
        )
        texts = [chunk.text for chunk in chunks]

        embeddings = self.embedding_service.embed_texts(texts)

        self.vector_store.add_chunks(
           chunks,
           embeddings,
        )

        return {
            "response": IngestionResponse(
                success=True,
                message="Document ingested successfully.",
                document_id=document_metadata.document_id,
                chunks_created=len(chunks),
            ),
            "chunks": chunks,
            "metadata": document_metadata,
        }

    def replace_document(self, document_id: str, content: str) -> Dict:
        """Re-index an existing policy while preserving its document identity."""
        if not content.strip():
            from policy_qa_assistant.ingestion.document_loader import EmptyDocumentError

            raise EmptyDocumentError("Document contains no readable text.")
        existing_chunks = self.vector_store.document_chunks(document_id)
        if not existing_chunks:
            raise KeyError("Document not found")

        existing = DocumentMetadata.model_validate(existing_chunks[0]["metadata"])
        extracted = self.metadata_extractor.extract(content, existing.title or existing.file_name)
        metadata = existing.model_copy(
            update={
                "title": extracted["title"],
                "version": extracted["version"],
                "effective_date": extracted["effective_date"],
                "file_hash": hashlib.sha256(content.encode("utf-8")).hexdigest(),
            }
        )
        chunks = self.chunker.chunk_document(content, metadata)
        embeddings = self.embedding_service.embed_texts([chunk.text for chunk in chunks])
        self.vector_store.replace_document_chunks(document_id, chunks, embeddings)
        return {
            "response": IngestionResponse(
                success=True,
                message="Document updated successfully.",
                document_id=document_id,
                chunks_created=len(chunks),
            ),
            "chunks": chunks,
            "metadata": metadata,
        }

"""
Models used during document ingestion.
"""

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


class DocumentMetadata(BaseModel):
    """
    Metadata extracted from an uploaded document.
    """

    document_id: str

    file_name: str

    title: Optional[str] = None

    version: Optional[str] = None

    effective_date: Optional[str] = None

    section_name: Optional[str] = None

    page_number: Optional[int] = None

    file_hash: str

    uploaded_at: datetime = Field(default_factory=datetime.utcnow)


class DocumentChunk(BaseModel):
    """
    Represents one retrievable chunk.
    """

    chunk_id: str

    document_id: str

    chunk_index: int

    text: str

    metadata: DocumentMetadata


class IngestionResponse(BaseModel):
    """
    API response after ingestion.
    """

    success: bool

    message: str

    document_id: Optional[str] = None

    chunks_created: int = 0
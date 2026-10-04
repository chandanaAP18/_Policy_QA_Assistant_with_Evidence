from __future__ import annotations

import re
import uuid
from policy_qa_assistant.config.settings import get_settings
from policy_qa_assistant.models.document_models import DocumentChunk, DocumentMetadata


class DocumentChunker:
    def __init__(self) -> None:
        self.settings = get_settings()

    def chunk_document(self, text: str, metadata: DocumentMetadata) -> list[DocumentChunk]:
        # Keep numbered policy sections together whenever possible.
        blocks = [block.strip() for block in re.split(r"(?=^\d+(?:\.\d+)*\s+)", text, flags=re.MULTILINE) if block.strip()]
        chunks: list[DocumentChunk] = []
        for block in blocks:
            heading = re.match(r"^(\d+(?:\.\d+)*\s+[^\n]+)", block)
            section = heading.group(1).strip() if heading else "Introduction"
            for offset in range(0, len(block), self.settings.chunk_size - self.settings.chunk_overlap):
                part = block[offset:offset + self.settings.chunk_size].strip()
                if not part:
                    continue
                chunk_metadata = metadata.model_copy(update={"section_name": section})
                chunks.append(DocumentChunk(chunk_id=str(uuid.uuid4()), document_id=metadata.document_id,
                                            chunk_index=len(chunks), text=part, metadata=chunk_metadata))
                if offset + self.settings.chunk_size >= len(block):
                    break
        return chunks

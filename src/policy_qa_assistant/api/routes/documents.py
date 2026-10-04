"""
Document ingestion API.
"""

from pathlib import Path
import hmac
import shutil
import tempfile

from fastapi import APIRouter, File, Header, HTTPException, UploadFile
from pydantic import BaseModel, Field
from policy_qa_assistant.config.settings import get_settings
from policy_qa_assistant.retrieval.vector_store import VectorStore

from policy_qa_assistant.ingestion.ingestion_service import (
    IngestionService,
)
from policy_qa_assistant.ingestion.document_loader import EmptyDocumentError, UnsupportedDocumentError

router = APIRouter(
    prefix="/documents",
    tags=["Documents"],
)

service = IngestionService()


class DocumentUpdate(BaseModel):
    content: str = Field(min_length=1, max_length=1_000_000)


@router.post("/ingest")
async def ingest_document(
    file: UploadFile = File(...),
):
    """
    Upload and ingest a policy document.
    """

    try:
        # Keep the client-provided filename in metadata.  A bare temporary
        # filename would otherwise make citations refer to an opaque upload.
        original_name = Path(file.filename or "policy-document").name
        suffix = Path(original_name).suffix

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=suffix,
            prefix=f"upload_{Path(original_name).stem}_",
        ) as temp_file:

            shutil.copyfileobj(file.file, temp_file)

            temp_path = Path(temp_file.name)

        try:
            result = service.ingest_document(temp_path, original_file_name=original_name)
        finally:
            temp_path.unlink(missing_ok=True)

        return result["response"]

    except (ValueError, OSError, EmptyDocumentError, UnsupportedDocumentError) as exc:

        raise HTTPException(
            status_code=400,
            detail=str(exc),
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Document ingestion failed") from exc


@router.get("")
def list_documents():
    return {"documents": VectorStore().documents()}


@router.get("/{document_id}")
def get_document(document_id: str):
    chunks = VectorStore().document_chunks(document_id)
    if not chunks:
        raise HTTPException(status_code=404, detail="Document not found")
    metadata = chunks[0]["metadata"]
    return {
        "document": metadata,
        "chunks": [
            {"section": row["metadata"].get("section_name"), "text": row["text"]}
            for row in chunks
        ],
    }


@router.put("/{document_id}")
def update_document(
    document_id: str,
    update: DocumentUpdate,
    x_policy_admin_key: str | None = Header(default=None),
):
    """Update a policy document and rebuild only its indexed evidence chunks."""
    configured_key = get_settings().policy_admin_key
    if not configured_key or not x_policy_admin_key or not hmac.compare_digest(
        x_policy_admin_key, configured_key
    ):
        raise HTTPException(status_code=403, detail="Not authorized to edit policy documents.")
    try:
        return service.replace_document(document_id, update.content)["response"]
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except EmptyDocumentError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

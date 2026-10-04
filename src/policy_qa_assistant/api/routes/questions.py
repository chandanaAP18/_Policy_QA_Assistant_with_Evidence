from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, ConfigDict, Field, field_validator

from policy_qa_assistant.services.question_service import QuestionService

router = APIRouter(prefix="/questions", tags=["Questions"])
service = QuestionService()


class QuestionRequest(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "question": "What is the probation period?",
                "top_k": 5,
            }
        }
    )

    question: str = Field(min_length=3, max_length=2000)
    top_k: int | None = Field(default=None, ge=1, le=20)
    document_ids: list[str] | None = Field(
        default=None,
        max_length=20,
        description="Optional document IDs. When supplied, search only these uploaded policies.",
    )

    @field_validator("document_ids")
    @classmethod
    def validate_document_ids(cls, document_ids: list[str] | None) -> list[str] | None:
        if document_ids is None:
            return None
        if not document_ids or any(not document_id.strip() for document_id in document_ids):
            raise ValueError("document_ids must contain one or more uploaded document IDs.")
        if any(document_id == "string" for document_id in document_ids):
            raise ValueError(
                "Replace the Swagger placeholder 'string' with an ID returned by POST /documents/ingest, or remove document_ids."
            )
        return document_ids


@router.post("")
def ask_question(request: QuestionRequest):
    question_id, response = service.ask(request.question, request.top_k, request.document_ids)
    return {"question_id": question_id, **response}


@router.get("/{question_id}")
def get_question(question_id: str):
    response = service.get(question_id)
    if not response:
        raise HTTPException(404, "Question not found")
    return {"question_id": question_id, **response}


@router.get("/{question_id}/debug")
def get_debug(question_id: str):
    response = service.get(question_id, debug=True)
    if not response:
        raise HTTPException(404, "Question not found")
    return response

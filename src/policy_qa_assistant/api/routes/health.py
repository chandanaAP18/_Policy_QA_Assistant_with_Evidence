"""Health check endpoint — used for uptime checks and container orchestration."""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from fastapi import APIRouter
from pydantic import BaseModel

from policy_qa_assistant.config.settings import get_settings

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Health"])


class HealthResponse(BaseModel):
    status: str
    app_env: str
    embedding_model: str
    gemini_model: str
    timestamp: str


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check",
    description="Returns basic liveness and configuration information for the service.",
)
def health_check() -> HealthResponse:
    settings = get_settings()
    logger.debug("Health check requested")
    return HealthResponse(
        status="ok",
        app_env=settings.app_env,
        embedding_model=settings.embedding_model,
        gemini_model=settings.gemini_model,
        timestamp=datetime.now(timezone.utc).isoformat(),
    )

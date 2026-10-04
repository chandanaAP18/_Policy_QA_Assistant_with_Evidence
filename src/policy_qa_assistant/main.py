"""
Policy Q&A Assistant with Evidence — FastAPI application entrypoint.

Run locally with:

    uvicorn policy_qa_assistant.main:app --reload --port 8000

Swagger UI:  http://localhost:8000/docs
ReDoc:       http://localhost:8000/redoc
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path
from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from policy_qa_assistant.api.routes.health import router as health_router
from policy_qa_assistant.api.routes.documents import router as documents_router
from policy_qa_assistant.config.settings import get_settings
from policy_qa_assistant.logging_config import configure_logging
from policy_qa_assistant.api.routes.questions import router as questions_router
from policy_qa_assistant.api.routes.evaluation import router as evaluation_router


configure_logging()
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup / shutdown hooks.

    Phase 1 only logs startup/shutdown and validates configuration.
    Later phases will initialize the embedding model, ChromaDB client,
    and Gemini client here so they are created once and shared via
    `app.state`, rather than re-created per-request.
    """
    settings = get_settings()
    logger.info(
        "Starting Policy Q&A Assistant | env=%s | embedding_model=%s | gemini_model=%s | top_k=%s",
        settings.app_env,
        settings.embedding_model,
        settings.gemini_model,
        settings.top_k,
    )
    if not settings.gemini_api_key:
        logger.warning(
            "GEMINI_API_KEY is not set. Answer generation endpoints will fail "
            "until a valid key is configured in .env."
        )

    settings.ensure_directories()
    logger.info("Verified/created required data directories")

    yield

    logger.info("Shutting down Policy Q&A Assistant")


def create_app() -> FastAPI:
    """Application factory — keeps app construction testable and import-safe."""
    settings = get_settings()

    app = FastAPI(
        title="Policy Q&A Assistant with Evidence",
        description=(
            "A Retrieval-Augmented Generation service that answers employee "
            "questions about company policies using ONLY supplied policy "
            "documents, with mandatory citations and refusal when evidence "
            "is insufficient."
        ),
        version="0.1.0",
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"] if settings.app_env == "development" else [],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    register_exception_handlers(app)
    register_routers(app)

    static_dir = Path(__file__).resolve().parent / "static"
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

    @app.get("/", include_in_schema=False)
    def frontend() -> FileResponse:
        """Serve the browser UI; Swagger remains available at /docs."""
        return FileResponse(static_dir / "index.html")

    return app


def register_exception_handlers(app: FastAPI) -> None:
    """Central place for translating unexpected errors into clean JSON responses."""

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        logger.exception("Unhandled exception on %s %s", request.method, request.url.path)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "error": "internal_server_error",
                "message": "An unexpected error occurred. Check server logs for details.",
            },
        )


def register_routers(app: FastAPI) -> None:
    """Register all API routers. New routers are added here as phases progress."""
    app.include_router(health_router)
    app.include_router(documents_router)
    app.include_router(questions_router)
    app.include_router(evaluation_router)
    # Phase 2+: documents_router, questions_router, evaluation_router, debug_router


app = create_app()

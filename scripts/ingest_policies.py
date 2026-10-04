"""Ingest the controlled policy collection: python scripts/ingest_policies.py"""
from pathlib import Path
from policy_qa_assistant.ingestion.ingestion_service import IngestionService
from policy_qa_assistant.config.settings import get_settings
from policy_qa_assistant.retrieval.vector_store import VectorStore

VectorStore().clear()
service = IngestionService()
for path in sorted(get_settings().policy_docs_abs_path.glob("*")):
    if path.suffix.lower() in service.loader.SUPPORTED_EXTENSIONS:
        result = service.ingest_document(path)["response"]
        print(f"{path.name}: {result.message}")

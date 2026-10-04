"""Persistent JSON vector store for the small controlled collection."""
from __future__ import annotations

import json
from pathlib import Path

from policy_qa_assistant.config.settings import get_settings


class VectorStore:
    def __init__(self) -> None:
        self.path: Path = get_settings().chroma_db_abs_path / "policy_index.json"
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def _read(self) -> list[dict]:
        if not self.path.exists():
            return []
        try:
            return json.loads(self.path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return []

    def _write(self, rows: list[dict]) -> None:
        self.path.write_text(json.dumps(rows, indent=2), encoding="utf-8")

    def add_chunks(self, chunks, embeddings) -> None:
        rows = self._read()
        rows.extend({"id": chunk.chunk_id, "text": chunk.text, "embedding": embedding,
                     "metadata": chunk.metadata.model_dump(mode="json"), "chunk_index": chunk.chunk_index}
                    for chunk, embedding in zip(chunks, embeddings, strict=True))
        self._write(rows)

    def replace_document_chunks(self, document_id: str, chunks, embeddings) -> None:
        """Replace all indexed chunks for one document after a policy update."""
        rows = [row for row in self._read() if row["metadata"].get("document_id") != document_id]
        rows.extend(
            {
                "id": chunk.chunk_id,
                "text": chunk.text,
                "embedding": embedding,
                "metadata": chunk.metadata.model_dump(mode="json"),
                "chunk_index": chunk.chunk_index,
            }
            for chunk, embedding in zip(chunks, embeddings, strict=True)
        )
        self._write(rows)

    def documents(self) -> list[dict]:
        seen: dict[str, dict] = {}
        for row in self._read():
            metadata = row["metadata"]
            seen[metadata["document_id"]] = metadata
        return list(seen.values())

    def document_chunks(self, document_id: str) -> list[dict]:
        """Return the stored evidence chunks for one ingested document."""
        return sorted(
            (row for row in self._read() if row["metadata"].get("document_id") == document_id),
            key=lambda row: row["chunk_index"],
        )

    def has_hash(self, file_hash: str) -> bool:
        return any(row["metadata"].get("file_hash") == file_hash for row in self._read())

    def query(self, embedding: list[float], limit: int, document_ids: set[str] | None = None) -> list[dict]:
        """Return the most similar chunks, optionally limited to named uploads."""
        def score(candidate: list[float]) -> float:
            return sum(a * b for a, b in zip(embedding, candidate))
        rows = self._read()
        if document_ids is not None:
            rows = [row for row in rows if row["metadata"].get("document_id") in document_ids]
        for row in rows:
            row["score"] = score(row["embedding"])
        return sorted(rows, key=lambda row: row["score"], reverse=True)[:limit]

    def count(self) -> int:
        return len(self._read())

    def clear(self) -> None:
        """Reset only the derived local index, ready for repeatable ingestion."""
        self._write([])

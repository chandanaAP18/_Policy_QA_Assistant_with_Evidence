"""Guards against citations that are absent from the retrieved evidence."""
from __future__ import annotations


class CitationValidationError(ValueError):
    pass


def validate_citations(citations: list[dict], retrieved_sources: list[dict]) -> None:
    if not citations:
        raise CitationValidationError("A supported answer requires at least one citation.")
    available = {(source["file_name"], source.get("section")) for source in retrieved_sources}
    for citation in citations:
        key = (citation.get("file_name"), citation.get("section"))
        if key not in available:
            raise CitationValidationError("Citation does not correspond to retrieved evidence.")

"""Evidence-only answering and query audit storage."""
from __future__ import annotations

import json
import re
import sqlite3
import uuid
from datetime import datetime, timezone

from policy_qa_assistant.config.settings import get_settings
from policy_qa_assistant.retrieval.retriever import Retriever
from policy_qa_assistant.retrieval.embedding_service import EmbeddingService
from policy_qa_assistant.services.citation_validator import CitationValidationError, validate_citations
from policy_qa_assistant.agents.policy_agent import PolicyEvidenceAgent

REFUSAL = "Not mentioned in the selected policy evidence."


class QuestionService:
    def __init__(self) -> None:
        self.retriever = Retriever()
        self.agent = PolicyEvidenceAgent()
        self.db_path = get_settings().questions_db_abs_path
        with sqlite3.connect(self.db_path) as connection:
            connection.execute("CREATE TABLE IF NOT EXISTS questions (id TEXT PRIMARY KEY, question TEXT, response TEXT, debug TEXT, created_at TEXT)")

    @staticmethod
    def _source(row: dict) -> dict:
        metadata = row["metadata"]
        return {"chunk_id": row["id"], "document": metadata["title"], "file_name": metadata["file_name"],
                "section": metadata.get("section_name"), "page": metadata.get("page_number"),
                "text": row["text"], "score": round(row["score"], 3)}

    @staticmethod
    def _answer_from_evidence(text: str, question: str) -> str:
        # The response is an extracted policy statement, never a generated claim.
        sentences = [item.strip() for item in text.split(".") if item.strip()]
        question_words = set(re.findall(r"[a-z0-9]+", question.lower()))
        best = max(sentences, key=lambda sentence: len(question_words & set(re.findall(r"[a-z0-9]+", sentence.lower()))))
        # Remove a numbered heading that may precede the selected statement.
        if "\n" in best and re.match(r"^\d+(?:\.\d+)*\s+", best):
            best = best.split("\n", 1)[1].strip()
        return best + "."

    @staticmethod
    def _is_ambiguous(question: str) -> bool:
        stop_words = {"can", "i", "we", "the", "a", "an", "what", "is", "are", "do", "does", "how", "about", "policy", "take"}
        words = [word for word in EmbeddingService._tokens(question) if word not in stop_words]
        return len(words) < 2

    @staticmethod
    def _has_conflict(rows: list[dict]) -> bool:
        texts = [row["text"].lower() for row in rows]
        return any("must not" in first and " may " in second for first in texts for second in texts if first != second)

    @staticmethod
    def _policy_terms(question: str) -> set[str]:
        # Question grammar is not policy evidence.  Without this filter, a
        # question such as "What is required on public Wi-Fi?" could match an
        # unrelated passage merely because it contains "what" or "required".
        generic = {
            "a", "an", "and", "are", "be", "can", "did", "do", "does", "for", "how",
            "after", "book", "class", "i", "in", "is", "me", "my", "many", "much", "need", "needs", "of", "on", "or", "the", "to",
            "was", "we", "what", "when", "where", "which", "who", "why", "with", "you",
            "s", "name", "names", "available", "take", "takes", "time",
            "employee", "employees", "company", "policy", "rule", "rules", "employment", "work",
            "working", "system", "systems", "required", "requirement",
        }
        return {word for word in EmbeddingService._tokens(question) if len(word) > 1} - generic

    @classmethod
    def _has_lexical_evidence(cls, question: str, text: str) -> bool:
        query_words = cls._policy_terms(question)
        evidence_words = set(EmbeddingService._tokens(text))
        return bool(query_words & evidence_words)

    @classmethod
    def _evidence_coverage(cls, question: str, texts: list[str]) -> float:
        """Measure whether policy text covers the meaningful question terms.

        A single shared word must not turn an unrelated question into a policy
        answer.  Combining relevant passages still supports valid questions
        that need evidence from more than one policy.
        """
        terms = cls._policy_terms(question)
        if not terms:
            return 0.0
        evidence_terms = set(EmbeddingService._tokens(" ".join(texts)))
        return len(terms & evidence_terms) / len(terms)

    @staticmethod
    def _has_missing_time_condition(question: str, texts: list[str]) -> bool:
        """Do not infer a policy decision for a date not stated in evidence."""
        requested = {word for word in ("today", "tomorrow", "yesterday") if word in question.lower()}
        return bool(requested - set(EmbeddingService._tokens(" ".join(texts))))

    @staticmethod
    def _is_explicitly_unsupported(question: str) -> bool:
        lower = question.lower()
        return any(marker in lower for marker in ("law", "legal", "my personal", "salary record", "employee record", "exception"))

    def ask(
        self, question: str, top_k: int | None = None, document_ids: list[str] | None = None
    ) -> tuple[str, dict]:
        rows = self.retriever.retrieve(question, top_k, document_ids)
        sources = [self._source(row) for row in rows]
        evidence_rows = [row for row in rows if self._has_lexical_evidence(question, row["text"])]
        best = evidence_rows[0] if evidence_rows else (rows[0] if rows else None)
        threshold = get_settings().min_confidence_for_support
        selected_rows = [
            row for row in rows
            if best
            and row["score"] >= max(threshold, best["score"] * 0.4)
            and self._has_lexical_evidence(question, row["text"])
        ]
        # Keep one best passage per document for concise multi-source answers.
        seen_documents: set[str] = set()
        selected_rows = [row for row in selected_rows if not (row["metadata"]["document_id"] in seen_documents or seen_documents.add(row["metadata"]["document_id"]))]
        reason = None
        evidence_coverage = self._evidence_coverage(question, [row["text"] for row in evidence_rows])
        supported = bool(best and best["score"] >= threshold and evidence_coverage >= 0.6)
        if self._is_ambiguous(question):
            supported, reason = False, "The question is too ambiguous to match a specific policy rule."
        elif self._is_explicitly_unsupported(question):
            supported, reason = False, "The requested legal, personal-record, or exception information is not supported by the supplied policies."
        elif self._has_missing_time_condition(question, [row["text"] for row in evidence_rows]):
            supported, reason = False, "The policy evidence does not state the requested date or time condition."
        elif self._has_conflict([row for row in selected_rows if self._has_lexical_evidence(question, row["text"])]):
            supported, reason = False, "Retrieved policy passages contain conflicting rules."
        elif not supported:
            reason = "No retrieved passage had sufficient relevance to support a reliable answer."
        selected_sources = [self._source(row) for row in selected_rows]
        fallback_answer = " ".join(self._answer_from_evidence(row["text"], question) for row in selected_rows)
        draft = self.agent.draft(question, selected_sources, fallback_answer) if supported else None
        if draft:
            selected_sources = [selected_sources[index] for index in draft.citation_indexes]
        citations = [{key: source[key] for key in ("document", "file_name", "section", "page")} for source in selected_sources] if supported else []
        if supported:
            try:
                validate_citations(citations, sources)
            except CitationValidationError as exc:
                supported, citations, reason = False, [], str(exc)
        response = {
            "answer": draft.answer if supported and draft else REFUSAL,
            "supported": supported,
            "citations": citations,
            "evidence_quotes": [
                {"document": source["document"], "section": source["section"], "quote": source["text"]}
                for source in selected_sources
            ] if supported else [],
            "retrieved_sources": sources,
            "confidence": round(best["score"] * evidence_coverage, 3) if best else 0.0,
            "reason_for_refusal": None if supported else reason,
        }
        question_id = str(uuid.uuid4())
        debug = {"question": question, "document_scope": document_ids, "agent_mode": draft.mode if draft else "refusal",
                 "retrieved_chunks": sources, "evidence_coverage": round(evidence_coverage, 3),
                 "final_context": "\n\n".join(source["text"] for source in sources),
                 "final_answer": response["answer"], "cited_sources": response["citations"]}
        with sqlite3.connect(self.db_path) as connection:
            connection.execute("INSERT INTO questions VALUES (?, ?, ?, ?, ?)",
                               (question_id, question, json.dumps(response), json.dumps(debug), datetime.now(timezone.utc).isoformat()))
        return question_id, response

    def get(self, question_id: str, debug: bool = False) -> dict | None:
        column = "debug" if debug else "response"
        with sqlite3.connect(self.db_path) as connection:
            row = connection.execute(f"SELECT {column} FROM questions WHERE id = ?", (question_id,)).fetchone()
        return json.loads(row[0]) if row else None

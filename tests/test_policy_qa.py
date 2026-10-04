from pathlib import Path

import pytest

from policy_qa_assistant.config.settings import get_settings
from policy_qa_assistant.ingestion.document_loader import EmptyDocumentError, UnsupportedDocumentError
from policy_qa_assistant.ingestion.ingestion_service import IngestionService
from policy_qa_assistant.retrieval.vector_store import VectorStore
from policy_qa_assistant.services.question_service import QuestionService
from policy_qa_assistant.services.citation_validator import CitationValidationError, validate_citations
from policy_qa_assistant.api.routes.questions import QuestionRequest


@pytest.fixture(autouse=True)
def isolated_store(tmp_path, monkeypatch):
    settings = get_settings()
    monkeypatch.setattr(settings, "chroma_db_path", str(tmp_path / "vectors"))
    monkeypatch.setattr(settings, "questions_db_path", str(tmp_path / "questions.db"))


@pytest.fixture
def ingested(tmp_path):
    policy = tmp_path / "leave.md"
    policy.write_text("Leave Policy\nVersion: 1\n\n1 Annual Leave\nEmployees receive 20 paid annual leave days each year.", encoding="utf-8")
    service = IngestionService()
    service.ingest_document(policy)
    return policy


def test_document_is_chunked_and_stored(ingested):
    assert VectorStore().count() >= 1


def test_document_chunks_can_be_read_back(ingested):
    document_id = VectorStore().documents()[0]["document_id"]
    chunks = VectorStore().document_chunks(document_id)
    assert chunks and any("20 paid annual leave days" in chunk["text"] for chunk in chunks)


def test_document_content_can_be_replaced(ingested):
    document_id = VectorStore().documents()[0]["document_id"]
    IngestionService().replace_document(
        document_id, "Leave Policy\n\n1 Annual Leave\nEmployees receive 25 paid annual leave days each year."
    )
    chunks = VectorStore().document_chunks(document_id)
    assert any("25 paid annual leave days" in chunk["text"] for chunk in chunks)
    assert not any("20 paid annual leave days" in chunk["text"] for chunk in chunks)


def test_duplicate_document_is_rejected(ingested):
    result = IngestionService().ingest_document(ingested)["response"]
    assert not result.success


def test_ingestion_can_preserve_uploaded_filename(tmp_path):
    path = tmp_path / "temporary-upload.md"
    path.write_text("Leave Policy\n\n1 Annual Leave\nEmployees receive 20 days.", encoding="utf-8")
    IngestionService().ingest_document(path, original_file_name="employee-leave-policy.md")
    assert VectorStore().documents()[0]["file_name"] == "employee-leave-policy.md"


def test_supported_question_has_matching_citation(ingested):
    _, answer = QuestionService().ask("How many annual leave days do employees receive?")
    assert answer["supported"]
    assert answer["citations"][0]["document"] == "Leave Policy"


def test_unsupported_question_is_refused(ingested):
    _, answer = QuestionService().ask("What is the tax rule in Brazil?")
    assert not answer["supported"]
    assert answer["reason_for_refusal"]


def test_empty_document_is_rejected(tmp_path):
    path = tmp_path / "empty.md"
    path.write_text("   ", encoding="utf-8")
    with pytest.raises(EmptyDocumentError):
        IngestionService().ingest_document(path)


def test_unsupported_document_is_rejected(tmp_path):
    path = tmp_path / "policy.csv"
    path.write_text("a,b", encoding="utf-8")
    with pytest.raises(UnsupportedDocumentError):
        IngestionService().ingest_document(path)


def test_empty_query_returns_no_results():
    assert QuestionService().retriever.retrieve("   ") == []


def test_debug_record_contains_evidence(ingested):
    question_id, _ = QuestionService().ask("annual leave")
    debug = QuestionService().get(question_id, debug=True)
    assert debug["retrieved_chunks"] and debug["final_context"]


def test_terminology_variant_is_supported(ingested):
    _, answer = QuestionService().ask("How much vacation time is available?")
    assert answer["supported"]


def test_ambiguous_question_is_refused(ingested):
    _, answer = QuestionService().ask("Can I leave?")
    assert not answer["supported"]
    assert "ambiguous" in answer["reason_for_refusal"].lower()


def test_missing_citation_is_rejected():
    with pytest.raises(CitationValidationError):
        validate_citations([], [{"file_name": "policy.md", "section": "1 Rule"}])


def test_invalid_citation_is_rejected():
    with pytest.raises(CitationValidationError):
        validate_citations([{"file_name": "other.md", "section": "1 Rule"}], [{"file_name": "policy.md", "section": "1 Rule"}])


def test_conflicting_passages_are_detected():
    rows = [{"text": "Employees must not use a personal device."}, {"text": "Employees may use a personal device."}]
    assert QuestionService._has_conflict(rows)


def test_multi_document_question_uses_two_sources(tmp_path):
    first = tmp_path / "remote.md"
    first.write_text("Remote Policy\n\n1 Overseas Work\nCross-border remote work requires written HR approval.", encoding="utf-8")
    second = tmp_path / "security.md"
    second.write_text("Security Policy\n\n1 Wi-Fi\nPublic Wi-Fi requires use of the company VPN.", encoding="utf-8")
    service = IngestionService()
    service.ingest_document(first)
    service.ingest_document(second)
    _, answer = QuestionService().ask("Can I work overseas and use WiFi?")
    assert answer["supported"]
    assert len(answer["citations"]) >= 2


def test_question_scoped_to_uploaded_policy_never_uses_another_document(tmp_path):
    leave = tmp_path / "leave.md"
    leave.write_text(
        "Leave Policy\n\n1 Annual Leave\nEmployees receive 20 paid annual leave days each year.",
        encoding="utf-8",
    )
    security = tmp_path / "security.md"
    security.write_text(
        "Security Policy\n\n1 VPN\nPublic Wi-Fi requires use of the company VPN.", encoding="utf-8"
    )
    service = IngestionService()
    leave_result = service.ingest_document(leave)
    service.ingest_document(security)

    _, answer = QuestionService().ask(
        "What is required on public Wi-Fi?", document_ids=[leave_result["metadata"].document_id]
    )

    assert not answer["supported"]
    assert answer["answer"] == "Not mentioned in the selected policy evidence."
    assert all(source["document"] == "Leave Policy" for source in answer["retrieved_sources"])


def test_question_words_do_not_make_unrelated_evidence_look_supported(ingested):
    _, answer = QuestionService().ask("What is required on public Wi-Fi?")
    assert not answer["supported"]


def test_person_name_question_is_refused_when_role_is_absent(tmp_path):
    policy = tmp_path / "employee.md"
    policy.write_text(
        "Employee Policy Handbook\n\n1 Records\nEmployee name and work schedule records are maintained by HR.",
        encoding="utf-8",
    )
    IngestionService().ingest_document(policy)
    _, answer = QuestionService().ask("What is the company's CEO's name?")
    assert not answer["supported"]
    assert answer["answer"] == "Not mentioned in the selected policy evidence."


def test_calendar_holiday_question_is_not_treated_as_leave(ingested):
    _, answer = QuestionService().ask("Is tomorrow a holiday?")
    assert not answer["supported"]
    assert answer["answer"] == "Not mentioned in the selected policy evidence."


def test_partially_matching_question_is_refused_when_policy_lacks_its_condition(ingested):
    _, answer = QuestionService().ask("Can I take annual leave tomorrow?")
    assert not answer["supported"]
    assert answer["answer"] == "Not mentioned in the selected policy evidence."


def test_swagger_document_id_placeholder_is_rejected():
    with pytest.raises(ValueError, match="Swagger placeholder"):
        QuestionRequest(question="What is the probation period?", document_ids=["string"])

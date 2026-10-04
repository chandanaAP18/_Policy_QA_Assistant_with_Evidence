"""Reproducible evaluation against the supplied evaluator question set."""
from __future__ import annotations

import json
from pathlib import Path

from fastapi import APIRouter

from policy_qa_assistant.config.settings import get_settings
from policy_qa_assistant.services.question_service import QuestionService

router = APIRouter(prefix="/evaluation", tags=["Evaluation"])


@router.post("/run")
def run_evaluation():
    cases = json.loads((get_settings().project_root / "evaluation" / "questions.json").read_text(encoding="utf-8"))
    service = QuestionService()
    results = []
    debug_examples = []
    for case in cases:
        question_id, response = service.ask(case["question"])
        if len(debug_examples) < 3:
            debug_examples.append(service.get(question_id, debug=True))
        cited_documents = {citation["document"] for citation in response["citations"]}
        expected_documents = set(case["expected_documents"])
        results.append({"question": case["question"], "expected_support": case["should_answer"],
                        "actual_support": response["supported"],
                        "retrieval_hit": expected_documents.issubset({source["document"] for source in response["retrieved_sources"]}) if case["should_answer"] else None,
                        "citation_correct": not response["supported"] or expected_documents.issubset(cited_documents),
                        "unsupported_claim": response["supported"] and not case["should_answer"]})
    supported = [item for item in results if item["expected_support"]]
    refused = [item for item in results if not item["expected_support"]]
    mean = lambda values: round(sum(values) / len(values), 3) if values else 0.0
    report = {
        "total_questions": len(results),
        "retrieval_hit_rate": mean([item["retrieval_hit"] for item in supported]),
        "citation_correctness": mean([item["citation_correct"] for item in supported]),
        "supported_answer_accuracy": mean([item["actual_support"] for item in supported]),
        "unsupported_question_refusal_rate": mean([not item["actual_support"] for item in refused]),
        "answers_containing_unsupported_claims": sum(item["unsupported_claim"] for item in results),
        "results": results,
    }
    output = get_settings().project_root / "evaluation" / "latest_results.json"
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    (get_settings().project_root / "evaluation" / "debug_examples.json").write_text(
        json.dumps(debug_examples, indent=2), encoding="utf-8"
    )
    return report

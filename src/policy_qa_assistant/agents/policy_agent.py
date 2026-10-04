"""Agent workflow for grounded policy answers.

The agent has explicit retrieval, evidence assessment, drafting, and citation
verification stages. Gemini is used when configured; a deterministic extractive
fallback keeps the app usable without an API key or network access.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass

from policy_qa_assistant.config.settings import get_settings

logger = logging.getLogger(__name__)


@dataclass
class AgentDraft:
    answer: str
    citation_indexes: list[int]
    mode: str


class PolicyEvidenceAgent:
    def __init__(self) -> None:
        self.settings = get_settings()

    def draft(self, question: str, sources: list[dict], fallback: str) -> AgentDraft:
        if not self.settings.enable_llm or not self.settings.gemini_api_key:
            return AgentDraft(fallback, list(range(len(sources))), "extractive-fallback")
        try:
            draft = self._draft_with_gemini(question, sources)
            if not self._is_verbatim_grounded(draft.answer, sources):
                raise ValueError("Model draft contains wording not present in cited policy evidence")
            return draft
        except Exception as exc:  # Provider failures never weaken evidence safeguards.
            logger.warning("LLM draft unavailable; using extractive fallback: %s", type(exc).__name__)
            return AgentDraft(fallback, list(range(len(sources))), "extractive-fallback")

    @staticmethod
    def _is_verbatim_grounded(answer: str, sources: list[dict]) -> bool:
        """Only permit text that is present in the cited source passages."""
        normalise = lambda value: " ".join(value.lower().replace("\n", " ").split())
        passages = [normalise(source["text"]) for source in sources]
        statements = [normalise(statement) for statement in answer.split(".") if normalise(statement)]
        return bool(statements) and all(any(statement in passage for passage in passages) for statement in statements)

    def _draft_with_gemini(self, question: str, sources: list[dict]) -> AgentDraft:
        from google import genai

        evidence = [
            {"index": index, "document": source["document"], "section": source["section"], "text": source["text"]}
            for index, source in enumerate(sources)
        ]
        prompt = f'''You are a policy assistant. Answer ONLY from the evidence below.
If the evidence does not answer the question, return an empty answer and no citations.
Do not add recommendations, assumptions, or general knowledge.
Return strict JSON only: {{"answer":"...","citation_indexes":[0]}}.

Question: {question}
Evidence: {json.dumps(evidence)}'''
        client = genai.Client(api_key=self.settings.gemini_api_key)
        result = client.models.generate_content(model=self.settings.gemini_model, contents=prompt)
        payload = json.loads(result.text.strip().removeprefix("```json").removesuffix("```").strip())
        answer = str(payload.get("answer", "")).strip()
        indexes = [index for index in payload.get("citation_indexes", []) if isinstance(index, int) and 0 <= index < len(sources)]
        if not answer or not indexes:
            raise ValueError("Malformed model response or missing citation")
        return AgentDraft(answer, indexes, "gemini")

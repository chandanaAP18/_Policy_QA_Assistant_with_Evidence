"""
Formats the final response returned to the API.
"""

from typing import List, Dict


class AnswerFormatter:

    @staticmethod
    def format(
        answer: str,
        citations: List[Dict],
        confidence: float,
        supported: bool,
    ) -> Dict:

        return {
            "answer": answer,
            "supported": supported,
            "confidence": confidence,
            "citations": citations,
            "retrieved_sources": len(citations),
        }
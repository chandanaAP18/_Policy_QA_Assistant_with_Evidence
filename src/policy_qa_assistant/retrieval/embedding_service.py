"""Small deterministic embedding service used for the controlled policy corpus.

It deliberately has no network or model-download dependency, making the project
reproducible in a classroom environment.  The vector is a hashed bag-of-words
embedding and is persisted alongside each chunk.
"""
from __future__ import annotations

import hashlib
import math
import re


class EmbeddingService:
    dimensions = 256

    @staticmethod
    def _tokens(text: str) -> list[str]:
        normalised = text.lower()
        replacements = {
            # "Holiday" is deliberately not mapped to leave: a question about
            # tomorrow being a public holiday is not a leave-policy question.
            "time off": "leave", "vacation": "leave",
            "overseas": "cross border", "abroad": "cross border",
            "wireless internet": "wi fi", "wifi": "wi fi",
            "reimburse": "expense", "claim back": "expense",
        }
        for phrase, replacement in replacements.items():
            normalised = normalised.replace(phrase, replacement)
        normalised = re.sub(r"\bexpenses\b", "expense", normalised)
        normalised = re.sub(r"\breceipts\b", "receipt", normalised)
        normalised = re.sub(r"\bflights\b", "flight", normalised)
        normalised = re.sub(r"\bhours\b", "hour", normalised)
        return re.findall(r"[a-z0-9]+", normalised)

    def embed_text(self, text: str) -> list[float]:
        vector = [0.0] * self.dimensions
        for token in self._tokens(text):
            index = int(hashlib.sha256(token.encode("utf-8")).hexdigest(), 16) % self.dimensions
            vector[index] += 1.0
        norm = math.sqrt(sum(value * value for value in vector))
        return [value / norm for value in vector] if norm else vector

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [self.embed_text(text) for text in texts]

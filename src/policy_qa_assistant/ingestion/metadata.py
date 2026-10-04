"""
Metadata extraction for policy documents.
"""

import re
from typing import Dict, List, Optional


class MetadataExtractor:
    """
    Extracts metadata from policy documents.
    """

    VERSION_PATTERN = re.compile(r"Version\s*[:\-]?\s*(.+)", re.IGNORECASE)
    EFFECTIVE_DATE_PATTERN = re.compile(
        r"Effective Date\s*[:\-]?\s*(.+)",
        re.IGNORECASE,
    )

    SECTION_PATTERN = re.compile(
        r"^\d+(\.\d+)*\s+.+",
        re.MULTILINE,
    )

    def extract(
        self,
        text: str,
        default_title: str,
    ) -> Dict:
        """
        Extract metadata from document text.
        """

        title = self._extract_title(text) or default_title

        version = self._search(
            self.VERSION_PATTERN,
            text,
        )

        effective_date = self._search(
            self.EFFECTIVE_DATE_PATTERN,
            text,
        )

        sections = self._extract_sections(text)

        return {
            "title": title,
            "version": version,
            "effective_date": effective_date,
            "sections": sections,
        }

    @staticmethod
    def _extract_title(text: str) -> Optional[str]:
        """
        Assumes first non-empty line is the title.
        """

        for line in text.splitlines():
            line = line.strip()

            if line:
                return line.lstrip("#").strip()

        return None

    @staticmethod
    def _search(pattern, text):
        match = pattern.search(text)

        if match:
            return match.group(1).strip()

        return None

    def _extract_sections(
        self,
        text: str,
    ) -> List[str]:
        """
        Finds numbered section headings.
        """

        return [match.group(0) for match in self.SECTION_PATTERN.finditer(text)]

"""
Document Loader

Loads supported document types and extracts text.
"""

from pathlib import Path
from typing import Dict

from pypdf import PdfReader
from docx import Document


class UnsupportedDocumentError(Exception):
    """Raised when an unsupported document type is uploaded."""
    pass


class EmptyDocumentError(Exception):
    """Raised when a document contains no readable text."""
    pass


class DocumentLoader:
    """
    Loads supported document formats and extracts text.
    """

    SUPPORTED_EXTENSIONS = {
        ".pdf",
        ".docx",
        ".md",
        ".txt",
    }

    def load_document(self, file_path: Path) -> Dict:
        """
        Load a document and return extracted text and metadata.
        """

        suffix = file_path.suffix.lower()

        if suffix not in self.SUPPORTED_EXTENSIONS:
            raise UnsupportedDocumentError(
                f"Unsupported file type: {suffix}"
            )

        if suffix == ".pdf":
            text = self._load_pdf(file_path)

        elif suffix == ".docx":
            text = self._load_docx(file_path)

        elif suffix == ".md":
            text = self._load_text(file_path)

        elif suffix == ".txt":
            text = self._load_text(file_path)

        if not text.strip():
            raise EmptyDocumentError(
                "Document contains no readable text."
            )

        return {
            "file_name": file_path.name,
            "title": file_path.stem,
            "text": text,
        }

    def _load_pdf(self, file_path: Path) -> str:
        """
        Extract text from PDF.
        """

        reader = PdfReader(file_path)

        pages = []

        for page in reader.pages:
            page_text = page.extract_text()

            if page_text:
                pages.append(page_text)

        return "\n".join(pages)

    def _load_docx(self, file_path: Path) -> str:
        """
        Extract text from DOCX.
        """

        document = Document(file_path)

        paragraphs = [
            paragraph.text
            for paragraph in document.paragraphs
            if paragraph.text.strip()
        ]

        return "\n".join(paragraphs)

    def _load_text(self, file_path: Path) -> str:
        """
        Load Markdown or TXT files.
        """

        return file_path.read_text(
            encoding="utf-8",
            errors="ignore",
        )
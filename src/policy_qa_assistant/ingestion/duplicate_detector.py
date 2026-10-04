"""
Duplicate document detection using SHA-256 hashes.
"""

from pathlib import Path
import hashlib


class DuplicateDetector:
    """Calculates file hashes for duplicate detection."""

    @staticmethod
    def calculate_hash(file_path: Path) -> str:
        sha256 = hashlib.sha256()

        with open(file_path, "rb") as f:
            while chunk := f.read(8192):
                sha256.update(chunk)

        return sha256.hexdigest()
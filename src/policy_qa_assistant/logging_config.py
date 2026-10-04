"""
Centralized logging configuration for the Policy Q&A Assistant.

Call `configure_logging()` once, at application startup (done in `main.py`).
Every other module should simply do:

    import logging
    logger = logging.getLogger(__name__)

and log normally — handlers, formatting, and log file rotation are all
configured here in one place.
"""

from __future__ import annotations

import logging
import logging.handlers
import sys
from pathlib import Path

from policy_qa_assistant.config.settings import get_settings

_CONFIGURED = False

LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def configure_logging() -> None:
    """Configure the root logger with console + rotating file handlers.

    Idempotent: calling this more than once (e.g. in tests that import
    `main` multiple times) will not duplicate handlers.
    """
    global _CONFIGURED
    if _CONFIGURED:
        return

    settings = get_settings()
    log_dir: Path = settings.log_dir_abs_path
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "app.log"

    root_logger = logging.getLogger()
    root_logger.setLevel(settings.log_level)

    formatter = logging.Formatter(fmt=LOG_FORMAT, datefmt=DATE_FORMAT)

    console_handler = logging.StreamHandler(stream=sys.stdout)
    console_handler.setFormatter(formatter)
    console_handler.setLevel(settings.log_level)

    file_handler = logging.handlers.RotatingFileHandler(
        filename=str(log_file),
        maxBytes=5 * 1024 * 1024,  # 5 MB
        backupCount=5,
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)
    file_handler.setLevel(settings.log_level)

    root_logger.handlers.clear()
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)

    # Quiet down noisy third-party libraries unless we're debugging.
    for noisy_logger in ("httpx", "httpcore", "urllib3", "chromadb", "sentence_transformers"):
        logging.getLogger(noisy_logger).setLevel(logging.WARNING)

    _CONFIGURED = True
    logging.getLogger(__name__).info(
        "Logging configured (level=%s, file=%s)", settings.log_level, log_file
    )

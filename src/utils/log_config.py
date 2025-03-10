"""Logging configuration utilities."""

import logging
from pathlib import Path

from src.validation import LogLevel


def setup_logging(
    level: LogLevel = LogLevel.INFO,
    log_file: str | None = None,
    log_format: str | None = None,
) -> None:
    """Set up logging configuration.

    Args:
        level: The logging level to use.
        log_file: Optional path to a log file.
        log_format: Optional custom log format string.

    """
    if log_format is None:
        log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # Remove existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Add console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(logging.Formatter(log_format))
    root_logger.addHandler(console_handler)

    # Add file handler if specified
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(logging.Formatter(log_format))
        root_logger.addHandler(file_handler)

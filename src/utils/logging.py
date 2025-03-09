"""Logging utilities for the agent."""

import logging
import os

# Constants
DEFAULT_LOG_LEVEL = logging.INFO
DEFAULT_LOG_FORMAT = "%(levelname)s: %(message)s"
DEFAULT_FILE_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
DEFAULT_LOGS_DIR = "logs"
MIN_HANDLERS = 2

# Third-party loggers to adjust
THIRD_PARTY_LOGGERS = [
    "urllib3",
    "google.api_core",
    "google.auth",
    "google.oauth2",
]

# Global flag to track if logging has been set up
_logging_initialized = False


def setup_logging(
    name: str,
    level: int = logging.INFO,
    log_file: str | None = None,
    format_str: str | None = None,
) -> logging.Logger:
    """Set up logging configuration.

    Args:
        name: Logger name.
        level: Logging level.
        log_file: Optional log file path.
        format_str: Optional format string.

    Returns:
        Configured logger.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Clear existing handlers
    logger.handlers.clear()

    # Create formatter
    formatter = logging.Formatter(format_str or DEFAULT_FILE_FORMAT)

    # Add console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # Add file handler if specified
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    # Ensure minimum number of handlers
    while len(logger.handlers) < MIN_HANDLERS:
        handler = logging.StreamHandler()
        handler.setLevel(level)
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    # Set third-party loggers to WARNING level
    for logger_name in THIRD_PARTY_LOGGERS:
        third_party_logger = logging.getLogger(logger_name)
        third_party_logger.setLevel(logging.WARNING)

    return logger


def get_logger(
    name: str,
    level: int = logging.INFO,
    log_file: str | None = None,
    format_str: str | None = None,
) -> logging.Logger:
    """Get or create a logger with the specified configuration.

    Args:
        name: Logger name.
        level: Logging level.
        log_file: Optional log file path.
        format_str: Optional format string.

    Returns:
        Configured logger.
    """
    return setup_logging(name, level, log_file, format_str)

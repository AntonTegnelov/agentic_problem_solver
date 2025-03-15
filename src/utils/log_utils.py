"""Logging utilities."""

from __future__ import annotations

import json
import logging
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

# Constants
MAX_TASK_DESCRIPTION_LENGTH = 100


def setup_logging(
    *,  # Force keyword arguments
    level: int = logging.INFO,
    log_file: str | None = None,
    verbose: bool = False,
) -> None:
    """Set up logging configuration.

    Args:
        level: Logging level.
        log_file: Path to log file.
        verbose: Whether to enable verbose logging.

    """
    # Set up root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)

    # Create formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    console_handler.setFormatter(formatter)

    # Add console handler to root logger
    root_logger.addHandler(console_handler)

    # Add file handler if specified
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

    # Set verbose logging
    if verbose:
        root_logger.setLevel(logging.DEBUG)


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance.

    Args:
        name: Logger name.

    Returns:
        Logger instance.

    """
    return logging.getLogger(name)


@dataclass
class DelegationInfo:
    """Information about a delegation decision."""

    source_agent_id: str
    target_agent_id: str
    task: str
    reason: str
    additional_info: dict[str, Any] | None = None


def log_delegation_decision(
    logger: logging.Logger,
    delegation_info: DelegationInfo,
) -> None:
    """Log a delegation decision made by an agent.

    Args:
        logger: Logger instance.
        delegation_info: Information about the delegation decision.

    """
    log_data = asdict(delegation_info)
    logger.info("Delegation decision: %s", json.dumps(log_data))

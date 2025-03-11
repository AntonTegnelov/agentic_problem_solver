"""Tests for log utilities."""

import logging
import tempfile
from pathlib import Path

from src.utils.log_utils import get_logger, setup_logging


def test_setup_logging_basic() -> None:
    """Test basic setup_logging functionality."""
    # Reset root logger handlers
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Test with default parameters
    setup_logging()

    # Check that the root logger has at least one handler
    assert root_logger.handlers
    assert any(isinstance(h, logging.StreamHandler) for h in root_logger.handlers)
    assert root_logger.level == logging.INFO


def test_setup_logging_with_file() -> None:
    """Test setup_logging with a log file."""
    # Reset root logger handlers
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Create a temporary log file
    with tempfile.TemporaryDirectory() as temp_dir:
        log_file = Path(temp_dir) / "test.log"

        # Set up logging with the file
        setup_logging(log_file=str(log_file))

        # Check that the file handler was added
        file_handlers = [
            h for h in root_logger.handlers if isinstance(h, logging.FileHandler) and h.baseFilename == str(log_file)
        ]
        assert file_handlers

        # Check that the log file was created
        assert log_file.exists()

        # Close and remove the file handler before the temp directory is deleted
        for handler in file_handlers:
            handler.close()
            root_logger.removeHandler(handler)


def test_setup_logging_verbose() -> None:
    """Test setup_logging with verbose flag."""
    # Reset root logger handlers
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Set up logging with verbose flag
    setup_logging(verbose=True)

    # Check that the log level was set to DEBUG
    assert root_logger.level == logging.DEBUG


def test_setup_logging_custom_level() -> None:
    """Test setup_logging with custom level."""
    # Reset root logger handlers
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Set up logging with custom level
    setup_logging(level=logging.WARNING)

    # Check that the log level was set correctly
    assert root_logger.level == logging.WARNING
    assert all(h.level == logging.WARNING for h in root_logger.handlers)


def test_get_logger() -> None:
    """Test get_logger function."""
    # Get a logger
    logger_name = "test_logger"
    logger = get_logger(logger_name)

    # Check that the logger has the correct name
    assert logger.name == logger_name
    assert isinstance(logger, logging.Logger)

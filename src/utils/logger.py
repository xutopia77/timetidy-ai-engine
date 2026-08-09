"""
Logging utility for AI Engine

Provides unified logging with timestamp and log levels.

Usage:
    from utils.logger import log
    
    log.debug("Debug message")
    log.info("Info message")
    log.warning("Warning message")
    log.error("Error message")
"""

import logging
import sys
import os
from typing import Optional


class _AbbrevFormatter(logging.Formatter):
    _LEVEL_ABBREV = {
        "TRACE": "T",
        "DEBUG": "D",
        "INFO": "I",
        "WARNING": "W",
        "ERROR": "E",
        "CRITICAL": "C",
        "FATAL": "F",
    }

    def format(self, record: logging.LogRecord) -> str:
        record.levelname = self._LEVEL_ABBREV.get(record.levelname, record.levelname)
        return super().format(record)


def setup_logger(
    name: str = "ai-engine",
    level: Optional[int] = None
) -> logging.Logger:
    """Setup logger with unified format
    
    Args:
        name: Logger name
        level: Log level (DEBUG, INFO, WARNING, ERROR). 
               If None, reads from AI_ENGINE_LOG_LEVEL env var, defaults to INFO.
    
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    
    # Return existing logger if already configured
    if logger.handlers:
        return logger
    
    # Determine log level
    if level is None:
        env_level = os.getenv("AI_ENGINE_LOG_LEVEL", "INFO").upper()
        level_map = {
            "DEBUG": logging.DEBUG,
            "INFO": logging.INFO,
            "WARNING": logging.WARNING,
            "ERROR": logging.ERROR,
        }
        level = level_map.get(env_level, logging.INFO)
    
    logger.setLevel(level)
    
    # Console handler (stderr)
    handler = logging.StreamHandler(sys.stderr)
    handler.setLevel(level)
    
    # Format with timestamp and level
    formatter = _AbbrevFormatter(
        fmt="[%(levelname)s][%(asctime)s.%(msecs)03d]%(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    
    return logger


def set_log_level(level: int) -> None:
    """Set log level at runtime
    
    Args:
        level: logging.DEBUG, INFO, WARNING, or ERROR
    """
    log.setLevel(level)
    for handler in log.handlers:
        handler.setLevel(level)


def set_log_file(log_file_path: Optional[str] = None) -> None:
    """Set log file path for file logging
    
    Args:
        log_file_path: Path to log file. If None or empty, file logging is disabled.
    """
    # Remove existing file handlers
    for handler in log.handlers[:]:
        if isinstance(handler, logging.FileHandler):
            log.removeHandler(handler)
    
    # Add new file handler if path is provided
    if log_file_path:
        # Ensure directory exists
        log_dir = os.path.dirname(log_file_path)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)
        
        # Create file handler
        file_handler = logging.FileHandler(log_file_path, encoding='utf-8')
        file_handler.setLevel(log.level)
        
        # Use same format as console
        formatter = _AbbrevFormatter(
            fmt="[%(asctime)s.%(msecs)03d] [%(levelname)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        file_handler.setFormatter(formatter)
        log.addHandler(file_handler)


# Global logger instance for import
log = setup_logger()
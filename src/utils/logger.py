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
    formatter = logging.Formatter(
        fmt="[%(asctime)s.%(msecs)03d] [%(levelname)s] %(message)s",
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


# Global logger instance for import
log = setup_logger()

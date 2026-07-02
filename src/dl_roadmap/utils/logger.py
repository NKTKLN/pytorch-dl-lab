"""Application logging configuration module.

This module provides the setup_logger function for configuring log output
to the console and, optionally, to a file.
"""

import sys
from dataclasses import dataclass

from loguru import logger


@dataclass
class LoggerConfig:
    """Configuration options for setup_logger.

    Attributes:
        disable_logging: If True, all logging sinks are removed and
            setup_logger becomes a no-op.
        log_level: Minimum level to emit (e.g. "DEBUG", "INFO").
        log_path: File path to write logs to. If empty, file logging
            is disabled and only console output is configured.
        log_format: loguru format string shared by the console and
            file sinks.
    """

    disable_logging: bool = False
    log_level: str = "INFO"
    log_path: str = ""

    log_format: str = "<cyan>[{time:HH:mm:ss}]</cyan> <lvl>[{level}]</lvl> - {message}"


def setup_logger(config: LoggerConfig) -> None:
    """Configure loguru sinks for console and optional file output.

    Removes any existing sinks first, then adds a colorized stdout
    sink and, if config.log_path is set, a rotating file sink.

    Args:
        config: Logging options; see LoggerConfig.
    """
    logger.remove()

    if config.disable_logging:
        return

    logger.add(
        sys.stdout,
        format=config.log_format,
        level=config.log_level,
        colorize=True,
        enqueue=True,
        backtrace=True,
        diagnose=True,
    )

    if config.log_path:
        logger.debug(f"Adding file logger: {config.log_path}")
        logger.add(
            config.log_path,
            format=config.log_format,
            level=config.log_level,
            colorize=False,
            enqueue=True,
            backtrace=True,
            diagnose=True,
            rotation="10 MB",
            retention="10 days",
            compression="zip",
        )
    else:
        logger.debug("Log file path is not specified; file logging is disabled")

    logger.debug(f"Logging level is set to: {config.log_level}")
    if config.log_path:
        logger.debug(f"Logs will be written to file: {config.log_path}")
    else:
        logger.debug("Log file is not specified; output is console-only")

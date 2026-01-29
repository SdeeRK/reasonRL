import sys

from loguru import logger


def setup_logger():
    """Configure loguru logger with a structured format."""
    logger.remove()
    log_format = "<cyan>{module}</cyan>:<cyan>{function}</cyan> | <level>{level: <8}</level> | <level>{message}</level>"
    logger.add(sys.stderr, format=log_format, level="INFO", colorize=True)


def get_logger(name: str | None = None):
    """
    Get a logger instance.

    Args:
        name: The name of the logger (typically __name__).
              If provided, binds the 'name' extra field.
              Loguru usually detects module name automatically, but this allows specific naming.
    """
    # Simply return the global logger.
    # Loguru handles context automatically, but if users want standard `get_logger(__name__)`
    # we can support binding, though loguru prefers auto-detection.
    # However, to support standard pattern:
    if name:
        return logger.bind(name=name)
    return logger

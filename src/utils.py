"""Shared utilities: logging and timing."""

from __future__ import annotations

import logging
import sys
import time
from contextlib import contextmanager
from typing import Iterator

from config import settings

_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"


def get_logger(name: str) -> logging.Logger:
    """Return a module logger writing to stderr with a consistent format."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(logging.Formatter(_FORMAT))
        logger.addHandler(handler)
        logger.setLevel(settings.log_level)
        logger.propagate = False
    return logger


@contextmanager
def timed(logger: logging.Logger, label: str) -> Iterator[None]:
    """Log how long the enclosed block took."""
    start = time.perf_counter()
    yield
    logger.info("%s finished in %.2fs", label, time.perf_counter() - start)

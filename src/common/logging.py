"""Tiny logging helper so every stage prints consistently."""
from __future__ import annotations

import logging
import sys

_CONFIGURED = False


def get_logger(name: str = "3dgs") -> logging.Logger:
    global _CONFIGURED
    if not _CONFIGURED:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(
            logging.Formatter("[%(asctime)s] %(levelname)-7s %(name)s: %(message)s",
                              datefmt="%H:%M:%S")
        )
        root = logging.getLogger()
        root.handlers[:] = [handler]
        root.setLevel(logging.INFO)
        _CONFIGURED = True
    return logging.getLogger(name)

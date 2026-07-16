"""Project-wide logging setup.

Import ``get_logger`` anywhere and call it with ``__name__``. A single handler
is configured once, so repeated imports do not duplicate log lines.
"""

from __future__ import annotations

import logging
import sys

_CONFIGURED = False


def _configure_root() -> None:
    global _CONFIGURED
    if _CONFIGURED:
        return
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(
        logging.Formatter(
            fmt="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
            datefmt="%H:%M:%S",
        )
    )
    root = logging.getLogger("insight_autopilot")
    root.setLevel(logging.INFO)
    root.addHandler(handler)
    root.propagate = False
    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Return a namespaced logger under the ``insight_autopilot`` root."""
    _configure_root()
    short = name.split(".")[-1]
    return logging.getLogger(f"insight_autopilot.{short}")

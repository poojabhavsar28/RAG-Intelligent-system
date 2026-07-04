"""
Logging configuration, applied once at app startup.

The original code called logging.basicConfig(...) at import time in three
different modules (main.py, ai_assistant.py, scanning.py). basicConfig() is a
no-op after the first call in a process, so two of those three calls were
silently doing nothing — a subtle bug that made log format changes appear to
not take effect.
"""
import logging
import sys


def configure_logging(log_level: str = "INFO") -> None:
    root = logging.getLogger()
    root.setLevel(log_level)

    # Avoid duplicate handlers if configure_logging() is called more than once
    # (e.g. in tests that spin up the app repeatedly).
    if root.handlers:
        return

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s"
        )
    )
    root.addHandler(handler)

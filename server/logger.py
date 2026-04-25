"""
server/logger.py — Structured JSON logging for all environment events.
Every reset, step, query, apply, and error is logged with full context.
"""
import json
import logging
import sys
import time
from server.config import cfg


def _json_formatter(record: logging.LogRecord) -> str:
    payload = {
        "ts": round(time.time(), 3),
        "level": record.levelname,
        "msg": record.getMessage(),
    }
    if hasattr(record, "extra"):
        payload.update(record.extra)
    return json.dumps(payload)


class _JsonHandler(logging.StreamHandler):
    def emit(self, record: logging.LogRecord):
        try:
            print(_json_formatter(record), file=sys.stdout, flush=True)
        except Exception:
            self.handleError(record)


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(getattr(logging, cfg.LOG_LEVEL, logging.INFO))
        logger.addHandler(_JsonHandler())
        logger.propagate = False
    return logger


def log_event(logger: logging.Logger, event: str, **kwargs):
    """Log a structured event with arbitrary key-value context."""
    record = logging.LogRecord(
        name=logger.name, level=logging.INFO,
        pathname="", lineno=0, msg=event, args=(), exc_info=None,
    )
    record.extra = {"event": event, **kwargs}
    logger.handle(record)

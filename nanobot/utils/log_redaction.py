"""Runtime log redaction for credentials and transient session tokens."""

from __future__ import annotations

import logging
import re
from typing import Any

from loguru import logger

_INSTALLED = False
_ORIGINAL_GET_MESSAGE = logging.LogRecord.getMessage

_SENSITIVE_QUERY_KEYS = (
    "access_key",
    "ticket",
    "token",
    "secret",
    "app_secret",
    "authorization",
    "api_key",
    "apikey",
    "password",
)
_QUERY_SECRET_RE = re.compile(
    rf"([?&](?:{'|'.join(re.escape(k) for k in _SENSITIVE_QUERY_KEYS)})=)([^&\s\]]+)",
    re.IGNORECASE,
)
_HEADER_SECRET_RE = re.compile(
    r"\b(authorization|x-api-key|api-key|app-secret)\s*[:=]\s*([^\s,;]+)",
    re.IGNORECASE,
)


def redact_sensitive_log_text(value: Any) -> Any:
    """Redact secrets from a log message or argument while preserving type shape."""
    if not isinstance(value, str):
        return value
    value = _QUERY_SECRET_RE.sub(r"\1[REDACTED]", value)
    return _HEADER_SECRET_RE.sub(r"\1=[REDACTED]", value)


class RedactingFilter(logging.Filter):
    """Standard logging filter that redacts message and formatting args."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.msg = redact_sensitive_log_text(record.msg)
        if isinstance(record.args, tuple):
            record.args = tuple(redact_sensitive_log_text(arg) for arg in record.args)
        elif isinstance(record.args, dict):
            record.args = {
                key: redact_sensitive_log_text(value)
                for key, value in record.args.items()
            }
        return True


def _patch_loguru_record(record: dict[str, Any]) -> None:
    record["message"] = redact_sensitive_log_text(record.get("message", ""))


def install_log_redaction() -> None:
    """Install redaction for loguru and standard logging once per process."""
    global _INSTALLED
    if _INSTALLED:
        return
    _INSTALLED = True

    def redacted_get_message(record: logging.LogRecord) -> str:
        return str(redact_sensitive_log_text(_ORIGINAL_GET_MESSAGE(record)))

    logging.LogRecord.getMessage = redacted_get_message

    root = logging.getLogger()
    if not any(isinstance(filter_, RedactingFilter) for filter_ in root.filters):
        root.addFilter(RedactingFilter())
    for handler in root.handlers:
        if not any(isinstance(filter_, RedactingFilter) for filter_ in handler.filters):
            handler.addFilter(RedactingFilter())

    # Lark/lark-oapi may attach its own named loggers/handlers.
    for name in ("lark", "lark_oapi", "lark_oapi.ws", "lark_oapi.ws.client"):
        sdk_logger = logging.getLogger(name)
        if not any(isinstance(filter_, RedactingFilter) for filter_ in sdk_logger.filters):
            sdk_logger.addFilter(RedactingFilter())
        for handler in sdk_logger.handlers:
            if not any(isinstance(filter_, RedactingFilter) for filter_ in handler.filters):
                handler.addFilter(RedactingFilter())

    logger.configure(patcher=_patch_loguru_record)

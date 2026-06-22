import io
import logging

from nanobot.utils.log_redaction import install_log_redaction, redact_sensitive_log_text


def test_redacts_sensitive_url_query_params() -> None:
    text = (
        "connected to wss://example/ws?"
        "access_key=abc123&service_id=42&ticket=secret-ticket"
    )

    redacted = redact_sensitive_log_text(text)

    assert "access_key=[REDACTED]" in redacted
    assert "ticket=[REDACTED]" in redacted
    assert "service_id=42" in redacted
    assert "abc123" not in redacted
    assert "secret-ticket" not in redacted


def test_standard_logging_output_is_redacted() -> None:
    install_log_redaction()
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    logger = logging.getLogger("test.log_redaction")
    logger.handlers = [handler]
    logger.propagate = False
    logger.setLevel(logging.INFO)

    logger.info("url=%s", "wss://example/ws?access_key=abc&ticket=def")

    output = stream.getvalue()
    assert "access_key=[REDACTED]" in output
    assert "ticket=[REDACTED]" in output
    assert "abc" not in output
    assert "def" not in output

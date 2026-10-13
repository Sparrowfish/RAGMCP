"""Small structured tracing seam; replace the sink with OpenTelemetry if needed."""
from contextlib import contextmanager
import logging
import time
from typing import Any, Iterator

logger = logging.getLogger("conversational_mcp")


@contextmanager
def trace(operation: str, **attributes: Any) -> Iterator[None]:
    started = time.perf_counter()
    logger.info("tool_trace_started", extra={"operation": operation, **attributes})
    try:
        yield
    except Exception:
        logger.exception("tool_trace_failed", extra={"operation": operation, **attributes})
        raise
    finally:
        logger.info("tool_trace_finished", extra={"operation": operation, "duration_ms": round((time.perf_counter() - started) * 1000), **attributes})

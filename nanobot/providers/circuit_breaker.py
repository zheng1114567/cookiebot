"""Circuit breaker for LLM provider calls — prevents cascading failures."""

from __future__ import annotations

import time
from enum import Enum
from typing import Callable


class CircuitState(Enum):
    CLOSED = "closed"       # Normal operation
    OPEN = "open"           # Failing — reject fast
    HALF_OPEN = "half_open" # Testing if recovered


class CircuitBreaker:
    """State machine that trips when failures exceed a threshold.

    Protects against cascade failures by failing fast when the provider
    is known to be unhealthy, then periodically testing for recovery.
    """

    def __init__(
        self,
        name: str,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        half_open_max_calls: int = 3,
    ):
        self.name = name
        self._failure_threshold = failure_threshold
        self._recovery_timeout = recovery_timeout
        self._half_open_max_calls = half_open_max_calls

        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._last_failure_time = 0.0
        self._half_open_calls = 0
        self._success_count = 0
        self._total_calls = 0
        self._total_failures = 0

    @property
    def state(self) -> CircuitState:
        """Current state, with automatic transition from OPEN → HALF_OPEN after recovery_timeout."""
        if self._state == CircuitState.OPEN:
            if time.monotonic() - self._last_failure_time >= self._recovery_timeout:
                self._state = CircuitState.HALF_OPEN
                self._half_open_calls = 0
        return self._state

    @property
    def is_available(self) -> bool:
        """Whether a call should be attempted."""
        s = self.state
        return s != CircuitState.OPEN

    def call(self, fn: Callable, *args, **kwargs):
        """Execute *fn* with circuit breaker protection.

        Raises ``CircuitBreakerOpenError`` when the circuit is OPEN.
        """
        if not self.is_available:
            raise CircuitBreakerOpenError(
                f"Circuit breaker '{self.name}' is OPEN — rejecting call"
            )
        self._total_calls += 1
        try:
            result = fn(*args, **kwargs)
            self._on_success()
            return result
        except Exception as e:
            self._on_failure()
            raise

    async def acall(self, fn, *args, **kwargs):
        """Async version of :meth:`call`."""
        if not self.is_available:
            raise CircuitBreakerOpenError(
                f"Circuit breaker '{self.name}' is OPEN — rejecting call"
            )
        self._total_calls += 1
        try:
            result = await fn(*args, **kwargs)
            self._on_success()
            return result
        except Exception as e:
            self._on_failure()
            raise

    def _on_success(self) -> None:
        if self._state == CircuitState.HALF_OPEN:
            self._half_open_calls += 1
            self._success_count += 1
            if self._half_open_calls >= self._half_open_max_calls:
                # Stable recovery — reset to CLOSED
                self._state = CircuitState.CLOSED
                self._failure_count = 0
                self._half_open_calls = 0
                self._success_count = 0
        elif self._state == CircuitState.CLOSED:
            self._failure_count = 0  # Reset failure count on success

    def _on_failure(self) -> None:
        self._total_failures += 1
        self._last_failure_time = time.monotonic()
        if self._state == CircuitState.HALF_OPEN:
            # One failure in half-open → back to OPEN
            self._state = CircuitState.OPEN
            self._half_open_calls = 0
        elif self._state == CircuitState.CLOSED:
            self._failure_count += 1
            if self._failure_count >= self._failure_threshold:
                self._state = CircuitState.OPEN

    def reset(self) -> None:
        """Manually reset the breaker to CLOSED."""
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._half_open_calls = 0
        self._success_count = 0

    @property
    def stats(self) -> dict:
        """Return diagnostic statistics."""
        return {
            "name": self.name,
            "state": self._state.value,
            "failure_count": self._failure_count,
            "failure_threshold": self._failure_threshold,
            "total_calls": self._total_calls,
            "total_failures": self._total_failures,
            "success_rate": round(
                (self._total_calls - self._total_failures) / max(self._total_calls, 1) * 100, 1
            ),
        }


class CircuitBreakerOpenError(Exception):
    """Raised when a call is rejected because the circuit breaker is OPEN."""

import logging

from openai import (
    APIConnectionError,
    APITimeoutError,
    InternalServerError,
    RateLimitError,
)
from tenacity import (
    retry_if_exception_type,
    stop_after_attempt,
    wait_random_exponential,  # Crucial Fix: Prevents Thundering Herd problem via randomized jitter
)
from tenacity.asyncio import AsyncRetrying

logger = logging.getLogger(__name__)

# Comprehensive tuple of transient network and infrastructure anomalies safe to retry
TRANSIENT_ERRORS = (
    RateLimitError,  # 429: Too Many Requests
    APITimeoutError,  # 408: Request Timeout
    APIConnectionError,  # Network dropouts / DNS resolution failures
    InternalServerError,  # 500/502/503/504: OpenAI side infrastructure blips
)


def _before_sleep_callback(retry_state) -> None:
    """Intelligent logging callback to prevent log-chapping under high concurrency."""
    attempt = retry_state.attempt_number
    # Access max attempts from the stopper dynamically if available, else fallback to string
    max_attempts = getattr(retry_state.stop, "max_attempt_number", "?")
    exc = retry_state.outcome.exception()
    exc_name = type(exc).__name__ if exc else "UnknownError"

    log_message = (
        f"Transient AI gateway failure ({exc_name}). "
        f"Retrying execution loop: attempt {attempt}/{max_attempts}. "
        f"Backoff active, sleeping for {retry_state.idle_for:.2f}s..."
    )

    # Escalate log level if the upstream proxy continues to fail systematically
    if attempt >= 2:
        logger.warning(log_message)
    else:
        logger.info(log_message)


def retry_on_rate_limit(func):
    """Resilient async decorator driving exponential backoffs with jitter.

    Protects downstream infrastructure from synchronized storming (Thundering Herd)
    and handles global OpenAI backend dropouts gracefully within HTTP gateway bounds.
    """
    retrier = AsyncRetrying(
        retry=retry_if_exception_type(TRANSIENT_ERRORS),
        wait=wait_random_exponential(
            multiplier=1, min=1, max=15
        ),  # Safe ceiling fitting into standard 30s timeouts
        stop=stop_after_attempt(3),  # 3 attempts balance latency budgets perfectly
        before_sleep=_before_sleep_callback,
        reraise=True,
    )
    return retrier["wrapper"](func)

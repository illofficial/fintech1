import logging

from openai import RateLimitError
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

_retry_logger = logging.getLogger("app.services.retry")

# Shared backoff policy for transient OpenAI rate limiting. Defined once so every
# service applies identical, well-tested retry semantics.
retry_on_rate_limit = retry(
    retry=retry_if_exception_type(RateLimitError),
    wait=wait_exponential(multiplier=1, min=1, max=60),
    stop=stop_after_attempt(5),
    before_sleep=before_sleep_log(_retry_logger, logging.WARNING),
    reraise=True,
)

"""Rate limiting middleware to prevent API abuse."""
import time
import logging
from collections import defaultdict
from typing import Dict, Tuple, Optional
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from app.utils.exceptions import RateLimitError
from app.config import settings

logger = logging.getLogger(__name__)


class RateLimiter:
    """
    Token bucket rate limiter implementation.

    Supports per-IP and per-API-key rate limiting with configurable
    limits and time windows.
    """

    def __init__(self):
        """Initialize rate limiter with storage dictionaries."""
        # Storage: {identifier: (tokens, last_update_time)}
        self.buckets: Dict[str, Tuple[float, float]] = defaultdict(
            lambda: (0.0, time.time())
        )
        # Track total requests per identifier
        self.request_counts: Dict[str, int] = defaultdict(int)

    def _get_tokens(
        self, identifier: str, max_tokens: float, refill_rate: float
    ) -> float:
        """
        Get current token count for identifier.

        Args:
            identifier: Unique identifier (IP or API key)
            max_tokens: Maximum tokens in bucket
            refill_rate: Tokens added per second

        Returns:
            Current token count
        """
        tokens, last_update = self.buckets[identifier]
        now = time.time()
        time_passed = now - last_update

        # Refill tokens based on time passed
        tokens = min(max_tokens, tokens + time_passed * refill_rate)

        self.buckets[identifier] = (tokens, now)
        return tokens

    def _consume_token(self, identifier: str) -> None:
        """
        Consume one token from the bucket.

        Args:
            identifier: Unique identifier
        """
        tokens, last_update = self.buckets[identifier]
        self.buckets[identifier] = (tokens - 1, last_update)
        self.request_counts[identifier] += 1

    def check_rate_limit(
        self,
        identifier: str,
        max_requests: int = 100,
        window_seconds: int = 60,
    ) -> Tuple[bool, Optional[int]]:
        """
        Check if request should be allowed.

        Args:
            identifier: Unique identifier (IP or API key)
            max_requests: Maximum requests allowed in window
            window_seconds: Time window in seconds

        Returns:
            Tuple of (is_allowed, retry_after_seconds)
        """
        refill_rate = max_requests / window_seconds
        tokens = self._get_tokens(identifier, max_requests, refill_rate)

        if tokens >= 1:
            self._consume_token(identifier)
            return True, None
        else:
            # Calculate retry after time
            retry_after = int((1 - tokens) / refill_rate) + 1
            return False, retry_after

    def get_stats(self, identifier: str) -> Dict[str, int]:
        """
        Get rate limit statistics for identifier.

        Args:
            identifier: Unique identifier

        Returns:
            Dictionary with statistics
        """
        tokens, _ = self.buckets.get(identifier, (0.0, time.time()))
        return {
            "total_requests": self.request_counts.get(identifier, 0),
            "remaining_tokens": int(tokens),
        }

    def cleanup_old_entries(self, max_age_seconds: int = 3600):
        """
        Remove old entries from storage to prevent memory bloat.

        Args:
            max_age_seconds: Maximum age of entries to keep
        """
        now = time.time()
        old_identifiers = [
            identifier
            for identifier, (_, last_update) in self.buckets.items()
            if now - last_update > max_age_seconds
        ]

        for identifier in old_identifiers:
            del self.buckets[identifier]
            if identifier in self.request_counts:
                del self.request_counts[identifier]

        if old_identifiers:
            logger.info(f"Cleaned up {len(old_identifiers)} old rate limit entries")


# Global rate limiter instance
_rate_limiter = RateLimiter()


def get_rate_limiter() -> RateLimiter:
    """
    Get the global rate limiter instance.

    Returns:
        RateLimiter instance
    """
    return _rate_limiter


class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    Middleware to apply rate limiting to API requests.

    Supports different rate limits for different endpoint types.
    """

    def __init__(self, app, limiter: Optional[RateLimiter] = None):
        """
        Initialize rate limit middleware.

        Args:
            app: FastAPI application
            limiter: RateLimiter instance (uses global if not provided)
        """
        super().__init__(app)
        self.limiter = limiter or get_rate_limiter()

        # Define rate limits for different endpoint types
        self.rate_limits = {
            # Authentication endpoints: 5 requests per minute
            "auth": {"max_requests": 5, "window_seconds": 60},
            # Chat/streaming endpoints: 20 requests per minute
            "chat": {"max_requests": 20, "window_seconds": 60},
            # General API: 100 requests per minute
            "general": {"max_requests": 100, "window_seconds": 60},
            # Health/info endpoints: 300 requests per minute
            "info": {"max_requests": 300, "window_seconds": 60},
        }

    def _get_rate_limit_type(self, path: str) -> str:
        """
        Determine rate limit type based on endpoint path.

        Args:
            path: Request path

        Returns:
            Rate limit type
        """
        if path.startswith("/api/auth"):
            return "auth"
        elif path.startswith("/api/chat") or path.startswith("/api/stream"):
            return "chat"
        elif path in ["/health", "/", "/docs", "/openapi.json"]:
            return "info"
        else:
            return "general"

    def _get_identifier(self, request: Request) -> str:
        """
        Get unique identifier for rate limiting.

        Prefers API key over IP address for authenticated requests.

        Args:
            request: FastAPI request

        Returns:
            Unique identifier
        """
        # Try to get API key from Authorization header
        auth_header = request.headers.get("authorization", "")
        if auth_header.startswith("Bearer "):
            api_key = auth_header[7:]
            # Use hash of API key as identifier
            import hashlib
            return f"key:{hashlib.sha256(api_key.encode()).hexdigest()[:16]}"

        # Fallback to IP address
        # Check for X-Forwarded-For header (for proxied requests)
        forwarded_for = request.headers.get("x-forwarded-for")
        if forwarded_for:
            # Take the first IP in the chain
            client_ip = forwarded_for.split(",")[0].strip()
        else:
            client_ip = request.client.host if request.client else "unknown"

        return f"ip:{client_ip}"

    async def dispatch(self, request: Request, call_next):
        """
        Process request with rate limiting.

        Args:
            request: FastAPI request
            call_next: Next middleware in chain

        Returns:
            Response or rate limit error
        """
        # Get identifier and rate limit type
        identifier = self._get_identifier(request)
        rate_limit_type = self._get_rate_limit_type(request.url.path)
        limits = self.rate_limits[rate_limit_type]

        # Check rate limit
        allowed, retry_after = self.limiter.check_rate_limit(
            identifier,
            max_requests=limits["max_requests"],
            window_seconds=limits["window_seconds"],
        )

        if not allowed:
            logger.warning(
                f"Rate limit exceeded for {identifier} on {request.url.path}",
                extra={
                    "identifier": identifier,
                    "path": request.url.path,
                    "rate_limit_type": rate_limit_type,
                },
            )

            # Raise rate limit error
            raise RateLimitError(
                message=f"Rate limit exceeded. Please try again in {retry_after} seconds.",
                retry_after=retry_after,
                details={
                    "limit_type": rate_limit_type,
                    "max_requests": limits["max_requests"],
                    "window_seconds": limits["window_seconds"],
                },
            )

        # Process request
        response = await call_next(request)

        # Add rate limit headers
        stats = self.limiter.get_stats(identifier)
        response.headers["X-RateLimit-Limit"] = str(limits["max_requests"])
        response.headers["X-RateLimit-Remaining"] = str(stats["remaining_tokens"])
        response.headers["X-RateLimit-Reset"] = str(limits["window_seconds"])

        return response


async def cleanup_rate_limiter():
    """Periodic cleanup task for rate limiter."""
    limiter = get_rate_limiter()
    limiter.cleanup_old_entries()

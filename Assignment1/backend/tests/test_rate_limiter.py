"""Tests for rate limiting functionality."""
import pytest
import time
from app.middleware.rate_limiter import RateLimiter


class TestRateLimiter:
    """Tests for RateLimiter class."""

    def test_basic_rate_limiting(self):
        """Test basic rate limiting functionality."""
        limiter = RateLimiter()

        # First request should be allowed
        allowed, retry_after = limiter.check_rate_limit("user1", max_requests=2, window_seconds=60)
        assert allowed is True
        assert retry_after is None

        # Second request should be allowed
        allowed, retry_after = limiter.check_rate_limit("user1", max_requests=2, window_seconds=60)
        assert allowed is True
        assert retry_after is None

        # Third request should be denied
        allowed, retry_after = limiter.check_rate_limit("user1", max_requests=2, window_seconds=60)
        assert allowed is False
        assert retry_after is not None
        assert retry_after > 0

    def test_different_identifiers(self):
        """Test that different identifiers have separate limits."""
        limiter = RateLimiter()

        # User1 makes requests
        limiter.check_rate_limit("user1", max_requests=1, window_seconds=60)

        # User2 should still be allowed
        allowed, _ = limiter.check_rate_limit("user2", max_requests=1, window_seconds=60)
        assert allowed is True

    def test_token_refill(self):
        """Test that tokens refill over time."""
        limiter = RateLimiter()

        # Consume all tokens
        limiter.check_rate_limit("user1", max_requests=1, window_seconds=1)

        # Wait for tokens to refill
        time.sleep(1.1)

        # Should be allowed again
        allowed, _ = limiter.check_rate_limit("user1", max_requests=1, window_seconds=1)
        assert allowed is True

    def test_get_stats(self):
        """Test getting rate limit statistics."""
        limiter = RateLimiter()

        # Make some requests
        limiter.check_rate_limit("user1", max_requests=10, window_seconds=60)
        limiter.check_rate_limit("user1", max_requests=10, window_seconds=60)

        stats = limiter.get_stats("user1")
        assert stats["total_requests"] == 2
        assert "remaining_tokens" in stats

    def test_cleanup_old_entries(self):
        """Test cleanup of old entries."""
        limiter = RateLimiter()

        # Add an entry
        limiter.check_rate_limit("user1", max_requests=10, window_seconds=60)

        # Manually set old timestamp
        limiter.buckets["user1"] = (10.0, time.time() - 7200)  # 2 hours ago

        # Cleanup should remove it
        deleted = limiter.cleanup_old_entries(max_age_seconds=3600)
        assert deleted > 0
        assert "user1" not in limiter.buckets

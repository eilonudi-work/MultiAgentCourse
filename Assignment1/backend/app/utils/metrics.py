"""Performance metrics and monitoring utilities."""
import time
import logging
from collections import defaultdict
from typing import Dict, Optional
from datetime import datetime, timedelta
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger(__name__)


class MetricsCollector:
    """
    Collect and store application performance metrics.

    Tracks request counts, response times, error rates, and other metrics.
    """

    def __init__(self):
        """Initialize metrics collector."""
        # Request metrics
        self.request_count: Dict[str, int] = defaultdict(int)
        self.error_count: Dict[str, int] = defaultdict(int)
        self.response_times: Dict[str, list] = defaultdict(list)

        # System metrics
        self.startup_time = datetime.utcnow()
        self.total_requests = 0
        self.total_errors = 0

        # Endpoint-specific metrics
        self.endpoint_metrics: Dict[str, Dict] = defaultdict(
            lambda: {
                "count": 0,
                "errors": 0,
                "total_time": 0.0,
                "min_time": float("inf"),
                "max_time": 0.0,
            }
        )

    def record_request(
        self,
        method: str,
        path: str,
        status_code: int,
        response_time: float,
    ) -> None:
        """
        Record request metrics.

        Args:
            method: HTTP method
            path: Request path
            status_code: HTTP status code
            response_time: Response time in seconds
        """
        endpoint = f"{method} {path}"

        # Update counters
        self.request_count[endpoint] += 1
        self.total_requests += 1

        if status_code >= 400:
            self.error_count[endpoint] += 1
            self.total_errors += 1

        # Update response times
        self.response_times[endpoint].append(response_time)

        # Keep only last 1000 response times per endpoint
        if len(self.response_times[endpoint]) > 1000:
            self.response_times[endpoint] = self.response_times[endpoint][-1000:]

        # Update endpoint metrics
        metrics = self.endpoint_metrics[endpoint]
        metrics["count"] += 1
        if status_code >= 400:
            metrics["errors"] += 1
        metrics["total_time"] += response_time
        metrics["min_time"] = min(metrics["min_time"], response_time)
        metrics["max_time"] = max(metrics["max_time"], response_time)

    def get_endpoint_stats(self, endpoint: str) -> Dict:
        """
        Get statistics for a specific endpoint.

        Args:
            endpoint: Endpoint identifier

        Returns:
            Dictionary with endpoint statistics
        """
        metrics = self.endpoint_metrics.get(endpoint)
        if not metrics or metrics["count"] == 0:
            return {}

        response_times = self.response_times.get(endpoint, [])

        stats = {
            "request_count": metrics["count"],
            "error_count": metrics["errors"],
            "error_rate": metrics["errors"] / metrics["count"] if metrics["count"] > 0 else 0,
            "avg_response_time": metrics["total_time"] / metrics["count"],
            "min_response_time": metrics["min_time"],
            "max_response_time": metrics["max_time"],
        }

        # Calculate percentiles if we have response times
        if response_times:
            sorted_times = sorted(response_times)
            stats["p50_response_time"] = self._percentile(sorted_times, 0.50)
            stats["p95_response_time"] = self._percentile(sorted_times, 0.95)
            stats["p99_response_time"] = self._percentile(sorted_times, 0.99)

        return stats

    def _percentile(self, sorted_list: list, percentile: float) -> float:
        """
        Calculate percentile value from sorted list.

        Args:
            sorted_list: Sorted list of values
            percentile: Percentile to calculate (0.0 to 1.0)

        Returns:
            Percentile value
        """
        if not sorted_list:
            return 0.0

        k = (len(sorted_list) - 1) * percentile
        f = int(k)
        c = f + 1

        if c >= len(sorted_list):
            return sorted_list[-1]

        d0 = sorted_list[f] * (c - k)
        d1 = sorted_list[c] * (k - f)
        return d0 + d1

    def get_summary(self) -> Dict:
        """
        Get overall application metrics summary.

        Returns:
            Dictionary with summary metrics
        """
        uptime = datetime.utcnow() - self.startup_time
        uptime_seconds = uptime.total_seconds()

        # Calculate overall statistics
        all_response_times = []
        for times in self.response_times.values():
            all_response_times.extend(times)

        summary = {
            "uptime_seconds": uptime_seconds,
            "uptime_formatted": str(uptime).split(".")[0],  # Remove microseconds
            "total_requests": self.total_requests,
            "total_errors": self.total_errors,
            "error_rate": self.total_errors / self.total_requests if self.total_requests > 0 else 0,
            "requests_per_second": self.total_requests / uptime_seconds if uptime_seconds > 0 else 0,
        }

        # Add overall response time statistics
        if all_response_times:
            sorted_times = sorted(all_response_times)
            summary["avg_response_time"] = sum(sorted_times) / len(sorted_times)
            summary["p50_response_time"] = self._percentile(sorted_times, 0.50)
            summary["p95_response_time"] = self._percentile(sorted_times, 0.95)
            summary["p99_response_time"] = self._percentile(sorted_times, 0.99)

        return summary

    def get_all_endpoints(self) -> Dict:
        """
        Get metrics for all endpoints.

        Returns:
            Dictionary mapping endpoints to their statistics
        """
        return {
            endpoint: self.get_endpoint_stats(endpoint)
            for endpoint in self.endpoint_metrics.keys()
        }

    def reset(self) -> None:
        """Reset all metrics."""
        self.request_count.clear()
        self.error_count.clear()
        self.response_times.clear()
        self.endpoint_metrics.clear()
        self.startup_time = datetime.utcnow()
        self.total_requests = 0
        self.total_errors = 0
        logger.info("Metrics reset")


# Global metrics collector
_metrics_collector: Optional[MetricsCollector] = None


def get_metrics_collector() -> MetricsCollector:
    """
    Get the global metrics collector instance.

    Returns:
        MetricsCollector instance
    """
    global _metrics_collector
    if _metrics_collector is None:
        _metrics_collector = MetricsCollector()
    return _metrics_collector


class MetricsMiddleware(BaseHTTPMiddleware):
    """
    Middleware to collect request/response metrics.

    Records response times, status codes, and other metrics for monitoring.
    """

    def __init__(self, app, collector: Optional[MetricsCollector] = None):
        """
        Initialize metrics middleware.

        Args:
            app: FastAPI application
            collector: MetricsCollector instance (uses global if not provided)
        """
        super().__init__(app)
        self.collector = collector or get_metrics_collector()

    async def dispatch(self, request: Request, call_next):
        """
        Process request and collect metrics.

        Args:
            request: FastAPI request
            call_next: Next middleware in chain

        Returns:
            Response with timing headers
        """
        # Record start time
        start_time = time.time()

        # Process request
        try:
            response = await call_next(request)
            status_code = response.status_code
        except Exception as e:
            # Record error
            response_time = time.time() - start_time
            self.collector.record_request(
                request.method,
                request.url.path,
                500,
                response_time,
            )
            raise

        # Calculate response time
        response_time = time.time() - start_time

        # Record metrics
        self.collector.record_request(
            request.method,
            request.url.path,
            status_code,
            response_time,
        )

        # Add response time header
        response.headers["X-Response-Time"] = f"{response_time:.4f}s"

        # Log slow requests (> 1 second)
        if response_time > 1.0:
            logger.warning(
                f"Slow request: {request.method} {request.url.path} took {response_time:.2f}s",
                extra={
                    "method": request.method,
                    "path": request.url.path,
                    "response_time": response_time,
                    "status_code": status_code,
                },
            )

        return response

"""
Latency Logging Middleware
Measures and logs HTTP request latency for all endpoints
"""

import logging
import time
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request

logger = logging.getLogger(__name__)


class LatencyLoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        start = time.time()
        response = await call_next(request)
        duration_ms = (time.time() - start) * 1000

        logger.info(
            f"HTTP {request.method} {request.url.path} "
            f"→ {response.status_code} | {duration_ms:.1f}ms"
        )

        response.headers["X-Response-Time-Ms"] = str(round(duration_ms, 2))
        return response

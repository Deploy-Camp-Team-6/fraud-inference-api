from uuid import uuid4

import structlog
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseFunction
from starlette.requests import Request
from starlette.responses import Response

REQUEST_ID_HEADER = "X-Request-ID"

class RequestIDMiddleware(BaseHTTPMiddleware):
    """
    Middleware to add a unique request ID to each incoming request.
    - If X-Request-ID header is present, it will be used.
    - Otherwise, a new UUID will be generated.
    - The ID is added to the structlog context and the response headers.
    """
    async def dispatch(
        self, request: Request, call_next: RequestResponseFunction
    ) -> Response:
        # Get request ID from headers or generate a new one
        request_id = request.headers.get(REQUEST_ID_HEADER, str(uuid4()))

        # Bind the request ID to the context for logging
        structlog.contextvars.clear_contextvars()
        structlog.contextvars.bind_contextvars(request_id=request_id)

        # Process the request
        response = await call_next(request)

        # Add request ID to response headers
        response.headers[REQUEST_ID_HEADER] = request_id

        return response

from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import Response, JSONResponse
from starlette.status import HTTP_413_REQUEST_ENTITY_TOO_LARGE
from app.config import settings
from app.utils.logger import logger

class LimitUploadSizeMiddleware(BaseHTTPMiddleware):
    """
    Middleware to limit the maximum size of incoming requests (especially file uploads)
    based on settings.MAX_FILE_SIZE.
    """
    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        if request.method == "POST":
            content_length = request.headers.get("content-length")
            if content_length:
                try:
                    size = int(content_length)
                    if size > settings.MAX_FILE_SIZE:
                        max_mb = settings.MAX_FILE_SIZE / (1024 * 1024)
                        logger.warning(
                            f"Upload block: request content-length ({size} bytes) "
                            f"exceeds limit ({settings.MAX_FILE_SIZE} bytes)."
                        )
                        return JSONResponse(
                            status_code=HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                            content={
                                "success": False,
                                "error": f"Upload size exceeds maximum allowed limit of {max_mb} MB."
                            }
                        )
                except ValueError:
                    pass

        return await call_next(request)

import secrets

from fastapi import HTTPException, Security
from fastapi.security.api_key import APIKeyHeader
from starlette.status import HTTP_403_FORBIDDEN

from app.core.config import settings

API_KEY_HEADER_NAME = "X-API-Key"
api_key_header = APIKeyHeader(name=API_KEY_HEADER_NAME, auto_error=False)


def get_api_key(key: str = Security(api_key_header)) -> str:
    """
    Dependency that requires and validates an API key.
    Raises HTTPException if the key is missing or invalid.
    """
    if not key:
        raise HTTPException(status_code=HTTP_403_FORBIDDEN, detail="API key is missing")

    # In a real application, you might check against a database or a secrets manager.
    # Here, we check against a list from environment variables.
    # Using secrets.compare_digest helps prevent timing attacks.
    is_valid = any(
        secrets.compare_digest(key, valid_key) for valid_key in settings.API_KEYS
    )

    if not is_valid:
        raise HTTPException(status_code=HTTP_403_FORBIDDEN, detail="Invalid API key")

    return key

from fastapi import Depends, HTTPException, status
from fastapi.security import APIKeyHeader

EXPECTED_API_KEY = "kairos-secret"
api_key_header = APIKeyHeader(name="X-API-key")


def get_current_user(api_key: str = Depends(api_key_header)) -> str:
    if api_key != EXPECTED_API_KEY:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API key")
    return "kairos-user"
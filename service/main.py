import time

from fastapi import FastAPI, Header, Query, Path, Depends, HTTPException, status
from fastapi.security import APIKeyHeader
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from service.exceptions import PairNotFoundError, ResultsFileNotFoundError
from .services.chat_service import generate_chat_answer
from .services.pairs_service import get_pair_result as lookup_pair_result
import logging
from analysis.logging_config import setup_logging

setup_logging()
logger = logging.getLogger(__name__)

EXPECTED_API_KEY = "kairos-secret"

api_key_header = APIKeyHeader(name="X-API-key")

app = FastAPI()

@app.middleware("http")
async def log_requests(request, call_next):
    start = time.time()
    response = await call_next(request)
    duration_ms = (time.time() - start) * 1000
    logger.info(
        "%s %s -> %d (%.1fms)",
        request.method, request.url.path, response.status_code, duration_ms,
    )
    return response

@app.exception_handler(PairNotFoundError)
def pair_not_found_handler(request, exc: PairNotFoundError):
    logger.warning("Pair not found %s", exc.pair_name)
    return JSONResponse(
        status_code = 404,
        content={"error": str(exc), "pair_name": exc.pair_name},
    )

@app.exception_handler(ResultsFileNotFoundError)
def results_file_not_found_handler(request, exc: ResultsFileNotFoundError):
    logger.warning("%s Results file not found.", exc.source)
    return JSONResponse(
        status_code=503,
        content={"error": str(exc)},
    )

class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=500)
    top_k: int = Field(default=5, ge=1, le=20)
    static: bool = Field(default=True)

class ChatResponse(BaseModel):
    answer: str

def get_current_user(api_key: str = Depends(api_key_header)) -> str:
    if api_key != EXPECTED_API_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key"
        )
    return "kairos-user"

@app.get("/")
def read_root():
    return {"status": "ok"}

@app.post("/api/chat", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    current_user: str = Depends(get_current_user),
) -> ChatResponse:
    answer = await generate_chat_answer(current_user, request.message, request.static)
    return ChatResponse(answer=answer)

@app.get("/api/pairs/{pair_name}")
def get_pair_results(
    pair_name: str = Path(min_length=3, max_length=20),
    fold: int | None = Query(default=None, ge=1, le=15),
    static: bool = Query(default=True),
    x_api_key: str | None = Header(default=None, min_length=2),
):
    result = lookup_pair_result(pair_name, fold, static)
    return {**result, "x_api_key": x_api_key}
import time
import logging

from fastapi import FastAPI
from fastapi.responses import JSONResponse

from service.exceptions import PairNotFoundError, ResultsFileNotFoundError
from service.services.chat_service import generate_chat_answer
from service.routers import chat, pairs

from analysis.logging_config import setup_logging

setup_logging()
logger = logging.getLogger(__name__)

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

app.include_router(chat.router)

app.include_router(pairs.router)

@app.get("/")
def read_root():
    return {"status": "ok"}


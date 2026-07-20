from fastapi import APIRouter, Path, Query, Header
from service.services.pairs_service import get_pair_result as lookup_pair_result

router = APIRouter(
    prefix="/api/pairs",
    tags=["pairs"]
)

@router.get("/{pair_name}")
def get_pair_results(
    pair_name: str = Path(min_length=3, max_length=20),
    fold: int | None = Query(default=None, ge=1, le=15),
    static: bool = Query(default=True),
    x_api_key: str | None = Header(default=None, min_length=2),
):
    result = lookup_pair_result(pair_name, fold, static)
    return {**result, "x_api_key": x_api_key}
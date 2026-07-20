from pathlib import Path
import pandas as pd

from service.exceptions import PairNotFoundError, ResultsFileNotFoundError

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"
STATIC_FOLDS_CSV = DATA_DIR / "static_walk_forward_folds.csv"
KALMAN_FOLDS_CSV = DATA_DIR / "kalman_walk_forward_folds.csv"

def get_pair_result(pair_name: str, fold: int | None, static: bool = True) -> dict:
    csv_path = STATIC_FOLDS_CSV if static else KALMAN_FOLDS_CSV
    source = "static" if static else "kalman"
    
    if not csv_path.exists():
        raise ResultsFileNotFoundError(source)
    
    df = pd.read_csv(csv_path)
    matches = df[df["pair_name"] == pair_name]

    if fold is not None:
        matches = matches[matches["fold"] == fold]

    if matches.empty:
        raise PairNotFoundError(pair_name)

    return {
        "pair_name": pair_name,
        "fold": fold,
        "results": matches.to_dict(orient="records"),
    }

from pathlib import Path
import re
import pandas as pd
import asyncio

from service.exceptions import PairNotFoundError, ResultsFileNotFoundError

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"
STATIC_FOLDS_CSV = DATA_DIR / "static_walk_forward_folds.csv"
KALMAN_FOLDS_CSV = DATA_DIR / "kalman_walk_forward_folds.csv"

def _search_pair_mentions(message: str, known_pairs: list[str]) -> list[str]:
    """
    Find pair names mantioned in the message.
    """
    return [pair for pair in known_pairs if pair in message]


async def generate_chat_answer(user: str, message: str, static: bool = True) -> str:
    csv_path = STATIC_FOLDS_CSV if static else KALMAN_FOLDS_CSV

    if not csv_path.exists():
        raise ResultsFileNotFoundError("static" if static else "kalman")

    df = pd.read_csv(csv_path)
    known_pairs = df["pair_name"].unique().tolist()

    mentioned = _search_pair_mentions(message, known_pairs)
    if not mentioned:
        raise PairNotFoundError(message)

    pair = mentioned[0]
    context = await fetch_external_context(pair)

    pair_rows = df[df["pair_name"] == pair]
    total_trades = int(pair_rows["trade_count"].sum())
    avg_return = pair_rows["total_return"].mean()

    return (
        f"{user} asked about {pair}: across {len(pair_rows)} folds, "
        f"{total_trades} trades were taken, averaging a "
        f"{avg_return * 100:.2f}% return per fold. {context}"
    )

async def fetch_external_context(pair: str) -> str:
    """
    Simulated external service call (stand-in for a real Azure/API call
    await asyncio.sleep mimics network latency.
    """
    await asyncio.sleep(1)
    return f"[external context placeholder for {pair}]"

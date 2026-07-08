import os
import logging
from datetime import datetime, timedelta
from pathlib import Path

from dotenv import load_dotenv

from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

from logging_config import setup_logging

load_dotenv()

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"

DATA_DIR.mkdir(exist_ok=True)

end_date = datetime.now()
start_date = end_date - timedelta(days=2 * 365)


def create_client():
    api_key = os.getenv("ALPACA_API_KEY")
    secret_key = os.getenv("ALPACA_SECRET_KEY")

    if not api_key or not secret_key:
        raise RuntimeError("Missing Alpaca credentials in .env")

    return StockHistoricalDataClient(api_key, secret_key)


def build_request():

    end_date = datetime.now()
    start_date = end_date - timedelta(days=2 * 365)

    return StockBarsRequest(
        symbol_or_symbols=[
            # Energy Stocks
            # Traditional blue-chip competitors in US energy space
            "XOM",  # Exxon Mobil
            "CVX",  # Chevron
            # London-listed giants, 2 largest European operators
            "SHEL",  # Shell
            "BP",  # BP
            # 2 Euro operators heavy concentrated in international gas pricing and aggressive renerable transition
            "TTE",  # TotalEnergies
            "EQNR",  # Equinor
            # Highly liquid, US-focused operators
            # highly sensitive to WTI (West Texas Intermediate) pricing dynamics
            "DVN",  # Devon Energy
            "FANG",  # Diamondback Energy
            # top indepndent U.S. refiners
            # Great for testing crack spreads (diff between crude oil prices and the refined products)
            "VLO",  # Valero Energy
            "MPC",  # Marathon Petroleum
            # Food/Beverage Stocks
            # Classic Pair. Historically co-integrated pricing relationships
            "KO",  # Coca-Cola
            "PEP",  # PepsiCo
            # Protein-feed pair
            # Tyson's livestock margins and Bunge's grain/soybean crush margins
            # share tight historical economic equulibrium
            "TSN",  # Tyson Foods
            "BG",  # Bunge Global
            # Two major packaged food manufactures
            # freq exhibit mean-reverting price spreads
            # driven by agricultural commodity costs (e.g., wheat, dairy)
            "KHC",  # Kraft Heinz
            "GIS",  # General Mills
            # Snack/candy pairing subject to same coco, sugar, and logistic input costs
            "MDLZ",  # Mondelez International
            "HSY",  # Hershey
            # Pairs-tested to track wholesale vs. retail margin divergences
            "WMT",  # Walmart
            "COST",  # Costco
        ],
        timeframe=TimeFrame.Day,
        start=start_date,
        end=end_date,
    )


def main():
    setup_logging()
    client = create_client()
    request = build_request()

    logger.info(
        "Downloading daily bars for %d symbols.", len(request.symbol_or_symbols)
    )

    bars_df = client.get_stock_bars(request).df
    bars_df.to_parquet(DATA_DIR / "bars.parquet", index=True)

    logger.info("Saved %d rows to %s.", len(bars_df), DATA_DIR)


if __name__ == "__main__":
    main()

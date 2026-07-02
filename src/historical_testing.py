from dotenv import load_dotenv
import os
from datetime import datetime, timedelta
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
import pandas as pd

load_dotenv()

ALPACA_API_KEY = os.getenv("ALPACA_API_KEY")
ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")
client = StockHistoricalDataClient(ALPACA_API_KEY, ALPACA_SECRET_KEY)

end_date = datetime.now()
start_date = end_date - timedelta(days=2 * 365)

request_params = StockBarsRequest(
    symbol_or_symbols=[

        # Energy Stocks

        # Traditional blue-chip competitors in US energy space
        "XOM", # Exxon Mobil
        "CVX", # Chevron

        # London-listed giants, 2 largest European operators
        "SHEL", #Shell
        "BP", #BP

        # 2 Euro operators heavy concentrated in international gas pricing and aggressive renerable transition
        "TTE", # TotalEnergies
        "EQDRY", # Equinor

        # Highly liquid, US-focused operators 
        # highly sensitive to WTI (West Texas Intermediate) pricing dynamics
        "DVN", # Devon Energy
        "FANG", # Diamondback Energy

        # top indepndent U.S. refiners
        # Great for testing crack spreads (diff between crude oil prices and the refined products)
        "VLO", # Valero Energy
        "MPC", # Marathon Petroleum

        # Food/Beverage Stocks

        # Classic Pair. Historically co-integrated pricing relationships
        "KO", # Coca-Cola
        "PEP", # PepsiCo

        # Protein-feed pair
        # Tyson's livestock margins and Bunge's grain/soybean crush margins
        # share tight historical economic equulibrium
        "TSN", # Tyson Foods
        "BG", # Bunge Global

        # Two major packaged food manufactures
        # freq exhibit mean-reverting price spreads 
        # driven by agricultural commodity costs (e.g., wheat, dairy)
        "KHC", # Kraft Heinz
        "GIS", # General Mills

        # Snack/candy pairing subject to same coco, sugar, and logistic input costs 
        "MDLZ", # Mondelez International
        "HSY", # Hershey

        # Pairs-tested to track wholesale vs. retail margin divergences
        "WMT", # Walmart
        "COST" # Costco
    ],
    timeframe=TimeFrame.Day,
    start=start_date,
    end=end_date
)

bars_df = client.get_stock_bars(request_params).df
print(bars_df.head())




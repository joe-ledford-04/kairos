from pathlib import Path

import pandas as pd  
from statsmodels.tsa.stattools import adfuller

project_root = Path(__file__).resolve().parent.parent
data_dir = project_root / "Data"

bars_df = pd.read_parquet(data_dir / "bars.parquet")

# Augmented Dicky-Fuller Test
adf_df = bars_df.reset_index()[["symbol", "timestamp", "close"]]

for symbol, group in adf_df.groupby("symbol"):
    closes = group["close"]

    stat, pvalue, *_ = adfuller(closes)

    print(f"{symbol}: p-value = {pvalue:.4f}")

# All non-stationary


# Hedge Ratio Testing
# exxon_df = adf_df.loc[symbol]["XOM"]
# print(exxon_df.head())
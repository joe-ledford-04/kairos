from pathlib import Path

import pandas as pd  
from statsmodels.tsa.stattools import adfuller
import statsmodels.api as sm

project_root = Path(__file__).resolve().parent.parent
data_dir = project_root / "Data"

bars_df = pd.read_parquet(data_dir / "bars.parquet")

# ------ Augmented Dicky-Fuller Test -------
adf_df = bars_df.reset_index()[["symbol", "timestamp", "close"]]

for symbol, group in adf_df.groupby("symbol"):
    closes = group["close"]

    stat, pvalue, *_ = adfuller(closes)

    print(f"{symbol}: p-value = {pvalue:.4f}")

# All non-stationary


# ------ Hedge Ratio Testing ------
stock_data = {}
for symbol, group in adf_df.groupby("symbol"):
    stock_data[symbol] = group

exxon_df = stock_data["XOM"].reset_index(drop=True)
chevron_df = stock_data["CVX"].reset_index(drop=True)

x = exxon_df["close"]
y = chevron_df["close"]

x_with_constant = sm.add_constant(x)

model = sm.OLS(y, x_with_constant)
results = model.fit()

hedge_ratio = results.params["close"]

print(results.summary())
print(hedge_ratio)


    
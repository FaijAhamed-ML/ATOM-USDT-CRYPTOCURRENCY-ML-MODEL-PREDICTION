import requests
import pandas as pd

url = "https://binance.com"
params = {"symbol": "ATOMUSDT", "interval": "15m", "limit": 500}
data = requests.get(url, params=params).json()

df = pd.DataFrame(
    data,
    columns=[
        "open_time",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "close_time",
        "quote_asset_volume",
        "trades",
        "taker_buy_base",
        "taker_buy_quote",
        "ignore",
    ],
)
df["close"] = df["close"].astype(float)
print(df[["open_time", "close"]].tail())

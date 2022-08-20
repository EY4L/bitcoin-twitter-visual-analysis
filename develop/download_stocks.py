import yfinance as yf

# Read Bitcoin USD price from yahoo
bitcoin = yf.Ticker("BTC-USD")

# Subset on date range to match twitter data
bitcoin_history = bitcoin.history(
    start="2021-02-07",
    end="2021-12-04",
    interval="1d",
    actions=False,
)

# Save as CSV
bitcoin_history.to_csv(
    "bitcoin_history.csv",
    sep=",",
    encoding="utf-8",
)

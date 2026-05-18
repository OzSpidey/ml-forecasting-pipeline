import yfinance as yf
import pandas as pd


def fetch(ticker: str, period: str = "3y") -> pd.DataFrame:
    df = yf.download(ticker, period=period, auto_adjust=True, progress=False)
    # yfinance returns MultiIndex columns for single-ticker downloads in newer versions
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0].lower() for c in df.columns]
    else:
        df.columns = [c.lower() for c in df.columns]
    df.index.name = "date"
    return df.dropna()

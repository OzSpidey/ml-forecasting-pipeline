import pandas as pd
import numpy as np

FEATURE_COLS = [
    "return_1d", "return_5d", "return_10d",
    "ma_7", "ma_21", "ma_50",
    "macd", "macd_signal", "macd_hist",
    "rsi_14",
    "bb_width", "bb_position",
    "volume_ratio",
    "volatility_10d", "volatility_30d",
    "close_lag_1", "close_lag_2", "close_lag_3", "close_lag_5", "close_lag_10",
    "high_lag_1", "low_lag_1",
    "day_of_week", "month",
]


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["return_1d"] = df["close"].pct_change()
    df["return_5d"] = df["close"].pct_change(5)
    df["return_10d"] = df["close"].pct_change(10)

    df["ma_7"] = df["close"].rolling(7).mean()
    df["ma_21"] = df["close"].rolling(21).mean()
    df["ma_50"] = df["close"].rolling(50).mean()

    ema12 = df["close"].ewm(span=12).mean()
    ema26 = df["close"].ewm(span=26).mean()
    df["macd"] = ema12 - ema26
    df["macd_signal"] = df["macd"].ewm(span=9).mean()
    df["macd_hist"] = df["macd"] - df["macd_signal"]

    delta = df["close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    df["rsi_14"] = 100 - (100 / (1 + gain / (loss + 1e-8)))

    bb_mid = df["close"].rolling(20).mean()
    bb_std = df["close"].rolling(20).std()
    bb_upper = bb_mid + 2 * bb_std
    bb_lower = bb_mid - 2 * bb_std
    df["bb_width"] = (bb_upper - bb_lower) / (bb_mid + 1e-8)
    df["bb_position"] = (df["close"] - bb_lower) / (bb_upper - bb_lower + 1e-8)

    vol_ma = df["volume"].rolling(10).mean()
    df["volume_ratio"] = df["volume"] / (vol_ma + 1e-8)

    df["volatility_10d"] = df["return_1d"].rolling(10).std() * np.sqrt(252)
    df["volatility_30d"] = df["return_1d"].rolling(30).std() * np.sqrt(252)

    for lag in [1, 2, 3, 5, 10]:
        df[f"close_lag_{lag}"] = df["close"].shift(lag)
    df["high_lag_1"] = df["high"].shift(1)
    df["low_lag_1"] = df["low"].shift(1)

    df["day_of_week"] = df.index.dayofweek
    df["month"] = df.index.month

    df["target"] = df["close"].shift(-1)

    return df.dropna()

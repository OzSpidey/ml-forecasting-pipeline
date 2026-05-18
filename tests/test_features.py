import numpy as np
import pandas as pd
import pytest
from src.features import add_features, FEATURE_COLS


def _make_ohlcv(n: int = 120) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    close = 100 + np.cumsum(rng.normal(0, 1, n))
    dates = pd.bdate_range("2023-01-01", periods=n)
    return pd.DataFrame({
        "open": close * (1 + rng.normal(0, 0.002, n)),
        "high": close * (1 + np.abs(rng.normal(0, 0.005, n))),
        "low": close * (1 - np.abs(rng.normal(0, 0.005, n))),
        "close": close,
        "volume": rng.integers(1_000_000, 5_000_000, n).astype(float),
    }, index=dates)


def test_all_feature_cols_present():
    df = add_features(_make_ohlcv())
    missing = [c for c in FEATURE_COLS if c not in df.columns]
    assert missing == [], f"Missing columns: {missing}"


def test_no_nulls_in_features():
    df = add_features(_make_ohlcv())
    nulls = df[FEATURE_COLS].isnull().sum().sum()
    assert nulls == 0, f"Found {nulls} null values in features"


def test_target_is_next_close():
    raw = _make_ohlcv(60)
    df = add_features(raw)
    # target[i] should equal close[i+1] in the original frame
    for i in range(min(10, len(df) - 1)):
        date = df.index[i]
        next_date = df.index[i + 1]
        assert abs(df.loc[date, "target"] - raw.loc[next_date, "close"]) < 1e-6


def test_rsi_bounds():
    df = add_features(_make_ohlcv())
    assert df["rsi_14"].between(0, 100).all(), "RSI out of [0, 100]"


def test_bb_position_finite():
    df = add_features(_make_ohlcv())
    assert np.isfinite(df["bb_position"]).all()

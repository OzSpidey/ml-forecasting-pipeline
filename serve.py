"""
FastAPI prediction service.

Run:  uvicorn serve:app --reload
Docs: http://localhost:8000/docs
"""
import json
import os
import time
import joblib
import numpy as np
import pandas as pd
from datetime import datetime, timezone
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from src.config import TICKERS
from src.fetch import fetch
from src.features import add_features, FEATURE_COLS

MODEL_STALE_DAYS = 7

app = FastAPI(
    title="Stock Forecasting API",
    description="Next-day close price predictions powered by XGBoost + MLflow",
    version="1.0.0",
)

_model_cache: dict = {}


def _load_model(ticker: str):
    if ticker not in _model_cache:
        path = f"models/{ticker}_best.joblib"
        if not os.path.exists(path):
            raise FileNotFoundError(f"No trained model for {ticker}. Run: python retrain.py --tickers {ticker}")
        _model_cache[ticker] = joblib.load(path)
    return _model_cache[ticker]


def _load_meta(ticker: str) -> dict:
    path = f"models/{ticker}_meta.json"
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return json.load(f)


class PredictionResponse(BaseModel):
    ticker: str
    current_price: float
    predicted_price: float
    expected_return_pct: float
    direction: str
    model_used: str
    test_rmse: float | None = None
    test_directional_accuracy: float | None = None


class HealthResponse(BaseModel):
    status: str
    available_tickers: list[str]
    trained_tickers: list[str]
    stale_tickers: list[str]


class BatchPredictionResponse(BaseModel):
    predictions: list[PredictionResponse]
    requested: int
    succeeded: int


def _model_age_days(ticker: str) -> float | None:
    path = f"models/{ticker}_best.joblib"
    if not os.path.exists(path):
        return None
    age = time.time() - os.path.getmtime(path)
    return age / 86400


@app.get("/health", response_model=HealthResponse)
def health():
    trained = [t for t in TICKERS if os.path.exists(f"models/{t}_best.joblib")]
    stale = [t for t in trained if (_model_age_days(t) or 0) > MODEL_STALE_DAYS]
    return HealthResponse(status="ok", available_tickers=TICKERS, trained_tickers=trained, stale_tickers=stale)


@app.get("/predict/{ticker}", response_model=PredictionResponse)
def predict(ticker: str):
    ticker = ticker.upper()
    if ticker not in TICKERS:
        raise HTTPException(400, f"Unsupported ticker. Choose from: {TICKERS}")

    try:
        model = _load_model(ticker)
    except FileNotFoundError as e:
        raise HTTPException(503, str(e))

    raw = fetch(ticker, period="6mo")
    df = add_features(raw)
    latest = df[FEATURE_COLS].iloc[[-1]]

    predicted = float(model.predict(latest)[0])
    current = float(df["close"].iloc[-1])
    ret_pct = round((predicted - current) / current * 100, 3)

    meta = _load_meta(ticker)

    return PredictionResponse(
        ticker=ticker,
        current_price=round(current, 2),
        predicted_price=round(predicted, 2),
        expected_return_pct=ret_pct,
        direction="UP" if predicted > current else "DOWN",
        model_used=meta.get("winner", "xgboost"),
        test_rmse=meta.get("rmse"),
        test_directional_accuracy=meta.get("directional_accuracy"),
    )


@app.get("/predict/all", response_model=BatchPredictionResponse)
def predict_all():
    results = []
    trained = [t for t in TICKERS if os.path.exists(f"models/{t}_best.joblib")]
    for ticker in trained:
        try:
            results.append(predict(ticker))
        except Exception:
            pass
    return BatchPredictionResponse(predictions=results, requested=len(TICKERS), succeeded=len(results))


@app.get("/tickers")
def list_tickers():
    return {"tickers": TICKERS}

# Stock Forecasting Pipeline

![CI](https://github.com/OzSpidey/ml-forecasting-pipeline/actions/workflows/ci.yml/badge.svg)
![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=flat&logo=python&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0-FF6600?style=flat)
![Prophet](https://img.shields.io/badge/Prophet-1.1-4285F4?style=flat)
![PyTorch](https://img.shields.io/badge/PyTorch-2.1-EE4C2C?style=flat&logo=pytorch&logoColor=white)
![MLflow](https://img.shields.io/badge/MLflow-2.12-0194E2?style=flat&logo=mlflow&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.110-009688?style=flat&logo=fastapi&logoColor=white)
![Dash](https://img.shields.io/badge/Plotly_Dash-2.16-3F4F75?style=flat)
![Docker](https://img.shields.io/badge/Docker-Compose-2496ED?style=flat&logo=docker&logoColor=white)
![Optuna](https://img.shields.io/badge/Optuna-3.6-5865F2?style=flat)
![pytest](https://img.shields.io/badge/pytest-8.0-0A9EDC?style=flat&logo=pytest&logoColor=white)

> A production-grade MLOps pipeline that trains three competing models (XGBoost · Prophet · LSTM) to forecast next-day stock closing prices, tracks every experiment in MLflow, auto-promotes the best model to Production, and serves predictions through a FastAPI endpoint and an interactive Plotly Dash dashboard.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      Data Layer                                 │
│   yfinance (OHLCV)  →  Feature Engineering (24 indicators)     │
│   RSI · MACD · Bollinger Bands · Lag Features · Volatility      │
└────────────────────────┬────────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────────┐
│                    Training Layer                                │
│  ┌────────────┐  ┌──────────────┐  ┌──────────────────────┐    │
│  │  XGBoost   │  │   Prophet    │  │    LSTM (PyTorch)     │    │
│  │  tabular   │  │  time-series │  │  sequence model       │    │
│  │  gradient  │  │  decomp +    │  │  GRU hidden=64        │    │
│  │  boosting  │  │  seasonality │  │  layers=2  seq=30d    │    │
│  └─────┬──────┘  └──────┬───────┘  └──────────┬───────────┘    │
│        └────────────────┴──────────────────────┘               │
│                         │  MLflow Tracking                      │
│              params · metrics · artifacts · registry            │
└────────────────────────┬────────────────────────────────────────┘
                         │  promote best RMSE → Production
┌────────────────────────▼────────────────────────────────────────┐
│                    Serving Layer                                 │
│  ┌─────────────────────┐    ┌──────────────────────────────┐    │
│  │  FastAPI REST API   │    │  Plotly Dash Dashboard        │   │
│  │  /predict/{ticker}  │    │  candlestick · MACD · RSI     │   │
│  │  /health  /tickers  │    │  model leaderboard · forecast │   │
│  └─────────────────────┘    └──────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────────┐
│                    Automation Layer                              │
│           GitHub Actions — Weekly Retraining (Mon 06:00 UTC)    │
│           docker-compose — MLflow + API + Dashboard             │
└─────────────────────────────────────────────────────────────────┘
```

---

## Metrics Tracked

| Metric | Description |
|---|---|
| **RMSE** | Root Mean Squared Error on held-out test set |
| **MAE** | Mean Absolute Error |
| **MAPE** | Mean Absolute Percentage Error |
| **DA%** | Directional Accuracy — % of days the model correctly predicted up/down |

---

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Train all models
```bash
# All 5 default tickers (AAPL, MSFT, TSLA, NVDA, SPY)
python retrain.py

# Specific tickers only
python retrain.py --tickers AAPL NVDA

# Skip LSTM for faster run
python retrain.py --no-lstm
```

Sample output:
```
================================================
  AAPL
================================================
  753 trading days loaded
  XGBoost | RMSE=2.3471  DA=54.8%
  Prophet | RMSE=4.1203  DA=51.2%
  LSTM    | RMSE=3.0812  DA=53.1%
  [AAPL] Best overall: xgboost | RMSE=2.3471 → Production

================================================
  LEADERBOARD
================================================
  Ticker   Model      RMSE     MAPE      DA%
  ------------------------------------------
  AAPL     xgboost    2.35    1.28%    54.8% ←
  AAPL     prophet    4.12    2.31%    51.2%
  AAPL     lstm       3.08    1.74%    53.1%
```

### 3. View experiments in MLflow
```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
# Open http://localhost:5000
```

### 4. Start the prediction API
```bash
uvicorn serve:app --reload
# Docs at http://localhost:8000/docs
```

Sample response:
```json
{
  "ticker": "AAPL",
  "current_price": 189.30,
  "predicted_price": 191.45,
  "expected_return_pct": 1.136,
  "direction": "UP",
  "model_used": "xgboost",
  "test_rmse": 2.3471,
  "test_directional_accuracy": 54.8
}
```

### 5. Launch the dashboard
```bash
python dashboard.py
# Open http://localhost:8050
```

---

## Docker (all services)
```bash
docker-compose up --build
```

| Service | URL |
|---|---|
| MLflow UI | http://localhost:5000 |
| FastAPI docs | http://localhost:8000/docs |
| Dash dashboard | http://localhost:8050 |

---

## Project Structure

```
ml-forecasting-pipeline/
├── src/
│   ├── config.py            # tickers, hyperparams, MLflow URI
│   ├── fetch.py             # yfinance data fetcher
│   ├── features.py          # 24 technical indicators + lag features
│   ├── train_xgboost.py     # XGBoost + MLflow logging
│   ├── train_prophet.py     # Prophet + MLflow logging
│   ├── train_lstm.py        # PyTorch LSTM + MLflow logging
│   ├── evaluate.py          # RMSE, MAE, MAPE, directional accuracy
│   ├── backtest.py          # walk-forward backtest, Sharpe, max drawdown
│   ├── tune.py              # Optuna hyperparameter search for XGBoost
│   └── register.py          # auto-promote best model to Production
├── tests/
│   ├── test_features.py     # feature engineering unit tests
│   └── test_evaluate.py     # metric computation unit tests
├── retrain.py               # training orchestrator (--tune, --no-lstm flags)
├── serve.py                 # FastAPI: /predict/{ticker}, /predict/all, /health
├── dashboard.py             # Plotly Dash interactive dashboard
├── pyproject.toml           # project metadata + pytest config
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── .github/workflows/
    ├── ci.yml               # run tests on every push
    └── retrain.yml          # weekly retraining on GitHub Actions
```

---

## Features Engineered

| Category | Features |
|---|---|
| Returns | 1-day, 5-day, 10-day percentage returns |
| Moving Averages | MA7, MA21, MA50 |
| MACD | MACD line, signal line, histogram |
| RSI | 14-period Relative Strength Index |
| Bollinger Bands | Band width, price position within bands |
| Volume | 10-day volume ratio |
| Volatility | 10-day and 30-day annualised realised volatility |
| Lag Features | Close/High/Low lagged 1–10 days |
| Calendar | Day of week, month |

---

## Tickers

Default: `AAPL MSFT TSLA NVDA SPY`

Add more in `src/config.py`:
```python
TICKERS = ["AAPL", "MSFT", "TSLA", "NVDA", "SPY", "GOOGL", "AMZN"]
```

---

## Backtesting

The walk-forward backtester in `src/backtest.py` simulates real trading — it trains on an expanding window, predicts one day ahead, and goes long when the model signals UP:

```
  Backtest | Sharpe=0.84  Return=18.3%  MaxDD=-12.4%  B&H=14.1%
```

Metrics logged to MLflow alongside training metrics so you can compare signal quality over time.

## Hyperparameter Tuning (Optuna)

```bash
python retrain.py --tune   # runs 50 Optuna trials per ticker before training
```

Searches over: `n_estimators`, `max_depth`, `learning_rate`, `subsample`, `colsample_bytree`, `min_child_weight`, `reg_alpha`, `reg_lambda`. Best params are logged as a separate `xgboost_tuned_{ticker}` run.

## Unit Tests

```bash
pytest tests/ -v
```

Tests cover: feature completeness, RSI bounds, Bollinger Band finiteness, target alignment, metric correctness, and edge cases.

## CI / Scheduled Retraining

GitHub Actions retrains all models every Monday at 06:00 UTC and uploads MLflow artifacts. Trigger manually via **Actions → Weekly Model Retraining → Run workflow**.

---

## License

MIT

"""
Orchestrator: fetch data → engineer features → train all three models →
log to MLflow → promote best to Production → run backtest.

Usage:
    python retrain.py                          # all default tickers
    python retrain.py --tickers AAPL MSFT      # specific tickers
    python retrain.py --tickers AAPL --no-lstm # skip LSTM (faster)
    python retrain.py --tune                   # Optuna hyperparameter search first
"""
import argparse
import mlflow
import xgboost as xgb
from src.config import TICKERS, MLFLOW_TRACKING_URI, MLFLOW_EXPERIMENT
from src.fetch import fetch
from src.features import add_features, FEATURE_COLS
from src.train_xgboost import train as train_xgb
from src.train_prophet import train as train_prophet
from src.train_lstm import train as train_lstm
from src.register import promote_best
from src.backtest import run_backtest


def main(tickers: list[str], skip_lstm: bool = False, tune: bool = False) -> None:
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    summary = {}
    for ticker in tickers:
        print(f"\n{'='*48}")
        print(f"  {ticker}")
        print("="*48)

        raw = fetch(ticker)
        df = add_features(raw)
        print(f"  {len(df)} trading days loaded")

        if tune:
            from src.tune import tune_xgboost
            print("  Running Optuna search (50 trials)...")
            tune_xgboost(df, ticker, n_trials=50)

        xgb_metrics, xgb_model = train_xgb(df, ticker)
        prophet_metrics, prophet_model = train_prophet(df, ticker)

        candidates = {
            "xgboost": (xgb_metrics, xgb_model),
            "prophet": (prophet_metrics, prophet_model),
        }

        if not skip_lstm:
            lstm_metrics, lstm_model, scaler = train_lstm(df, ticker)
            candidates["lstm"] = (lstm_metrics, lstm_model)

        winner = promote_best(ticker, candidates)

        print("  Running backtest...")
        bt_model = xgb.XGBRegressor(n_estimators=500, max_depth=6, learning_rate=0.05, random_state=42)
        run_backtest(df, bt_model, FEATURE_COLS, ticker)
        summary[ticker] = {k: v[0] for k, v in candidates.items()}
        summary[ticker]["winner"] = winner

    print(f"\n{'='*48}")
    print("  LEADERBOARD")
    print("="*48)
    header = f"  {'Ticker':<8} {'Model':<10} {'RMSE':>8} {'MAPE':>8} {'DA%':>7}"
    print(header)
    print("  " + "-"*42)
    for ticker, data in summary.items():
        winner = data.pop("winner")
        for model, m in data.items():
            tag = " ←" if model == winner else ""
            print(f"  {ticker:<8} {model:<10} {m['rmse']:>8.2f} {m['mape']:>7.2f}% {m['directional_accuracy']:>6.1f}%{tag}")

    print(f"\nMLflow UI: mlflow ui --backend-store-uri {MLFLOW_TRACKING_URI}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tickers", nargs="+", default=TICKERS)
    parser.add_argument("--no-lstm", action="store_true")
    parser.add_argument("--tune", action="store_true", help="Run Optuna hyperparameter search")
    args = parser.parse_args()
    main(args.tickers, skip_lstm=args.no_lstm, tune=args.tune)

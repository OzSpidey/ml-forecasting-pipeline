"""
Walk-forward backtesting: simulate trading with model signals and compare
against buy-and-hold. Reports Sharpe ratio, max drawdown, and total return.
"""
import numpy as np
import pandas as pd
import mlflow
from src.config import MLFLOW_EXPERIMENT


def run_backtest(
    df: pd.DataFrame,
    model,
    feature_cols: list[str],
    ticker: str,
    initial_capital: float = 10_000.0,
) -> dict:
    """
    Walk-forward backtest: train on expanding window, predict one step ahead,
    take a long position when model predicts price increase.

    Returns a dict of performance metrics and logs them to MLflow.
    """
    min_train = int(len(df) * 0.6)
    actuals, predictions, dates = [], [], []

    for i in range(min_train, len(df) - 1):
        train = df.iloc[:i]
        X_train = train[feature_cols].values
        y_train = train["target"].values
        X_pred = df[feature_cols].iloc[[i]]

        model.fit(X_train, y_train)
        pred = float(model.predict(X_pred)[0])
        actual_next = float(df["close"].iloc[i + 1])

        predictions.append(pred)
        actuals.append(actual_next)
        dates.append(df.index[i + 1])

    actuals = np.array(actuals)
    predictions = np.array(predictions)
    prices = df["close"].iloc[min_train + 1:].values[:len(actuals)]

    # Strategy: go long when model predicts UP, hold cash otherwise
    signals = (predictions > prices[:-1]).astype(float)
    # Shift signals by 1 (trade next day open)
    signals = np.concatenate([[0], signals[:-1]])

    daily_returns = np.diff(prices) / prices[:-1]
    strategy_returns = signals[1:] * daily_returns
    bh_returns = daily_returns

    def _sharpe(r: np.ndarray) -> float:
        return float(np.sqrt(252) * r.mean() / (r.std() + 1e-8))

    def _max_drawdown(r: np.ndarray) -> float:
        cumulative = (1 + r).cumprod()
        peak = np.maximum.accumulate(cumulative)
        dd = (cumulative - peak) / (peak + 1e-8)
        return float(dd.min())

    def _total_return(r: np.ndarray) -> float:
        return float((1 + r).prod() - 1) * 100

    metrics = {
        "backtest_sharpe": round(_sharpe(strategy_returns), 4),
        "backtest_max_drawdown_pct": round(_max_drawdown(strategy_returns) * 100, 2),
        "backtest_total_return_pct": round(_total_return(strategy_returns), 2),
        "bh_sharpe": round(_sharpe(bh_returns), 4),
        "bh_total_return_pct": round(_total_return(bh_returns), 2),
        "backtest_n_trades": int(signals.sum()),
    }

    mlflow.set_experiment(MLFLOW_EXPERIMENT)
    with mlflow.start_run(run_name=f"backtest_{ticker}", nested=True):
        mlflow.log_param("ticker", ticker)
        mlflow.log_metrics(metrics)

    print(
        f"  Backtest | Sharpe={metrics['backtest_sharpe']:.2f}  "
        f"Return={metrics['backtest_total_return_pct']:.1f}%  "
        f"MaxDD={metrics['backtest_max_drawdown_pct']:.1f}%  "
        f"B&H={metrics['bh_total_return_pct']:.1f}%"
    )
    return metrics

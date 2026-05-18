"""
Optuna hyperparameter search for XGBoost.
Finds the best params and re-logs a tuned run to MLflow.

Usage:
    from src.tune import tune_xgboost
    best_params = tune_xgboost(df, ticker, n_trials=50)
"""
import mlflow
import numpy as np
import optuna
import xgboost as xgb
import pandas as pd
from src.config import MLFLOW_EXPERIMENT, TEST_SPLIT
from src.evaluate import compute_metrics
from src.features import FEATURE_COLS

optuna.logging.set_verbosity(optuna.logging.WARNING)


def tune_xgboost(df: pd.DataFrame, ticker: str, n_trials: int = 50) -> dict:
    X = df[FEATURE_COLS].values
    y = df["target"].values
    split = int(len(X) * (1 - TEST_SPLIT))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 200, 800),
            "max_depth": trial.suggest_int("max_depth", 3, 9),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 7),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.5, 3.0),
            "random_state": 42,
        }
        model = xgb.XGBRegressor(**params)
        model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
        preds = model.predict(X_test)
        return compute_metrics(y_test, preds)["rmse"]

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    best = study.best_params

    # Re-train with best params and log to MLflow
    mlflow.set_experiment(MLFLOW_EXPERIMENT)
    with mlflow.start_run(run_name=f"xgboost_tuned_{ticker}"):
        mlflow.log_params({**best, "ticker": ticker, "model_type": "xgboost_tuned", "n_trials": n_trials})
        model = xgb.XGBRegressor(**best, random_state=42)
        model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
        preds = model.predict(X_test)
        metrics = compute_metrics(y_test, preds)
        mlflow.log_metrics(metrics)
        mlflow.xgboost.log_model(model, "model", registered_model_name=f"xgb-tuned-{ticker}")
        print(f"  Tuned XGBoost | RMSE={metrics['rmse']:.4f}  DA={metrics['directional_accuracy']:.1f}%")

    return best

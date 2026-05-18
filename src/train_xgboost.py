import joblib
import mlflow
import mlflow.xgboost
import xgboost as xgb
import numpy as np
import pandas as pd
from src.config import MLFLOW_EXPERIMENT, TEST_SPLIT
from src.evaluate import compute_metrics
from src.features import FEATURE_COLS


def train(df: pd.DataFrame, ticker: str) -> tuple[dict, xgb.XGBRegressor]:
    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    X = df[FEATURE_COLS].values
    y = df["target"].values

    split = int(len(X) * (1 - TEST_SPLIT))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    params = {
        "n_estimators": 500,
        "max_depth": 6,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 3,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "random_state": 42,
    }

    with mlflow.start_run(run_name=f"xgboost_{ticker}"):
        mlflow.log_params({**params, "ticker": ticker, "model_type": "xgboost"})

        model = xgb.XGBRegressor(**params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            verbose=False,
        )

        preds = model.predict(X_test)
        metrics = compute_metrics(y_test, preds)
        mlflow.log_metrics(metrics)

        mlflow.xgboost.log_model(
            model, "model",
            registered_model_name=f"xgb-{ticker}",
        )

        print(f"  XGBoost | RMSE={metrics['rmse']:.4f}  DA={metrics['directional_accuracy']:.1f}%")

    return metrics, model

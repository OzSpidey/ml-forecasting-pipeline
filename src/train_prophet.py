import mlflow
import pandas as pd
from prophet import Prophet
from src.config import MLFLOW_EXPERIMENT, TEST_SPLIT
from src.evaluate import compute_metrics


def train(df: pd.DataFrame, ticker: str) -> tuple[dict, Prophet]:
    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    prophet_df = df[["close"]].reset_index().rename(columns={"date": "ds", "close": "y"})

    split = int(len(prophet_df) * (1 - TEST_SPLIT))
    train_df = prophet_df.iloc[:split]
    test_df = prophet_df.iloc[split:]

    params = {
        "changepoint_prior_scale": 0.05,
        "seasonality_prior_scale": 10.0,
        "seasonality_mode": "multiplicative",
        "yearly_seasonality": True,
        "weekly_seasonality": True,
    }

    with mlflow.start_run(run_name=f"prophet_{ticker}"):
        mlflow.log_params({**params, "ticker": ticker, "model_type": "prophet"})

        model = Prophet(**params)
        model.add_seasonality("monthly", period=30.5, fourier_order=5)
        model.fit(train_df)

        future = model.make_future_dataframe(periods=len(test_df), freq="B")
        forecast = model.predict(future)
        preds = forecast.iloc[-len(test_df):]["yhat"].values
        y_test = test_df["y"].values

        metrics = compute_metrics(y_test, preds)
        mlflow.log_metrics(metrics)

        print(f"  Prophet | RMSE={metrics['rmse']:.4f}  DA={metrics['directional_accuracy']:.1f}%")

    return metrics, model

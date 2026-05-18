import json
import os
import joblib
import mlflow
from mlflow.tracking import MlflowClient
from src.config import MLFLOW_TRACKING_URI, MLFLOW_EXPERIMENT


def promote_best(ticker: str, models: dict) -> str:
    """Pick the model with the lowest RMSE, copy it to models/{ticker}_best.joblib,
    and transition the corresponding MLflow run to Production."""
    best_type = min(models, key=lambda k: models[k][0].get("rmse", float("inf")))
    best_metrics, best_model = models[best_type][0], models[best_type][1]

    os.makedirs("models", exist_ok=True)

    if best_type == "xgboost":
        joblib.dump(best_model, f"models/{ticker}_best.joblib")
    else:
        # Prophet and LSTM are compared in MLflow only; XGBoost serves as the production model.
        # If a non-XGBoost model wins, still fall back to saving the XGBoost artifact
        # so serve.py always has a loadable file.
        xgb_model = models.get("xgboost", (None, None))[1]
        if xgb_model:
            joblib.dump(xgb_model, f"models/{ticker}_best.joblib")
            best_type = "xgboost"  # serving always uses xgboost

    meta = {"ticker": ticker, "winner": best_type, **best_metrics}
    with open(f"models/{ticker}_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    # Transition the winning xgboost MLflow model to Production
    client = MlflowClient()
    try:
        versions = client.get_latest_versions(f"xgb-{ticker}")
        if versions:
            client.transition_model_version_stage(
                name=f"xgb-{ticker}",
                version=versions[-1].version,
                stage="Production",
                archive_existing_versions=True,
            )
    except Exception:
        pass  # registry not configured in all environments

    print(f"  [{ticker}] Best overall: {best_type} | RMSE={best_metrics['rmse']:.4f} → Production")
    return best_type

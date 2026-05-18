import mlflow
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from src.config import MLFLOW_EXPERIMENT, TEST_SPLIT, LSTM_SEQ_LEN, LSTM_EPOCHS, LSTM_HIDDEN, LSTM_LAYERS
from src.evaluate import compute_metrics
from src.features import FEATURE_COLS


class _LSTM(nn.Module):
    def __init__(self, input_size: int, hidden: int, layers: int, dropout: float = 0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden, layers, batch_first=True, dropout=dropout)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :]).squeeze(-1)


def _make_sequences(data: np.ndarray, seq_len: int):
    X, y = [], []
    for i in range(seq_len, len(data)):
        X.append(data[i - seq_len:i])
        y.append(data[i, 0])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


def train(df: pd.DataFrame, ticker: str) -> tuple[dict, _LSTM, MinMaxScaler]:
    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    cols = ["close"] + [c for c in FEATURE_COLS if c != "close"]
    data = df[cols].values

    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data)

    X, y = _make_sequences(data_scaled, LSTM_SEQ_LEN)

    split = int(len(X) * (1 - TEST_SPLIT))
    X_train = torch.from_numpy(X[:split])
    y_train = torch.from_numpy(y[:split])
    X_test = torch.from_numpy(X[split:])

    params = {
        "hidden_size": LSTM_HIDDEN,
        "num_layers": LSTM_LAYERS,
        "dropout": 0.2,
        "lr": 0.001,
        "epochs": LSTM_EPOCHS,
        "seq_len": LSTM_SEQ_LEN,
    }

    with mlflow.start_run(run_name=f"lstm_{ticker}"):
        mlflow.log_params({**params, "ticker": ticker, "model_type": "lstm"})

        model = _LSTM(X_train.shape[2], LSTM_HIDDEN, LSTM_LAYERS)
        optimizer = torch.optim.Adam(model.parameters(), lr=params["lr"])
        criterion = nn.MSELoss()

        model.train()
        for epoch in range(LSTM_EPOCHS):
            optimizer.zero_grad()
            loss = criterion(model(X_train), y_train)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            if (epoch + 1) % 10 == 0:
                mlflow.log_metric("train_loss", loss.item(), step=epoch + 1)

        model.eval()
        with torch.no_grad():
            preds_scaled = model(X_test).numpy()

        # Inverse-transform: fill dummy array to undo MinMaxScaler
        def _inverse(vals: np.ndarray) -> np.ndarray:
            dummy = np.zeros((len(vals), data.shape[1]))
            dummy[:, 0] = vals
            return scaler.inverse_transform(dummy)[:, 0]

        preds = _inverse(preds_scaled)
        y_test = _inverse(y[split:])

        metrics = compute_metrics(y_test, preds)
        mlflow.log_metrics(metrics)

        print(f"  LSTM    | RMSE={metrics['rmse']:.4f}  DA={metrics['directional_accuracy']:.1f}%")

    return metrics, model, scaler

"""
Plotly Dash dashboard.

Run:  python dashboard.py
Open: http://localhost:8050
"""
import json
import os
import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import dash
from dash import dcc, html, Input, Output, dash_table
from src.config import TICKERS
from src.fetch import fetch
from src.features import add_features, FEATURE_COLS

app = dash.Dash(__name__, title="Stock Forecasting Dashboard")

DARK = "#0d1117"
CARD = "#161b22"
ACCENT = "#58a6ff"
GREEN = "#3fb950"
RED = "#f85149"
TEXT = "#c9d1d9"
MUTED = "#8b949e"

app.layout = html.Div(style={"backgroundColor": DARK, "minHeight": "100vh", "fontFamily": "monospace", "color": TEXT, "padding": "24px"}, children=[
    html.H1("Stock Forecasting Pipeline", style={"color": ACCENT, "marginBottom": "4px"}),
    html.P("XGBoost · Prophet · LSTM  —  tracked with MLflow", style={"color": MUTED, "marginBottom": "24px"}),

    html.Div(style={"display": "flex", "gap": "16px", "marginBottom": "24px"}, children=[
        html.Div(style={"flex": 1}, children=[
            html.Label("Ticker", style={"color": MUTED, "fontSize": "12px"}),
            dcc.Dropdown(
                id="ticker-dd",
                options=[{"label": t, "value": t} for t in TICKERS],
                value=TICKERS[0],
                style={"backgroundColor": CARD, "color": TEXT, "border": f"1px solid {MUTED}"},
            ),
        ]),
        html.Div(style={"flex": 1}, children=[
            html.Label("Look-back period", style={"color": MUTED, "fontSize": "12px"}),
            dcc.Dropdown(
                id="period-dd",
                options=[{"label": "6 months", "value": "6mo"}, {"label": "1 year", "value": "1y"}, {"label": "2 years", "value": "2y"}],
                value="1y",
                style={"backgroundColor": CARD, "color": TEXT, "border": f"1px solid {MUTED}"},
            ),
        ]),
    ]),

    dcc.Graph(id="price-chart", style={"marginBottom": "24px"}),
    dcc.Graph(id="indicator-chart", style={"marginBottom": "24px"}),

    html.H3("Model Comparison", style={"color": ACCENT, "marginBottom": "8px"}),
    html.Div(id="model-table"),

    html.Div(id="prediction-card", style={"marginTop": "24px"}),
])


def _meta(ticker: str) -> dict:
    path = f"models/{ticker}_meta.json"
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return json.load(f)


@app.callback(
    Output("price-chart", "figure"),
    Output("indicator-chart", "figure"),
    Output("model-table", "children"),
    Output("prediction-card", "children"),
    Input("ticker-dd", "value"),
    Input("period-dd", "value"),
)
def update(ticker: str, period: str):
    raw = fetch(ticker, period=period)
    df = add_features(raw)

    # ── Price chart ──────────────────────────────────────────────────────────
    fig_price = go.Figure()
    fig_price.add_trace(go.Candlestick(
        x=df.index, open=raw.loc[df.index, "open"], high=raw.loc[df.index, "high"],
        low=raw.loc[df.index, "low"], close=df["close"],
        increasing_line_color=GREEN, decreasing_line_color=RED, name="OHLC",
    ))
    fig_price.add_trace(go.Scatter(x=df.index, y=df["ma_21"], name="MA21", line=dict(color="#d29922", width=1)))
    fig_price.add_trace(go.Scatter(x=df.index, y=df["ma_50"], name="MA50", line=dict(color=ACCENT, width=1)))

    # Forecast line (last 20 days predicted by saved model)
    model_path = f"models/{ticker}_best.joblib"
    if os.path.exists(model_path):
        model = joblib.load(model_path)
        forecast_df = df.tail(20)
        preds = model.predict(forecast_df[FEATURE_COLS])
        pred_dates = pd.date_range(start=forecast_df.index[0], periods=len(preds) + 1, freq="B")[1:]
        fig_price.add_trace(go.Scatter(
            x=pred_dates, y=preds, name="XGBoost Forecast",
            line=dict(color="#bc8cff", width=2, dash="dot"),
        ))

    fig_price.update_layout(
        paper_bgcolor=DARK, plot_bgcolor=CARD, font_color=TEXT,
        title=f"{ticker} — Price & Forecast", xaxis_rangeslider_visible=False,
        legend=dict(bgcolor=CARD), margin=dict(l=40, r=20, t=50, b=30),
        height=450,
    )

    # ── Indicator chart ───────────────────────────────────────────────────────
    fig_ind = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.5, 0.5],
                            subplot_titles=["MACD", "RSI (14)"])
    fig_ind.add_trace(go.Bar(x=df.index, y=df["macd_hist"], name="MACD Hist",
                             marker_color=np.where(df["macd_hist"] >= 0, GREEN, RED)), row=1, col=1)
    fig_ind.add_trace(go.Scatter(x=df.index, y=df["macd"], name="MACD", line=dict(color=ACCENT, width=1)), row=1, col=1)
    fig_ind.add_trace(go.Scatter(x=df.index, y=df["macd_signal"], name="Signal", line=dict(color="#d29922", width=1)), row=1, col=1)
    fig_ind.add_trace(go.Scatter(x=df.index, y=df["rsi_14"], name="RSI", line=dict(color="#bc8cff", width=1)), row=2, col=1)
    fig_ind.add_hline(y=70, line_dash="dash", line_color=RED, opacity=0.5, row=2, col=1)
    fig_ind.add_hline(y=30, line_dash="dash", line_color=GREEN, opacity=0.5, row=2, col=1)
    fig_ind.update_layout(paper_bgcolor=DARK, plot_bgcolor=CARD, font_color=TEXT,
                          showlegend=True, legend=dict(bgcolor=CARD),
                          margin=dict(l=40, r=20, t=50, b=30), height=350)

    # ── Model comparison table ────────────────────────────────────────────────
    meta = _meta(ticker)
    if meta:
        table = dash_table.DataTable(
            columns=[{"name": c, "id": c} for c in ["Model", "RMSE", "MAE", "MAPE", "DA%", "Status"]],
            data=[{
                "Model": meta.get("winner", "xgboost"),
                "RMSE": meta.get("rmse", "—"),
                "MAE": meta.get("mae", "—"),
                "MAPE": f"{meta.get('mape', 0):.2f}%",
                "DA%": f"{meta.get('directional_accuracy', 0):.1f}%",
                "Status": "Production",
            }],
            style_table={"overflowX": "auto"},
            style_cell={"backgroundColor": CARD, "color": TEXT, "border": f"1px solid {MUTED}", "fontFamily": "monospace"},
            style_header={"backgroundColor": DARK, "color": ACCENT, "fontWeight": "bold"},
        )
    else:
        table = html.P("No trained model found. Run: python retrain.py", style={"color": RED})

    # ── Prediction card ───────────────────────────────────────────────────────
    if os.path.exists(model_path):
        model = joblib.load(model_path)
        latest = df[FEATURE_COLS].iloc[[-1]]
        predicted = float(model.predict(latest)[0])
        current = float(df["close"].iloc[-1])
        ret_pct = (predicted - current) / current * 100
        color = GREEN if ret_pct >= 0 else RED
        arrow = "▲" if ret_pct >= 0 else "▼"
        card = html.Div(style={"backgroundColor": CARD, "borderRadius": "8px", "padding": "20px",
                                "border": f"1px solid {color}", "display": "inline-block", "minWidth": "300px"}, children=[
            html.P("Next-Day Forecast", style={"color": MUTED, "margin": 0, "fontSize": "12px"}),
            html.H2(f"${predicted:.2f}", style={"color": TEXT, "margin": "4px 0"}),
            html.P(f"{arrow} {ret_pct:+.2f}% from ${current:.2f}", style={"color": color, "margin": 0}),
        ])
    else:
        card = html.Div()

    return fig_price, fig_ind, table, card


if __name__ == "__main__":
    app.run(debug=True, port=8050)

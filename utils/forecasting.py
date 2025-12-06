import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from math import sqrt
from datetime import timedelta
import tensorflow as tf

from models.lstm_model import train_lstm_model


def create_supervised_series(values: np.ndarray, n_lags: int):
    """
    Turn 1D series into supervised learning pairs:
    X: [t-n_lags ... t-1], y: [t]
    """
    X, y = [], []
    for i in range(n_lags, len(values)):
        X.append(values[i - n_lags:i])
        y.append(values[i])
    return np.array(X), np.array(y)


def lstm_forecast(df: pd.DataFrame,
                  target_col: str = "Close",
                  n_lags: int = 20,
                  horizon: int = 7,
                  epochs: int = 40,
                  batch_size: int = 32):

    # 1. Extract and scale the series
    series = df[target_col].dropna().values.astype("float32")

    if len(series) <= n_lags + horizon + 10:
        return None, {"rmse_train": 0.0, "rmse_test": 0.0}

    scaler = MinMaxScaler(feature_range=(0.0, 1.0))
    series_scaled = scaler.fit_transform(series.reshape(-1, 1)).flatten()

    # 2. Supervised data on scaled series
    X_scaled, y_scaled = create_supervised_series(series_scaled, n_lags)

    split_idx = int(len(X_scaled) * 0.8)
    X_train, X_val = X_scaled[:split_idx], X_scaled[split_idx:]
    y_train, y_val = y_scaled[:split_idx], y_scaled[split_idx:]

    # LSTM expects 3D input: (samples, timesteps, features)
    X_train = X_train[..., np.newaxis]
    X_val = X_val[..., np.newaxis]

    # 3. Train LSTM on scaled data
    model = train_lstm_model(
        X_train, y_train,
        X_val, y_val,
        epochs=epochs,
        batch_size=batch_size,
        input_length=n_lags,
    )

    # 4. Evaluate (RMSE in original price scale)
    y_train_pred_scaled = model.predict(X_train, verbose=0).flatten()
    y_val_pred_scaled = model.predict(X_val, verbose=0).flatten()

    # Inverse transform to price scale
    y_train_true = scaler.inverse_transform(y_train.reshape(-1, 1)).flatten()
    y_val_true = scaler.inverse_transform(y_val.reshape(-1, 1)).flatten()
    y_train_pred = scaler.inverse_transform(
        y_train_pred_scaled.reshape(-1, 1)
    ).flatten()
    y_val_pred = scaler.inverse_transform(
        y_val_pred_scaled.reshape(-1, 1)
    ).flatten()

    rmse_train = sqrt(mean_squared_error(y_train_true, y_train_pred))
    rmse_val = sqrt(mean_squared_error(y_val_true, y_val_pred))

    # 5. Multi-step forecast (still on scaled data, then inverse transform)
    last_window_scaled = series_scaled[-n_lags:].copy()
    preds_scaled = []

    for _ in range(horizon):
        x_input = last_window_scaled.reshape(1, n_lags, 1)
        next_pred_scaled = model.predict(x_input, verbose=0)[0, 0]
        preds_scaled.append(float(next_pred_scaled))
        last_window_scaled = np.roll(last_window_scaled, -1)
        last_window_scaled[-1] = next_pred_scaled

    # Convert scaled preds back to prices
    preds = scaler.inverse_transform(
        np.array(preds_scaled).reshape(-1, 1)
    ).flatten()

    last_date = df.index[-1]
    future_dates = [last_date + timedelta(days=i + 1) for i in range(horizon)]

    forecast_df = pd.DataFrame({
        "Date": future_dates,
        "Prediction": preds
    })

    metrics = {
        "rmse_train": rmse_train,
        "rmse_test": rmse_val
    }

    return forecast_df, metrics

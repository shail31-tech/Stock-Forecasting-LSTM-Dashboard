import numpy as np
import pandas as pd


def add_moving_averages(df: pd.DataFrame,
                        windows=(20, 50, 200)) -> pd.DataFrame:
    for w in windows:
        df[f"MA{w}"] = df["Close"].rolling(window=w, min_periods=1).mean()
    return df


def add_bollinger_bands(df: pd.DataFrame,
                        window: int = 20,
                        num_std: float = 2.0) -> pd.DataFrame:
    rolling_mean = df["Close"].rolling(window=window, min_periods=1).mean()
    rolling_std = df["Close"].rolling(window=window, min_periods=1).std()
    df["BB_upper"] = rolling_mean + num_std * rolling_std
    df["BB_lower"] = rolling_mean - num_std * rolling_std
    return df


def add_rsi(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """
    Compute Relative Strength Index (RSI) using Wilder's method.
    """
    delta = df["Close"].diff()

    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)

    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()

    rs = avg_gain / (avg_loss.replace(0, np.nan))
    rsi = 100 - (100 / (1 + rs))

    df["RSI"] = rsi
    return df


def add_macd(df: pd.DataFrame,
             fast: int = 12,
             slow: int = 26,
             signal: int = 9) -> pd.DataFrame:
    ema_fast = df["Close"].ewm(span=fast, adjust=False).mean()
    ema_slow = df["Close"].ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    macd_signal = macd.ewm(span=signal, adjust=False).mean()

    df["MACD"] = macd
    df["MACD_signal"] = macd_signal
    return df


def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = add_moving_averages(df)
    df = add_bollinger_bands(df)
    df = add_rsi(df)
    df = add_macd(df)
    return df


def compute_risk_metrics(returns: pd.Series,
                         risk_free_rate: float = 0.0) -> dict:
    returns = returns.dropna()
    if returns.empty:
        return {
            "annual_volatility": 0.0,
            "sharpe_ratio": 0.0,
            "max_drawdown": 0.0,
        }

    daily_mean = returns.mean()
    daily_std = returns.std()

    annual_vol = daily_std * np.sqrt(252)

    rf_daily = risk_free_rate / 252.0
    sharpe = 0.0
    if daily_std > 0:
        sharpe = (daily_mean - rf_daily) / daily_std * np.sqrt(252)

    cumulative = (1 + returns).cumprod()
    running_max = cumulative.cummax()
    drawdown = (cumulative - running_max) / running_max
    max_dd = drawdown.min()

    return {
        "annual_volatility": float(annual_vol),
        "sharpe_ratio": float(sharpe),
        "max_drawdown": float(max_dd),
    }

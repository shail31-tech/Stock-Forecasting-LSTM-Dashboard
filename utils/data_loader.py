from typing import Optional
import pandas as pd
import yfinance as yf


def load_data(ticker: str, start, end) -> Optional[pd.DataFrame]:
    """
    Download OHLCV data for a ticker between start and end dates.
    Flattens MultiIndex columns (from yfinance) to simple strings.
    """
    try:
        df = yf.download(ticker, start=start, end=end)

        if df is None or df.empty:
            return None

        # Ensure Date is a proper index
        df.index = pd.to_datetime(df.index)

        # Flatten MultiIndex columns if needed
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [
                col[0] if isinstance(col, tuple) else col
                for col in df.columns
            ]

        return df

    except Exception as e:
        print(f"Error loading data for {ticker}: {e}")
        return None

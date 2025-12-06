import streamlit as st
import pandas as pd
from datetime import date

from utils.data_loader import load_data
from utils.indicators import add_technical_indicators, compute_risk_metrics
from utils.forecasting import lstm_forecast

st.set_page_config(page_title="Stock Insights & Forecasting Dashboard",
                   layout="wide")

st.title("📈 Stock Insights & Forecasting Dashboard")

# ---- Sidebar inputs ----
st.sidebar.header("Settings")

ticker = st.sidebar.text_input("Stock Ticker", value="AAPL")
start = st.sidebar.date_input("Start Date", value=date(2020, 1, 1))
end = st.sidebar.date_input("End Date", value=date.today())

show_indicators = st.sidebar.checkbox("Show Technical Indicators", value=True)
show_risk = st.sidebar.checkbox("Show Risk Metrics", value=True)
show_forecast = st.sidebar.checkbox("Enable ML Forecasting", value=True)

forecast_horizon = st.sidebar.number_input(
    "Forecast horizon (days)", min_value=1, max_value=60, value=7
)
n_lags = st.sidebar.number_input(
    "Lag window for forecasting", min_value=3, max_value=60, value=10
)

if st.sidebar.button("Load & Analyze"):
    with st.spinner("Fetching data..."):
        df = load_data(ticker, start, end)

    if df is None or df.empty:
        st.error("No data returned. Check ticker or date range.")
    else:
        st.success(f"Loaded {len(df)} rows for {ticker}")

        # ---- Price chart ----
        st.subheader(f"Closing Price for {ticker}")
        st.line_chart(df["Close"])

        # ---- Technical Indicators ----
        if show_indicators:
            st.subheader("Technical Indicators")

            df_ind = add_technical_indicators(df.copy())
            if df_ind is None or df_ind.empty:
                st.error("Failed to compute technical indicators. Data may be too short.")
                st.stop()

            ma_cols = [c for c in df_ind.columns if c.startswith("MA")]
            bb_cols = ["BB_upper", "BB_lower"]
            macd_cols = ["MACD", "MACD_signal"]
            rsi_col = "RSI"

            st.markdown("**Moving Averages**")
            st.line_chart(df_ind[["Close"] + ma_cols])

            st.markdown("**Bollinger Bands**")
            st.line_chart(df_ind[["Close"] + bb_cols])

            st.markdown("**MACD**")
            st.line_chart(df_ind[macd_cols])

            st.markdown("**RSI**")
            st.line_chart(df_ind[[rsi_col]])

        # ---- Returns & Risk metrics ----
        st.subheader("Daily Returns")
        returns = df["Close"].pct_change().dropna()
        st.line_chart(returns)

        if show_risk:
            st.subheader("Risk Metrics")
            risk = compute_risk_metrics(returns)
            col1, col2, col3 = st.columns(3)
            col1.metric("Annualized Volatility", f"{risk['annual_volatility']:.2%}")
            col2.metric("Sharpe Ratio (rf=0)", f"{risk['sharpe_ratio']:.2f}")
            col3.metric("Max Drawdown", f"{risk['max_drawdown']:.2%}")

        # ---- Forecasting ----
        if show_forecast:
            st.subheader("📉 LSTM Price Forecast")

            forecast_df, metrics = lstm_forecast(
                df,
                target_col="Close",
                n_lags=int(n_lags),
                horizon=int(forecast_horizon),
            )

            if forecast_df is None:
                st.warning("Not enough data to run forecasting. Reduce lag/horizon.")
            else:
                st.write("Model: **LSTM**")
                c1, c2 = st.columns(2)
                c1.metric("Train RMSE", f"{metrics['rmse_train']:.4f}")
                c2.metric("Validation RMSE", f"{metrics['rmse_test']:.4f}")

            st.markdown("**Historical Close + Forecast**")
            combined = pd.concat(
                [df[["Close"]], forecast_df.set_index("Date")],
                axis=0
            )
            st.line_chart(combined)

            # 🔹 NEW: separate chart just for predictions
            st.markdown("**Forecast Only (LSTM predictions)**")
            st.line_chart(
                forecast_df.set_index("Date")[["Prediction"]]
            )

            st.markdown("**Forecast Table**")
            st.dataframe(forecast_df)


else:
    st.info("👈 Set ticker, dates and click **Load & Analyze** to start.")

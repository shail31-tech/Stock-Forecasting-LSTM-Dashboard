🚀 LSTM Stock Forecasting Dashboard

An interactive Streamlit dashboard that visualizes stock performance, computes technical indicators, analyzes risk metrics, and generates multi-step price forecasts using a custom-built LSTM deep learning model.

This project is ideal for showcasing machine learning, time-series forecasting, data engineering, and full-stack ML app development skills.

📌 Features
📈 1. Stock Data Fetching

Downloads OHLC data using yfinance

Dynamic ticker input (default: AAPL)

Adjustable date range

📊 2. Technical Indicators

Automatically computes:

Moving Averages (MA20, MA50, MA200)

Bollinger Bands

MACD & Signal Line

RSI (Wilder’s method)

All displayed with beautiful Streamlit line charts.

🧮 3. Risk Metrics

Includes professional trading desk metrics:

Annualized Volatility

Sharpe Ratio

Max Drawdown

Daily Returns Visualization

🤖 4. LSTM Price Forecasting

A custom-built LSTM neural network predicts future stock closing prices:

✔ Multi-step forecasting

Predicts N days ahead (configurable).

✔ Scaled training

Uses MinMaxScaler for improved model stability.

✔ Train/Test RMSE

Displays accuracy metrics.

✔ 2 Plot Views

Historical + Predicted

Predictions-only (future values)

🧠 Technologies Used
Category	Tools
ML / DL	TensorFlow (LSTM), NumPy, scikit-learn
Data	pandas, yfinance
Dashboard	Streamlit
Visualization	Plotly / Streamlit charts
Environment	Conda (Python 3.9)
🗂️ Project Structure
├── app.py
│
├── models/
│   └── lstm_model.py
│
├── utils/
│   ├── data_loader.py
│   ├── indicators.py
│   └── forecasting.py
│
├── requirements.txt
└── README.md

⚙️ Installation & Setup
1️⃣ Clone the repository
git clone https://github.com/USERNAME/Stock-Forecasting-LSTM-Dashboard.git
cd Stock-Forecasting-LSTM-Dashboard

2️⃣ Create a clean environment

(Recommended: Python 3.9)

conda create -n finance39 python=3.9
conda activate finance39

3️⃣ Install dependencies
pip install -r requirements.txt

4️⃣ Run the dashboard
streamlit run app.py

📌 Usage
Sidebar Controls

Choose ticker (e.g., AAPL, TSLA, MSFT)

Select date range

Enable/disable:

Technical indicators

Risk metrics

LSTM forecasting

Configure:

Forecast horizon

LSTM lag window

Output Includes:

Stock price chart

Technical indicator charts

Daily returns

Risk metrics

Forecast charts

Forecast table

📈 Example Output
✔ Historical + Forecast plot

Shows how the LSTM model extends the price into the future.

✔ Prediction-only chart

A clean forward-looking forecast view.

✔ Forecast Table

Displays predicted price for each future date.

🎯 Model Details
LSTM Architecture

Input shape: (lag_window, 1)

1 LSTM layer (64 units)

Dense output layer (1 unit)

Adam optimizer (1e-3)

EarlyStopping callback

MinMax scaling for stability

Forecasting Method

Uses multi-step iterative prediction, feeding each predicted value back into the model.

🌟 Future Improvements

Add GRU or Transformer-based forecasting

Compare with ARIMA/Prophet

Add hyperparameter tuning panel

Deploy on Streamlit Cloud

Add buy/sell strategy simulation

🚀 LSTM Stock Forecasting Dashboard

An interactive Streamlit dashboard that visualizes stock performance, computes technical indicators, analyzes risk metrics, and forecasts future prices using a custom-built LSTM neural network.

📌 Features
📈 Stock Data Fetching

Fetches OHLC data from yfinance

User-defined ticker (e.g., AAPL, TSLA, MSFT)

Adjustable date range

📊 Technical Indicators

Includes plots for:

Moving Averages (MA20, MA50, MA200)

Bollinger Bands

MACD & Signal Line

RSI (Wilder’s Method)

🧮 Risk Metrics

Computed directly from daily returns:

Annualized Volatility

Sharpe Ratio

Max Drawdown

Daily Returns Chart

🤖 LSTM Forecasting

A custom LSTM neural network predicts future closing prices.

Features:

Multi-step iterative forecasting

Automatic scaling (MinMaxScaler)

Train/Test RMSE

Two charts:

Historical + Forecast

Forecast-only

🧠 Technologies Used
Category	Tools
Machine Learning	TensorFlow (LSTM), NumPy, scikit-learn
Data Handling	pandas, yfinance
Visualization	Streamlit, Matplotlib/Plotly
Deployment	Streamlit
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

2️⃣ Create a virtual environment
conda create -n finance39 python=3.9
conda activate finance39

3️⃣ Install dependencies
pip install -r requirements.txt

4️⃣ Run the app
streamlit run app.py

📌 Usage
Sidebar Controls

Select stock ticker

Choose date range

Toggle:

Technical indicators

Risk metrics

Forecasting

Adjust:

Forecast horizon

LSTM lag window

Dashboard Output

Closing price chart

Technical indicator charts

Daily returns

Risk metrics

Forecast charts

Forecast-only chart

Forecast table

🤖 Model Details
LSTM Architecture

LSTM layer (64 units)

Dense output layer

Adam optimizer

EarlyStopping callback

MinMax scaling

Forecasting Method

Multi-step iterative prediction

Each predicted value becomes the next input

🌟 Future Improvements

Add GRU & Transformer forecasting

Compare with ARIMA / Prophet

Add sentiment analysis using news

Deploy to Streamlit Cloud

Add trading strategy simulations

🧑‍💻 Author

Shail Shah
📌 Data Science | Machine Learning | AI

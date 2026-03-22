import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from Tsla_stock import stock_data, stock_metrics
import warnings
import os 
warnings.filterwarnings("ignore")




# stock data will show distribution of returns and will also check for normality test

ticker = "TSLA"
start = "2010-01-01"
end = "2024-10-31"
dff = stock_data(ticker,start,end)

# we will look at distribution of intra day high-low for all these  years, min, max

stock_metrics(dff)

# Load and prepare data
data = pd.read_csv(r"C:\Users\jay\Documents\Tsla_Combined_data\Tsla_Combined_1_min.csv", index_col=0, parse_dates=True, skipfooter=1, engine="python")
# data.drop(["Change", "%Chg"], axis=1, inplace=True)
# data.rename(columns={"Last": "Close"}, inplace=True)

# List to store trade details
trades = []

def process_data(df):
    # Add technical indicators and calculated columns
    df["pct_change"] = np.log(df["Close"] / df["Open"]).cumsum()
    df["ewm"] = df["Close"].ewm(span=5).mean()
    df["sma"] = df["Close"].rolling(window=10).mean()
    df["upper"] = df["sma"] + 2 * df["Close"].rolling(window=10).std()
    df["lower"] = df["sma"] - 2 * df["Close"].rolling(window=10).std()
    df["run_low"] = np.minimum.accumulate(df["pct_change"])
    
    # Plotting the indicators for visual debugging
    df[["Close", "ewm", "sma", "upper", "lower"]].plot()
    plt.show()

    # Initialize state variables for each date
    signal_triggered = False
    trade_active = False
    stop_loss = None
    profit_target = None
    signal_price = None

    # Create columns for signals and events
    df["signal"] = 0
    df["stop_loss_trigger"] = 0
    df["profit_taken"] = 0

    # Iterate through the DataFrame to apply trading logic
    for i in range(len(df)):
        if not trade_active:
            # Check if signal condition is met
            if (
                df["ewm"].iloc[i] > df["sma"].iloc[i] and
                df["run_low"].iloc[i] < df["pct_change"].iloc[i] and
                df["run_low"].iloc[i] < -0.04 and
                df.index[i].time() < pd.to_datetime("10:30:00").time()
            ):
                df.at[df.index[i], "signal"] = 1  # Signal generated
                signal_triggered = True  # Signal generated
                trade_active = True  # Lock signal for trade
                signal_price = df["Close"].iloc[i]
                stop_loss = (0.98)*df["lower"].iloc[i]  # Set stop-loss price
                profit_target = signal_price * 1.03  # Set profit target (2% above signal price)

                # Save trade details
                trades.append({
                    "entry_price": signal_price,
                    "stop_loss": stop_loss,
                    "stop_loss_hit_price": None,
                    "profit_taken_price": None,
                    "signal_time": df.index[i],
                    "stop_loss_time": None,
                    "profit_time": None,
                })
                print(50 * "=")
                print(f"Signal generated at {df.index[i]} with entry price {signal_price}, stop-loss {stop_loss}, and profit target {profit_target}")

        elif trade_active and signal_triggered:
            # Check for stop-loss condition
            if df["Close"].iloc[i] <= stop_loss and trades[-1]["stop_loss_time"] is None:
                df.at[df.index[i], "stop_loss_trigger"] = 1
                trades[-1]["stop_loss_hit_price"] = df["Close"].iloc[i]
                trades[-1]["stop_loss_time"] = df.index[i]
                trade_active = False  # Reset for next trade cycle
                signal_triggered = False  # Reset signal state
                print(50*"=")
                print(f"Stop-loss triggered at {df.index[i]} with price {df['Close'].iloc[i]}")

            # Check for profit target condition
            elif df["Close"].iloc[i] >= profit_target and trades[-1]["profit_time"] is None:
                df.at[df.index[i], "profit_taken"] = 1
                trades[-1]["profit_taken_price"] = df["Close"].iloc[i]
                trades[-1]["profit_time"] = df.index[i]
                trade_active = False  # Reset for next trade cycle
                signal_triggered = False  # Reset signal state
                print(50*"=")
                print(f"Profit target reached at {df.index[i]} with price {df['Close'].iloc[i]}")

        # Check if the trade has not been closed by stop-loss or profit target by the end of the day
        if trade_active and i == len(df) - 1:  # Last candle of the day
            # Close the trade at the day's close price
            trade_active = False  # Reset for next trade cycle
            signal_triggered = False  # Reset signal state
            trades[-1]["close_price"] = df["Close"].iloc[-1]
            trades[-1]["close_time"] = df.index[-1]
            print(50 * "=")
            print(f"End of day - trade closed at {df.index[-1]} with price {df['Close'].iloc[-1]}")

# Process data for each unique day in the dataset
unique_dates = np.unique(data.index.date)
for unique_date in unique_dates:
    # Filter data for the current date
    df = data[data.index.date == unique_date].copy()
    print(f"Processing data for {unique_date}")
    process_data(df)

# Convert the list of trades into a DataFrame
tradess = pd.DataFrame(trades)

# Display all trade details
print("\nTrade Details:")
print(tradess.tail())



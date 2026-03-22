
import pandas as pd
import mplfinance as mpf
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as scs
import statsmodels.api as sm
from statsmodels.stats.stattools import jarque_bera
import warnings
warnings.filterwarnings("ignore")
import yfinance as yf
import logging

# Function to find the distribution of returns
def returns_distribution(data, year_wise = None):
    """ The function uses data as input which can be rendered via glob library
        the function calculates and plots distribution of returns plot overlaid by
        pdf for normally distributed data. For pdf we use min, max, mean and standard deviation
        to plot.
        
        Additionally to check normality we also use jarque bera test to validate if the series
        is normally distributed or not"""
    try:
        df = pd.read_csv(data, parse_dates = True, skipfooter = 1, index_col = 0, engine = "python") # here one can iterate through files using glob
        df["log_returns"] = np.log(df.Last / df.Last.shift(1))
        df.dropna(inplace = True)
        log_returns_range = np.linspace(min(df["log_returns"]), max(df["log_returns"]), num = 1000) # max and min values of log_returns
        mean = df["log_returns"].mean() # mean of log returns
        std_dev = df["log_returns"].std() # standard deviation of log returns
        norm_pdf = scs.norm.pdf(log_returns_range, loc = mean, scale = std_dev) # using scipy we make pdf function using max, min range, std, mean
        fig, ax = plt.subplots(1,2, figsize = (15,7.5))
        sns.distplot(df["log_returns"], kde = False, norm_hist = True, ax = ax[0]) # norm_hist -> to make area of histogram -> 1
        ax[0].set_title("Distribution of Tsla returns", fontsize = 15)
        ax[0].plot(log_returns_range, norm_pdf, color = "Red", linewidth = 2, label =f"N({mean:.4f},{std_dev**2:.5f})")
        ax[0].legend(loc="best")
        # Q-Q plot analysis
        sm.qqplot(df["log_returns"], line = "q", ax = ax[1])
        ax[1].set_title("Q-Q plot", fontsize = 15)
        plt.show()
    except Exception as e:
        print("Could not compute the normality test for the given ticker", e)
        
    def jarque_bera_test():
        try:
            jb_stat, p_value, skewness, Kurtosis = jarque_bera(df["log_returns"])
            print(f"The T stat value for the Tsla returns is {jb_stat:.4f}, the P-Value is {p_value:.3f}, skewness:{skewness} and kurtosis of {Kurtosis}")
            print("="*100)
        except Exception as er:
            print("Could not compute Jarque bera test for the ticker", er)
    jarque_bera_test()   


def plot_chart_data(data):
    """This Function will plot interactive candlestick chart using the mpl finance libaray
    to see more details about how to use this library please refer https://github.com/matplotlib/mplfinance/tree/master"""
    try:
        required_columns = {"Open", "High", "Low", "Close", "Volume"}
        if set(data.columns) == required_columns:
            mpf.plot(data, type = "candlestick", volume = True, title = " TSLA Candlestick Chart")
        else:
            if required_columns.issubset(data.columns):
                data = data[list(required_columns)]
                mpf.plot(data, type = "candlestick", volume = True, title = " TSLA Candlestick Chart")           
    except Exception as err:
        print("The candlestick chart was not able to generare", err)
            
    
    
# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def calculate_log_returns(data):
    """Calculate log returns for each row in the dataframe."""
    data["log_returns"] = np.log(data.Close / data.Open)
    return data

def classify_candles(data):
    """Classify each candle as Bullish or Bearish based on Close and Open prices."""
    data["Candle_color"] = np.where(data["Close"] >= data["Open"], "Bullish", "Bearish")
    return data

def print_momentum_stats(momentum_data, momentum_type):
    """Print statistics for the given momentum type."""
    try:
        if momentum_data.empty:
            logging.warning(f"No data available for {momentum_type} momentum.")
            return

        mean_return = momentum_data["log_returns"].mean()
        highest_return = momentum_data["log_returns"].max()
        lowest_return = momentum_data["log_returns"].min()

        logging.info(f"{momentum_type.capitalize()} momentum stats: "
                     f"Mean return = {mean_return:.2f}, Highest return = {highest_return:.2f}, "
                     f"Lowest return = {lowest_return:.2f}")
        
        # Candle stats
        candle_stats = momentum_data["Candle_color"].value_counts()
        logging.info(f"{momentum_type.capitalize()} Momentum Candle stats:\n{candle_stats}")
    except Exception as e:
        logging.error(f"Error in printing {momentum_type} momentum stats: {e}")

def calculate_momentum(data):
    """Calculate the upward and downward momentum based on the 20-day SMA."""
    try:
        # Compute the 20-day simple moving average (SMA)
        data["sma"] = data["Close"].rolling(window=20).mean()
        data.dropna(inplace=True)  # Drop any rows with missing values

        # Split data into upward and downward momentum
        upwards_momentum = data[data["Close"] > data["sma"]].copy()
        downwards_momentum = data[data["Close"] < data["sma"]].copy()

        # Calculate log returns and classify candles for both momentum types
        upwards_momentum = classify_candles(calculate_log_returns(upwards_momentum))
        downwards_momentum = classify_candles(calculate_log_returns(downwards_momentum))

        # Print statistics for both upward and downward momentum
        print_momentum_stats(upwards_momentum, 'up')
        print_momentum_stats(downwards_momentum, 'down')

        # Detailed analysis for Bullish candles (to avoid redundancy)
        bullish_upwards = upwards_momentum[upwards_momentum["Candle_color"] == "Bullish"]
        bullish_downwards = downwards_momentum[downwards_momentum["Candle_color"] == "Bullish"]
        bearish_upwards = upwards_momentum[upwards_momentum["Candle_color"] == "Bearish"]
        bearish_downwards = downwards_momentum[downwards_momentum["Candle_color"] == "Bearish"]

        logging.info(f"Upwards Momentum Bullish Log Returns Max: {bullish_upwards['log_returns'].max():.2f}")
        logging.info(f"Downwards Momentum Bullish Log Returns Max: {bullish_downwards['log_returns'].max():.2f}")
        logging.info(f"Upwards Momentum Bearish Log Returns Max: {bearish_upwards['log_returns'].min():.2f}")
        logging.info(f"Downwards Momentum Bearish Log Returns Max: {bearish_downwards['log_returns'].min():.2f}")

    except Exception as e:
        logging.error(f"Error in calculating momentum: {e}")
        raise

# Example usage
if __name__ == "__main__":
    try:
        data = yf.download("TSLA", "2010-01-01", "2024-12-24", auto_adjust= True)
        data.columns = [ col[0] for col in data.columns.to_flat_index()]
        data.index = pd.to_datetime(data.index)
        plot_chart_data(data)
    except Exception as e:
        logging.critical(f"Failed to run the momentum calculation process: {e}")


    
        
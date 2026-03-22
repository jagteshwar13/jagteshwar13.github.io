""" finding whether  ccj(cameco) and (uec) can be used for statistical arbitrage
author : jagteshwar singh 
strategy : statistical arbitarge """

import numpy as np
import time
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.api import OLS
import warnings 
warnings.filterwarnings("ignore")
plt.style.use("dark_background")
# getting historical data from the instrumnets

start_time = time.time() # time.perf_counter() more precise
x = pd.read_csv(r"D:\stats_arbitrage_strategy\ccj_daily_historical-data-12-11-2024.csv", index_col = 0, skipfooter = 1, engine = "python")
y = pd.read_csv(r"D:\stats_arbitrage_strategy\uec_daily_historical-data-12-11-2024.csv", index_col = 0, skipfooter = 1, engine = "python")
x.index = pd.to_datetime(x.index)
y.index = pd.to_datetime(y.index)

for ticker in [x,y]:
    ticker.rename(columns = {"Last":"Close"}, inplace = True)
    ticker.drop(columns =["%Chg", "Volume", "Change", "Open", "High", "Low"], inplace  = True)
    ticker.Close.plot()
    plt.show()

end_time = time.time()
elapsed_time = end_time - start_time
print("The time taken to complete the task is %.4f seconds \n" % (elapsed_time))
    
df = pd.concat([x,y], axis = 1)
df.columns = ["ccj", "uec"]
df.index = pd.to_datetime(df.index)

# now we will be running the ols for finding the hedge ratio
model = OLS(df.ccj.iloc[:100], df.uec.iloc[:100])
model = model.fit()
df["spread"] = df.ccj- (model.params[0] * df.uec)
df.index = pd.to_datetime(df.index)
print(df.spread.plot())
plt.ylabel("spread")
plt.show()

""" so adf test outputs tuples - test statistic, p-value, used lags, number of observations, critical values and (aic / bic values if used)"""
adf = adfuller(df.spread, maxlag=1)
print("The t stat value is",adf[0])
print("Critical value for 99% is", adf[4]["1%"])

# Mean reversion strategy 

def reversion(data, lookback, std_dev):
    # moving average
    data["mvg_avg"] = data.spread.rolling(window = lookback).mean()
    data["mvg_std_dev"] = data.spread.rolling(window = lookback).std()
    # upper band and lower band
    data["upper"] = data.mvg_avg + std_dev* data.mvg_avg
    data["lower"] = data.mvg_avg - std_dev*data.mvg_avg
    
    # long entry condition
    data["long_entry"] = data.spread < data.lower
    data["long_exit"] = data.spread >= data.mvg_avg
    
    data["long_positions"] = np.nan
    data.loc[data.long_entry, "long_positions"] = 1
    data.loc[data.long_exit, "long_positions"] = 0
    data.long_positions = data.long_positions.fillna(method = "ffill")
    
    # short entry positions
    data["short_entry"] = data.spread > data.upper
    data["short_exit"] = data.spread <= data.mvg_avg
    data["short_positions"] = np.nan
    data.loc[data.short_entry, "short_positions"] = -1
    data.loc[data.short_exit, "short_positions"] = 0
    data["short_positions"] = data["short_positions"].fillna(method = "ffill")
    
    # Positions
    data["positions"] = data.long_positions+data.short_positions
    return data

print(reversion(df, 5, 2))

""" now we will compute the cumulative returns """
df["percentage_change"] = ((df.spread - df.spread.shift(1))/(model.params[0]*df.uec + df.ccj))
df["strategy_returns"] = df.positions.shift(1)*df.percentage_change
df["cumulative_returns"] = (df.strategy_returns+1).cumprod()
df.dropna(inplace = True)
# sharpe ratio
sharp_ = np.mean(df["strategy_returns"]/np.std(df["strategy_returns"]))*(252**0.5)
print("the sharpe ratio for the stratgey is:", sharp_)

# plot the cumulative results 
df.cumulative_returns.plot()
plt.show()

def drawdown_calculation(cumulate_returns):
    # calculate the running max
    running_max = np.maximum.accumulate(cumulate_returns.dropna())
    running_max[running_max<1] =1
    drawdown = cumulate_returns / running_max -1
    return drawdown

def drawdown_plot(drawdown):
    plt.figure(figsize = (12,7))
    drawdown.plot(color ="r")
    plt.ylabel("returns")
    plt.fill_between(drawdown.index, drawdown, color = "red")
    plt.grid(which="major", color = "k", linestyle = "-", linewidth =0.2)
    plt.show()
    
drawdown = drawdown_calculation(df.cumulative_returns)
print("The maximum drawdown is %.2f" % (drawdown.min()*100))
drawdown_plot(drawdown)

#df.to_csv("results.csv")
import os
print(os.getcwd())

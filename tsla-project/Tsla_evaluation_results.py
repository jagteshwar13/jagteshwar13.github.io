import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


file_name = r"C:\Users\jay\Documents\Tsla_Combined_data\Trades_Tsla.csv"
df = pd.read_csv(file_name)
df = df.drop(columns = ["Unnamed: 0"])
df['signal_time'] = pd.to_datetime(df['signal_time'])
df.index = df['signal_time'].dt.date  # Set index to the date part
print(df)

df[["stop_loss_hit_price","profit_taken_price", "close_price" ]] = df[["stop_loss_hit_price","profit_taken_price", "close_price" ]].fillna(0)
df["trade_exit_price"] = df["stop_loss_hit_price"]+df["profit_taken_price"]+df["close_price"]

df["return"] = (df["entry_price"]-df["trade_exit_price"])/ df["entry_price"]

print(df)

# how many were green trades and how many were red tardes

print((df["return"]>0).value_counts())

# True     62
# False    60
# Name: count, dtype: int64
# so 62 times return was positive and 60 times return was negative 

# total return of the strategy 

df["cum_returns"] = (1+df["return"]).cumprod()

plt.figure(figsize=(12,8))
plt.plot(df["cum_returns"], marker = "*")
plt.title("cumulative returns 2013-2024")
plt.xlabel("time")
plt.ylabel("cumulative returns")
plt.show()


print(df.cum_returns[-1])# returns
# total trades
print(len(df))
# net return per trade
print(df["return"].mean(), df["return"].min(), df["return"].max())
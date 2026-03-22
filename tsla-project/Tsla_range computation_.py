import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import datetime
import warnings
import pickle
import os 
pd.set_option("display.max_columns", None)
warnings.simplefilter("ignore")


file_path = r"C:\Users\jay\Documents\Tsla_Combined_data\Tsla_Combined_1_min.csv" # 1 min data for each year 
# Load the CSV data
data = pd.read_csv(file_path, index_col=0, parse_dates=True)

# Function to compute ranges
def range_computation(data):
    unique_dates = np.unique(data.index.date)
    valid_dates = []
    percentage_drops = []
    low_price = []
    high_price = []
    low_time = [] # first instance the condition was met 

    for unique_date in unique_dates:
        df = data[data.index.date == unique_date].copy() # analysing each day's 1 minute data 
        df_morning = df.between_time("09:30", "10:30") # window time of 1 hr for strategy requirements
        
        max_high = df_morning['High'].max()
        max_high_time = df_morning[df_morning['High'] == max_high].index[-1] # selecting the most recent high given if same highs were created 
        
        max_low = df_morning['Low'].min()
        max_low_time = df_morning[df_morning['Low'] == max_low].index[-1] # similarly selecting the most recent low
        
        if max_high_time < max_low_time:
            percentage_drop = (max_high - max_low) / max_high # Range between the high and low
            
            
            if percentage_drop > 0.04: # we want to see which is the most optimum drop so that we can choose the drop as basis of our long condition
                first_low_time = df_morning[df["Low"]<=max_high*(1-0.04)].index[0].time()
                valid_dates.append(unique_date)
                percentage_drops.append(percentage_drop)
                low_price.append(max_low)
                high_price.append(max_high)
                low_time.append(first_low_time) 
                

    return valid_dates, percentage_drops, low_price, high_price, low_time

# Call the function
valid_dates, percentage_drops,low_price, high_price, low_time = range_computation(data)
# Combine results into a DataFrame
results = pd.DataFrame({
    'Date': valid_dates,
    'Percentage Drop': percentage_drops,
    'High Price<10:30': high_price,
    'Low Price<10:30': low_price,
    'Low Time':low_time 

})
results.set_index('Date', inplace=True)
# Display the results
print(f"Number of days with > 4% drop where high is created before low: {len(results)}\n")
print(results)

# Plot the distribution of percentage drops
plt.hist(percentage_drops, bins=5, edgecolor='black')
plt.xlabel('Percentage Drop')
plt.ylabel('Frequency')
plt.title('Distribution of High-Low Percentage Drops (9:30 AM to 10:30 AM)')
plt.show()
#--------------------------------------------------------------------------------------
# Analysis of valid dates against daily data
def analysis_valid_dates(valid_dates):
    # Download daily data once, outside of the loop
    daily_data = yf.download("TSLA", start="2020-01-01", end="2020-12-31", auto_adjust=True)
    daily_data.index = pd.to_datetime(daily_data.index).date  # Ensure the index is in datetime.date format
    daily_data.columns = [col[0] for col in daily_data.columns.to_flat_index()]  # Flatten columns if necessary
    
    # Extract daily data for the valid dates
    try:
        daily_data_subset = daily_data.loc[valid_dates].copy()
        print("\n"*2)
        print(daily_data_subset)
    except KeyError as e:
        print(f"Error: Some dates are missing in the daily data: {e}")
    return daily_data_subset

# Call the analysis function
day_data = analysis_valid_dates(valid_dates)

# Next part is to compare statistics of the high - low between time frame 9:30 to 10:30 am with respect to the actual day high and lows
def combined_data(results, day_data):
    data_combined = pd.concat([results, day_data[["High","Low","Close"]]], axis = 1) 
    data_combined.index = pd.to_datetime(data_combined.index)
    return data_combined

df = combined_data(results, day_data)  
print("\n")     
print(df)
df[["Low Price<10:30", "Low", "Close"]].plot(marker = "o", linestyle = "None")
plt.show()

# destination = r"C:\Users\jay\Documents\Tsla_Combined_data"
# file_name = "Tsla_trade_data.csv"
# trade_data = os.path.join(destination, file_name)
# df.to_csv(trade_data)

    







    
    
























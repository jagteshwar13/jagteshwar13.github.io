import glob
import pandas as pd
import matplotlib.pyplot as plt
import os

# Get all CSV files in the specified folder
csv_files = glob.glob(r"C:\Users\jay\Documents\Tsla_yearly_minute_data\*.csv")

# Function to optimize data
def data_optimise(combined_data):
    # Convert the "Time" column to datetime (pandas can handle ISO8601 format automatically)
    combined_data["Time"] = pd.to_datetime(combined_data["Time"], errors='coerce')

    # Sort data by "Time" in ascending order
    combined_data = combined_data.sort_values(by="Time", ascending=True).reset_index(drop=True)

    # # Drop unnecessary columns
    # combined_data.drop(columns=["Change", "%Chg"], inplace=True)

    # # Rename "Last" column to "Close"
    # combined_data.rename(columns={"Last": "Close"}, inplace=True)

    # Set "Time" as the index
    combined_data = combined_data.set_index("Time")

    # Return the cleaned DataFrame
    return combined_data

# Read and concatenate all CSV files, then apply the optimization function to each
combined_data = pd.concat([pd.read_csv(file).iloc[:-1] for file in csv_files], ignore_index=True)

# Apply data_optimise function to the concatenated DataFrame
combined_data = data_optimise(combined_data)

# Print the cleaned and optimized DataFrame
print(combined_data)

# # some checks to see if there are nan values in the datset
# print(combined_data.isna().sum()) # 
output_directory = r"C:\Users\jay\Documents\Tsla_Combined_data"
file_name = "Tsla_Combined_1_min.csv"
output_file = os.path.join(output_directory, file_name)
combined_data.to_csv(output_file)



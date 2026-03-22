import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

"""
Portfolio Optimization using Monte Carlo Simulation
Based on Modern Portfolio Theory (CFA curriculum)
"""

# -----------------------------
# Load stock data from Excel
# -----------------------------
stock = ["TSLA", "UBER", "NVDA", "ORCL"]
data = pd.read_excel(r"D:\Data_python_work\Book1.xlsx", index_col="Date", parse_dates=True)
print("Stock prices:\n", data.head())

# -----------------------------
# Calculate daily returns
# -----------------------------
stock_ret = data.pct_change()
print("Daily returns (%):\n", (stock_ret*100).round(2).head())

# -----------------------------
# Calculate annualized mean returns and covariance
# -----------------------------
daily_returns = stock_ret.mean()
mean_returns = daily_returns * 252  # Annualized
cov_matrix = stock_ret.cov() * 252  # Annualized
print("\nAnnualized mean returns:\n", mean_returns)
print("\nAnnualized covariance matrix:\n", cov_matrix)

# -----------------------------
# Monte Carlo Simulation
# -----------------------------
num_iterations = 1000
simulation_res = np.zeros((3 + len(stock), num_iterations))  # rows: returns, stddv, Sharpe, weights

for i in range(num_iterations):
    weights = np.random.random(len(stock))
    weights /= np.sum(weights)  # normalize weights to sum 1
    
    # Portfolio return and standard deviation
    portfolio_return = np.sum(mean_returns * weights)
    portfolio_std_dev = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    
    # Store results
    simulation_res[0, i] = portfolio_return
    simulation_res[1, i] = portfolio_std_dev
    simulation_res[2, i] = portfolio_return / portfolio_std_dev  # Sharpe ratio
    simulation_res[3:, i] = weights

# -----------------------------
# Convert results to DataFrame
# -----------------------------
sim_frame = pd.DataFrame(simulation_res.T, columns=["returns", "stddv", "Sharpe_ratio"] + stock)
print("\nSimulated portfolios:\n", sim_frame.head(15))

# -----------------------------
# Identify key portfolios
# -----------------------------
# Maximum Sharpe ratio portfolio
max_sharpe = sim_frame.loc[sim_frame["Sharpe_ratio"].idxmax()]

# Minimum risk portfolio
min_std = sim_frame.loc[sim_frame["stddv"].idxmin()]

# Portfolio closest to target risk
target_sd = 0.30
best_portfolio = sim_frame.iloc[(sim_frame["stddv"] - target_sd).abs().argmin()]

print("\nPortfolio with max Sharpe ratio:\n", max_sharpe)
print("\nPortfolio with min risk:\n", min_std)
print(f"\nPortfolio closest to target SD ({target_sd*100:.1f}%):\n", best_portfolio)

# -----------------------------
# Plot Efficient Frontier
# -----------------------------
plt.figure(figsize=(10,6))
plt.scatter(sim_frame.stddv, sim_frame.returns, c=sim_frame.Sharpe_ratio, cmap="viridis", alpha=0.7)
plt.colorbar(label="Sharpe Ratio")
plt.xlabel("Standard Deviation")
plt.ylabel("Expected Return")
plt.title("Monte Carlo Portfolio Simulation")

# Highlight key portfolios
plt.scatter(max_sharpe.stddv, max_sharpe.returns, marker='*', color="r", s=200, label="Max Sharpe")
plt.scatter(min_std.stddv, min_std.returns, marker='*', color="b", s=200, label="Min Risk")
plt.scatter(best_portfolio.stddv, best_portfolio.returns, marker='*', color="g", s=200, label="Target Risk")

plt.legend()
plt.show()

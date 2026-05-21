import numpy as np
import pandas as pd
from Momentum_Trading.data import prices
from Momentum_Trading.module import Model
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

initial_capital = 100000
lookback = 60

model = Model(
    capital=initial_capital,
    clopri=prices,
    lookback=lookback
)

for date in prices.index:
    curr_index = prices.index.get_loc(date)

    # Skip dates before enough momentum history exists
    if curr_index < lookback:
        continue

    # First mark holdings to market before any rebalance
    model.calc_equity(date)

    # Rebalance if needed
    model.rebalance(date)

    # Recalculate equity after trades
    model.calc_equity(date)
    model.calc_weight(date)

    # Save portfolio snapshot
    model.update_portfolio(date)


results = pd.DataFrame(model.history)

if results.empty:
    raise ValueError("Backtest produced no results. Check your data and lookback window.")

results["Date"] = pd.to_datetime(results["Date"])
results = results.set_index("Date").sort_index()

# Compute returns from portfolio value history
results["Daily Return"] = results["Portfolio Value"].pct_change()

# Cumulative return relative to initial capital
results["Cumulative Return"] = results["Portfolio Value"] / initial_capital - 1

# Drawdown
results["Running Max"] = results["Portfolio Value"].cummax()
results["Drawdown"] = results["Portfolio Value"] / results["Running Max"] - 1

trading_days = 252

daily_returns = results["Daily Return"].dropna()

total_return = results["Portfolio Value"].iloc[-1] / initial_capital - 1

if len(results) > 1:
    annualized_return = (results["Portfolio Value"].iloc[-1] / initial_capital) ** (trading_days / len(results)) - 1
else:
    annualized_return = np.nan

annualized_vol = daily_returns.std() * np.sqrt(trading_days) if len(daily_returns) > 1 else np.nan

if annualized_vol and annualized_vol != 0:
    sharpe_ratio = annualized_return / annualized_vol
else:
    sharpe_ratio = np.nan

max_drawdown = results["Drawdown"].min()

print("========== BACKTEST RESULTS ==========")
print(f"Initial Capital:      ${initial_capital:,.2f}")
print(f"Final Portfolio Value:${results['Portfolio Value'].iloc[-1]:,.2f}")
print(f"Total Return:         {total_return:.2%}")
print(f"Annualized Return:    {annualized_return:.2%}")
print(f"Annualized Volatility:{annualized_vol:.2%}")
print(f"Sharpe Ratio:         {sharpe_ratio:.4f}")
print(f"Max Drawdown:         {max_drawdown:.2%}")

actions_df = pd.DataFrame(model.actions, columns=["Date", "Action", "Amount", "Stock"])
if not actions_df.empty:
    actions_df["Date"] = pd.to_datetime(actions_df["Date"])
    actions_df = actions_df.sort_values("Date")
    print("\n========== FIRST FEW TRADES ==========")
    print(actions_df.head(10))
else:
    print("\nNo trades were made.")


# Equity curve
plt.figure(figsize=(12, 6))
plt.plot(results.index, results["Portfolio Value"])
plt.title("Equity Curve")
plt.xlabel("Date")
plt.ylabel("Portfolio Value")
plt.grid(True)
plt.tight_layout()
plt.show()

# Cumulative return
plt.figure(figsize=(12, 6))
plt.plot(results.index, results["Cumulative Return"])
plt.title("Cumulative Return")
plt.xlabel("Date")
plt.ylabel("Cumulative Return")
plt.grid(True)
plt.tight_layout()
plt.show()

# Drawdown
plt.figure(figsize=(12, 5))
plt.plot(results.index, results["Drawdown"])
plt.title("Drawdown")
plt.xlabel("Date")
plt.ylabel("Drawdown")
plt.grid(True)
plt.tight_layout()
plt.show()

# Daily returns histogram
plt.figure(figsize=(10, 5))
plt.hist(daily_returns, bins=30)
plt.title("Distribution of Daily Returns")
plt.xlabel("Daily Return")
plt.ylabel("Frequency")
plt.grid(True)
plt.tight_layout()
plt.show()

# print(clopri.tail())

# momentum = mod.calc_momentum('2026-03-30')
# print(momentum)

# ranked = mod.ranking('2026-03-30')
# print(ranked)

# top = mod.select_top('2026-03-30')
# print(top)

# weight = mod.assign_weight('2026-03-30')
# print(weight)
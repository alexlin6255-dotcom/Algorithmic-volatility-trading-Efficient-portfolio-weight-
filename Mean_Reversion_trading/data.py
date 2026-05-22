import pandas as pd

data = pd.read_csv("Personal/Momentum_trading/Data/SPY.csv", index_col=0, header=[0, 1], parse_dates=True)

prices = data["Close"]

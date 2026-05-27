from data import prices
from module import Model

lookback = 20
mod = Model(100000, prices, lookback)

for date in prices.index:
    curr_index = prices.index.get_loc(date)
    if curr_index < 30:
        continue
    elif curr_index <100:
        print(mod.rolling_avg(date), "\n")
        print(mod.rolling_std(date), "\n")
        print(mod.correlation(date, "AAPL", "AMZN"), "\n")
from data import prices
from module import Model

lookback = 60
mod = Model(100000, prices, lookback)

for date in prices.index:
    curr_index = prices.index.get_loc(date)
    if curr_index < 70:
        continue
    elif curr_index <200:
        print(mod.rolling_avg(date), "\n")
        print(mod.rolling_std(date), "\n")
        print(mod.correlation(date, "AAPL", "AMZN"), "\n")
        # print(mod.hedge_ratio(date))
        print(mod.pick_pair(date))
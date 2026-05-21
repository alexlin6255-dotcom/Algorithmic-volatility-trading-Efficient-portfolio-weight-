This program using the concept of monetum trading to trade securities.
The program looks through all the securities, calculates their momentums and ranks them.
The best 2 securities will be picked to add to the portfolio, and will continued to be monitored.
Unfortunately, I have not implemented dynamic weighing, so the code in the mean time will assign all 2 of the securities with equal weight before executing trades on rebalance dates.
To remove any overselling, I have decided to assign the program rebalance days, where they are only allowed to readjust weights, sell or buy securities on that day.
Unfortunately the data in this file only imports 4 equities, but it can be easily scaled up to any number of securities if the user wishes to.
Furthermore, the program does not grab live data, and instead uses backtest data. However, this can be easily fixed by editing data.
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.stattools import adfuller


class Model:
    def __init__(self,capital,clopri,lookback):
        self.inicap = capital
        self.capital = capital
        self.price = clopri
        self.lookback = lookback
        self.portfolio_value = 0
        self.holdings = {}
        self.equity = 0
        self.history = []
        self.actions = []
        self.returns = []
        self.weight_history = []
        self.weight = {}
        self.returns = self.price.pct_change()

    def rolling_avg(self, date):
        curr_index = self.price.index.get_loc(date)
        avgs = dict()
        if curr_index >= self.lookback:
            for column in self.price.columns:
                avg = self.price[column].rolling(window=self.lookback).mean().loc[date]
                avgs[column] = avg
            return avgs
    def rolling_std(self,date):
        curr_index = self.price.index.get_loc(date)
        stds = dict()
        if curr_index >= self.lookback:
            for column in self.price.columns:
                std = self.price[column].rolling(window=self.lookback).std().loc[date] / 100
                stds[column] = std
            return stds
    def signal(self,date):
        curr_index = self.price.index.get_loc(date)
        signals = dict()
        stds = self.rolling_std(date)
        avgs = self.rolling_avg(date)
        for name in stds.keys():
            curr_price = self.price.iloc[curr_index][name]
            signal = (curr_price - avgs[name])/stds[name]
            signals[name] = signal
        return signals
    
    def covariance(self,date,stock1,stock2):
        curr_index = self.price.index.get_loc(date)
        avg1 = self.returns[stock1].iloc[curr_index - self.lookback : curr_index].mean() 
        avg2 = self.returns[stock2].iloc[curr_index - self.lookback : curr_index].mean() 
        sigma = 0
        for i in range(self.lookback):
            ret1 = self.returns[stock1].iloc[curr_index - i] 
            ret2 = self.returns[stock2].iloc[curr_index - i]
            sigma += ((ret1 - avg1) * (ret2 - avg2)) / 100
        covar = sigma / (self.lookback - 1)
        return covar
    
    def correlation(self,date,stock1,stock2):
        stds = self.rolling_std(date)
        correlation = self.covariance(date,stock1,stock2) / (stds[stock1] * stds[stock2])
        return correlation
    
    def hedge_ratio(self,date,stock1,stock2,window=120):
        curr_index = self.price.index.get_loc(date)
        if curr_index< window:
            raise ValueError(
                f"Not enough history before {date}"
            )
        prices1 = self.prices[stock1].iloc[curr_index - window : curr_index]
        prices2 = self.prices[stock2].iloc[curr_index - window : curr_index]

        if prices1.isnull().any() or prices2.isnull().any():
            raise ValueError(
                f"None type detected in the window"
            )
        
        X = prices2.values.reshape(-1,1)
        Y = prices1.values

        mod = LinearRegression().fit(X,Y)
        hedge_ratio = mod.coef_[0]
        intercept = mod.intercept_

        spread = prices1 - hedge_ratio * prices2
        adf_stat, adf_p = adfuller(spread)[0:2]

        return {
        "hedge_ratio" : hedge_ratio,
        "intercept"   : intercept,
        "spread"      : spread,
        "adf_stat"    : adf_stat,
        "adf_pvalue"  : adf_p,
        "stationary"  : adf_p < 0.05,
    }
        
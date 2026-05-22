import pandas as pd
import numpy as np
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
                std = self.price[column].rolling(window=self.lookback).std().loc[date]
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
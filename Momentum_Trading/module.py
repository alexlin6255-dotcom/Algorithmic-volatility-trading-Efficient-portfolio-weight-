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

    def calc_momentum(self,date):
        curr_index = self.price.index.get_loc(date) # This is a function to get the index

        if curr_index < self.lookback:
            return None
        
        elif curr_index >= self.lookback:
            past_index = curr_index - self.lookback
            momentum = dict()
            for column in self.price.columns:
                curr_price = self.price.iloc[curr_index][column]
                past_price = self.price.iloc[past_index][column]
                mom = (curr_price/past_price) - 1
                momentum[column] = mom
            return momentum
        
    def ranking(self,date):
        momentum = self.calc_momentum(date)
        if momentum is None:
            return None
        
        ranked = sorted(momentum.items(), key=lambda x: x[1], reverse=True)
        return ranked

    def select_top(self,date,n=2):
        ranked = self.ranking(date)
        top = []

        if ranked is None:
            return None
        else:
            for name in ranked:
                top.append(name)
            return top[:n]
        
    def assign_weight(self,date):
        selected = self.select_top(date)
        weights = {}
        if len(selected) == 0:
            return None
        else:
            weight = 1/len(selected)
            for name,momentum in selected:
                weights[name] = weight
            return weights
        
    def is_rebalance_date(self,date):
        curr_index = self.price.index.get_loc(date)
        if curr_index < self.lookback:
            return False
        elif curr_index%30 == 0:
            return True
        else:
            return False
        
    def calc_equity(self,date):
        curr_index = self.price.index.get_loc(date)
        equity = 0
        if len(self.holdings) > 0:
            for key,holding in self.holdings.items():
                curr_price = self.price.iloc[curr_index][key]
                equity += curr_price * holding
            self.equity = equity
            return equity


    def calc_weight(self,date):
        curr_index = self.price.index.get_loc(date)
        equity = self.calc_equity(date)
        if len(self.holdings) > 0:
            for key,holding in self.holdings.items():
                curr_price = self.price.iloc[curr_index][key]
                value = curr_price * holding
                weight = value/equity
                self.weight[key] = weight

    def rebalance(self,date):
        if self.is_rebalance_date(date) == True:
            curr_index = self.price.index.get_loc(date)
            target = self.assign_weight(date)
            sell_all = set(self.holdings.keys()).difference(set(target.keys()))
            for stock in sell_all:
                self.sell(stock,self.holdings[stock],date)
                del self.holdings[stock]
            for stock in target.keys():
                curr_price = self.price.iloc[curr_index][stock]
                target_value = (target[stock] * self.equity) + (target[stock] * self.capital)
                if stock in self.holdings.keys():
                    curr_value = self.holdings[stock] * curr_price
                    difference = target_value - curr_value
                    if difference < 0:
                        amount = abs(difference) // curr_price
                        self.sell(stock,amount,date)
                    elif difference > 0:
                        amount = difference // curr_price
                        self.buy(stock,amount,date)
                else:
                    amount = target_value // curr_price
                    self.buy(stock,amount,date)
            self.calc_weight(date)
            
    def sell(self,stock,amount,date):
        if stock in self.holdings.keys() and self.holdings[stock] >= amount:
            self.holdings[stock] -= amount
            self.capital += amount * self.price.loc[date, stock]
            self.actions.append([date,"sell",amount,stock])
        else:
            return "You do not own this stock"
    def buy(self,stock,amount,date):
        if self.capital >= amount * self.price.loc[date, stock]:
            if stock in self.holdings.keys():
                self.holdings[stock] += amount
                self.capital -= amount * self.price.loc[date, stock]
                self.actions.append([date,"buy",amount,stock])
            else:
                self.holdings[stock] = amount
                self.capital -= amount * self.price.loc[date, stock]
                self.actions.append([date,"buy",amount,stock])
        else:
            return "You don't have enough money or the stock is not in the index"

    def portfolio_return(self,date):
        curr_index = self.price.index.get_loc(date)
        if curr_index >= 1:
            past_index = self.price.index.get_loc(date) - 1
        equity = 0
        for key,holding in self.holdings.items():
            curr_price = self.price.iloc[past_index][key]
            equity += curr_price * holding
        ret = ((self.equity + self.capital) / (equity + self.capital) - 1) * 100
        return ret 
    
    def update_portfolio(self,date):
        self.portfolio_value = self.equity + self.capital
        self.weight_history.append([date,self.weight])
        self.returns.append([date,self.portfolio_return(date)])
        self.history.append({"Date": date, "Portfolio Value": self.portfolio_value})
    

    




# DATA SCIENCE
# - Algo
# - Data Structures
# - Data Visualization
# - Machine Learning
# - Data Collections(web crawlers, etc)


    








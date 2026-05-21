import pandas as pd

# ASK ARYO WHY ITS LIKE THIS
data = pd.read_csv("Personal/Momentum_trading/Data/SPY.csv", index_col=0, header=[0, 1], parse_dates=True)

prices = data["Close"]



# for i in range(col):
#     print(data.iloc[:,i])



# def extract(name):
#     list2 = []
#     for i in range(col):
#         if data.iloc[0,i] == name:
#             list1 = []
#             list1.append(data.iloc[0:,i])
#             list2.append(list1)
#     return 

    

            


# aapl = extract("AAPL")
# print(aapl.shape)


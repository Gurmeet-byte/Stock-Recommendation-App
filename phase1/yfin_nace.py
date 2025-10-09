import yfinance as yf
import pandas as pd 

data=yf.download("AAPL",start='2024-01-01',end='2025-10-10')

print(data.head())
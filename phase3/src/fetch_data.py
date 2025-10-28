import pandas as pd 
import numpy as np 
import yfinance as yf
import os 

def fetch_sp500_list():
    url='https://datahub.io/core/s-and-p-500-companies/r/constituents.csv'
    df=pd.read_csv(url)
    script_sir=os.path.dirname(os.path.abspath(__file__))
    data_dir=os.path.join(script_sir,"..",'data','raw')
    os.makedirs(data_dir,exist_ok=True)


    save_path=os.path.join(data_dir,'sp500_list.csv')
    df.to_csv(save_path,index=False)
    
    return df

def fetch_yahoo_data(symbols,limit=20):
    all_data=[]

    for i,symbols in enumerate(symbols[:limit]):
        try:
            print(f'Fetching the data for {i+1}/{limit}...')
            ticker=yf.Ticker(symbols)
            info=ticker.info
            hist=ticker.history(period='1y')

            data = {
                "Symbol": symbols,
                "CompanyName": info.get("longName"),
                "Sector": info.get("sector"),
                "MarketCap": info.get("marketCap"),
                "PE_Ratio": info.get("trailingPE"),
                "EPS": info.get("trailingEps"),
                "ROE": info.get("returnOnEquity"),
                "DebtToEquity": info.get("debtToEquity"),
                "Price": info.get("currentPrice"),
                "YearHigh": info.get("fiftyTwoWeekHigh"),
                "YearLow": info.get("fiftyTwoWeekLow"),
                "AvgVolume": info.get("averageVolume"),
            }
            all_data.append(data)
        except Exception as e:
            print(f"Error while fetching the data")

    df=pd.DataFrame(all_data)
    os.makedirs('data/raw',exist_ok=True)
    df.to_csv("data/raw/yahoo_fundamental_data.csv",index=False)
    print(f" Saved Yahoo data for {len(df)} companies")

    return df
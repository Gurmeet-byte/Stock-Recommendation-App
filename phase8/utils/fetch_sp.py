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


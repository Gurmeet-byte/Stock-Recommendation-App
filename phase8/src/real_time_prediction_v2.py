import pandas as pd 
import numpy as np 
import yfinance as yf 
import os 
import pickle as pkl 
from datetime import datetime
import requests
from io import StringIO

def load_model_and_scaler():
    model_path=r"C:\Users\abc\StockApp\phase8\models\stock_recommendor_model(2).pkl" #this is our newly trained model on extended featrues 
    scaler_path=r"C:\Users\abc\StockApp\phase8\models\scaler.pkl" #this is our scaler module which will handle the scaling of the data 

    if not os.path.exists(model_path): #if the path didn't exists 
        raise FileNotFoundError("Check The file path ")
    if not os.path.exists(scaler_path):
        raise FileNotFoundError("Check The file path ")
    
    with open(model_path,"rb") as f:
        model=pkl.load(f)

    with open(scaler_path,"rb") as f:
        scaler=pkl.load(f)

    return model,scaler
def live_fetch_data(symbols):
    all_data=[]
    for symbol in symbols:
        try:
            ticker=yf.Ticker(symbol)
            info=ticker.info
            hist=ticker.history(period='6mo')

            if hist.empty:
                print("No Historical Data Exists")
                continue

            volatility = hist["Close"].pct_change().std()

            stock_data = {
                "Symbol": symbol,
                "PE_Ratio": info.get("trailingPE", 0),
                "EPS": info.get("trailingEps", 0),
                "ROE": info.get("returnOnEquity", 0) * 100 if info.get("returnOnEquity") else 0,
                "DebtToEquity": info.get("debtToEquity", 0),
                "Price": info.get("currentPrice", 0),
                "YearHigh": info.get("fiftyTwoWeekHigh", 0),
                "YearLow": info.get("fiftyTwoWeekLow", 0),
                "AvgVolume": info.get("averageVolume", 0),
                "Volatality": volatility,
                "Current_price": info.get("currentPrice", 0),
                "Future_price": info.get("targetMeanPrice", 0),
                "Volatality_index": volatility,
                "Quality_Score": 0.7,
                "Growth_Score": 0.6,
                "Sector": info.get("sector", "Unknown")
            }
            all_data.append(stock_data)
        except Exception as e:
            print("There was an error ")
        
    df=pd.DataFrame(all_data)
    return df

def fetch_sp():
    url="https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                      "AppleWebKit/537.36 (KHTML, like Gecko) "
                      "Chrome/120.0 Safari/537.36"
    }
    response = requests.get(url, headers=headers)
    response.raise_for_status()
    df=pd.read_html((StringIO(response.text)))[0]
    return df['Symbol'].tolist()



def predcit_and_recommend(model,scaler,live_df,budget,sector_pref,horizon,target_stock):
    feature_cols = [
        "PE_Ratio", "EPS", "ROE", "DebtToEquity", "Price",
        "YearHigh", "YearLow", "AvgVolume", "Volatality",
        "Current_price", "Future_price", "Volatality_index",
        "Quality_Score", "Growth_Score"
    ]
    available_feaures=[f for f in feature_cols if f in live_df.columns]
    X_scaled=scaler.transform(live_df[available_feaures])
    preds=model.predict(X_scaled)
    live_df['Predicted_score']=preds
    df=live_df[live_df['Price']<=budget]

    if sector_pref.lower()!='any':
        df=df[df['Sector'].str.lower().str.contains(sector_pref,na=False)]

    
    if target_stock.upper() in df['Symbol'].values:
        target_sector=df.loc[df['Symbol']==target_stock.upper(),"Sector"].values[0]
        related_stock=df[df['Sector']==target_sector]
        df=pd.concat([related_stock,df]).drop_duplicates(subset="Symbol",keep='first')

    #Sort the values according to the horizon of the user we will sort  here by the Volatality 
    if horizon=='short':
        df=df.sort_values(by='Volatality',ascending=True)


    elif horizon=='long':
        df=df.sort_values(by="Growth_Score",ascending=False)

    return df.head(5)


if __name__=="__main__":
    model,scaler=load_model_and_scaler()
    budget=float(input("Enter the Budget for the Invesment: "))
    sector_pref=input("Enter the Sector Prefrence: ").strip().lower()
    horizon=input("Enter the Horizon of the Investment (long/short): ").strip().lower()
    target_stock=input("Enter the Target Stock:").strip().lower()

    symbols=[ 
        "NVDA", "MSFT", "AAPL", "GOOG", "GOOGL", "AMZN", "META", "AVGO", "TSLA",
        "NFLX", "PLTR", "COST", "ASML", "AMD", "CSCO", "AZN", "TMUS", "MU", "LIN",
        "PEP", "SHOP", "APP", "INTU", "AMAT", "LRCX", "PDD", "QCOM", "ARM", "INTC",
        "BKNG", "AMGN", "TXN", "ISRG", "GILD", "KLAC", "PANW", "ADBE", "HON",
        "CRWD", "CEG", "ADI", "ADP", "DASH", "CMCSA", "VRTX", "MELI", "SBUX",
        "CDNS", "ORLY", "SNPS", "MSTR", "MDLZ", "ABNB", "MRVL", "CTAS", "TRI",
        "MAR", "MNST", "CSX", "ADSK", "PYPL", "FTNT", "AEP", "WDAY", "REGN", "ROP",
        "NXPI", "DDOG", "AXON", "ROST", "IDXX", "EA", "PCAR", "FAST", "EXC", "TTWO",
        "XEL", "ZS", "PAYX", "WBD", "BKR", "CPRT", "CCEP", "FANG", "TEAM", "CHTR",
        "KDP", "MCHP", "GEHC", "VRSK", "CTSH", "CSGP", "KHC", "ODFL", "DXCM", "TTD",
        "ON", "BIIB", "LULU", "CDW", "GFS",'JNJ',"CPRI","AAL",""
    ]
    # symbols=fetch_sp()


    live_df=live_fetch_data(symbols)
    if live_df.empty:
        print("No Data is Avaialble ")
    else:
        recommendations=predcit_and_recommend(model,scaler,live_df,budget,sector_pref,horizon,target_stock)
        print("Here's The Recommended Stocks")
        print(recommendations[['Symbol',"Sector"]])


        output_dir=r'C:\Users\abc\StockApp\phase8\data\Recommendations'

        os.makedirs(output_dir,exist_ok=True)
        output_path=os.path.join(output_dir,f'Recommendations{datetime.today().date()}.csv')
        recommendations.to_csv(output_path,index=False)
        print(f"The Recommendations Were Saved at {r'C:\Users\abc\StockApp\phase8\data\Recommendations'} ")


        #Consumer Cyclical,Industrials



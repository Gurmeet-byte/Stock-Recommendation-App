import pandas as pd
from sklearn.preprocessing import StandardScaler
import os 


def preprocess_yahoo_data(raw_file=r'C:\Users\abc\StockApp\phase3\data\raw\yahoo_fundamental_data.csv'):
    if not os.path.exists(raw_file):
        print("File Did'nt find")

    df=pd.read_csv(raw_file)
    print("File is loaded")
    df=df.drop_duplicates(subset=['Symbol'])
    df = df.dropna(subset=["PE_Ratio", "EPS", "ROE", "DebtToEquity", "Price"])
    df.replace([float("inf"), float("-inf")], pd.NA, inplace=True)
    df.dropna(inplace=True)
    numeric_cols = ["PE_Ratio", "EPS", "ROE", "DebtToEquity", "Price", "YearHigh", "YearLow", "AvgVolume"]
    numeric_cols=[col for col in numeric_cols if col in df.columns]
    scaler=StandardScaler()
    df[numeric_cols]=scaler.fit_transform(df[numeric_cols])
    df['Volatality']=(df['YearHigh']-df['YearLow'])/df['Price']
    ordered_cols=['Symbol','CompanyName',"Sector"]+numeric_cols+['Volatality']
    df=df[ordered_cols]
    os.makedirs("data/preprocessed",exist_ok=True)
    output_path=r'C:\Users\abc\StockApp\phase3\data\preprocessed/cleaned_data.csv'
    df=df.to_csv(output_path,index=False)
    print("The Task was done")
    return df



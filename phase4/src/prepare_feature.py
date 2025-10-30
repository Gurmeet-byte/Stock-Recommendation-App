import pandas as pd
import os 
from sklearn.preprocessing import StandardScaler

def prepare_features():
    input_path=os.path.join("phase3","data","processed","merged_data.csv")
    output_dir=os.path.join("phase4","data","processed")
    os.makedirs(output_dir,exist_ok=True)
    output_path=os.path.join(output_dir,'model_ready_data.csv')

    df=pd.read_csv(input_path)
    print("Loaded Merged Dataset")


    df=df.drop_duplicates()
    if "Forward_Return" not in df.columns:
        raise ValueError("'Forward_Return' column missing. Check merge_data output.")
    
    df=df.dropna(subset=['Forward_Return'])
    print("The Data is Cleaned")

    numeric_cols = ["PE_Ratio", "EPS", "ROE", "DebtToEquity", "Price", "YearHigh", "YearLow", "AvgVolume"]
    numeric_cols=[col for col in numeric_cols if col in df.columns]

    scaler=StandardScaler()
    df[numeric_cols]=scaler.fit_transform(df[numeric_cols])

    if "YearHigh" in df.columns and "YearLow" in df.columns and "Price":
        df["Volatality_index"]=(df["YearHigh"]-df["YearLow"])/df["Price"]

    if "ROE" in df.columns and "DebtToEquity" in df.columns:
        df["Quality_Score"]=(df["ROE"]-df["DebtToEquity"]*0.1)    

    if "EPS" in df.columns and "ROE" in df.columns:
        df["Growth_Score"]=(df["EPS"]*df["ROE"])


    df=df.dropna()
    df.to_csv(output_path,index=False)
    print("The Model_ready_data is saved")

    return df

if __name__=="__main__":
    print(prepare_features())



    

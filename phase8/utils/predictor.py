import os
import pickle as pkl
import pandas as pd

def load_model():
    base_dir=os.path.dirname(os.path.abspath(__file__))
    project_root=os.path.dirname(base_dir)
    model_path=os.path.join(project_root,"models","Stock_predictor.pkl")
    if  not os.path.exists(model_path):
        raise FileNotFoundError("The file Didn't exists")
    
    with open(model_path,"rb") as f:
        model=pkl.load(f)
    return model
def predict(model, live_df):
    feature_cols = [
        "PE_Ratio", "EPS", "ROE", "DebtToEquity", "Price",
        "YearHigh", "YearLow", "AvgVolume", "Volatality",
        "Current_price", "Future_price", "Volatality_index",
        "Quality_Score", "Growth_Score"
    ]
    available = [c for c in feature_cols if c in live_df.columns]
    expected_features = ['PE_Ratio', 'EPS', 'ROE', 'DebtToEquity', 'Price', 'Volatality']
    live_df = live_df[[col for col in expected_features if col in live_df.columns]]

    live_df["Predicted_Score"] = model.predict(live_df)

    return live_df.sort_values(by="Predicted_Score", ascending=False)

import pandas as pd
import numpy as np 
import os 
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
import pickle 


def train_ml_model(input_file=r'C:\Users\abc\StockApp\phase3\data\preprocessed\cleaned_data.csv',model_dir=r'C:\Users\abc\StockApp\phase3\model'):

    if not os.path.exists(input_file):
        return FileNotFoundError
    

    df=pd.read_csv(input_file)
    print("The Dataset is loading")

    feature_cols=["PE_Ratio", "EPS", "ROE", "DebtToEquity", "Price", "Volatality"]
    available_features=[col for col in feature_cols if col in df.columns]

    df['target_score']=(
        0.3 * (1 - df["PE_Ratio"]) +
        0.3 * df["ROE"] +
        0.2 * df["EPS"] +
        0.2 * (1 - df["Volatality"])
    )
    X=df[available_features]
    Y=df['target_score']


    X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=42)

    model = XGBRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )
    print("Traning Models")
    model.fit(X_train,Y_train)


    preds=model.predict(X_test)
    mse=mean_squared_error(Y_test,preds)
    r2=r2_score(Y_test,preds)

    print("Model Evaluation ")
    print(f"MSE Score = {mse}")
    print(f"R2_Score = {r2}")

    cv_score=cross_val_score(model,X,Y,cv=5,scoring='r2')

    os.makedirs(model_dir,exist_ok=True)
    model_path=os.path.join(model_dir,"stock_recommendor_model.pkl")
    with open(model_path,"wb") as f:
        pickle.dump(model,f)

    print("Model is Saved")

    return model,X.columns

def score_stocks(input_file=r"C:\Users\abc\StockApp\phase3\data\preprocessed\cleaned_data.csv",
                 model_path=r"C:\Users\abc\StockApp\phase3\model\stock_recommendor_model.pkl",
                 top_n=10):
   
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"❌ Model not found at {model_path}. Run train_model.py first.")

    if not os.path.exists(input_file):
        raise FileNotFoundError(f"❌ Input file not found: {input_file}")

  
    df = pd.read_csv(input_file)
    with open(model_path,"rb") as f:
        model=pickle.load(f)

    feature_cols = ["PE_Ratio", "EPS", "ROE", "DebtToEquity", "Price", "Volatality"]
    available_features = [col for col in feature_cols if col in df.columns]

    if not available_features:
        raise ValueError("❌ No valid feature columns found in dataset.")


    df["Predicted_Return"] = model.predict(df[available_features])


    recommendations = df.sort_values(by="Predicted_Return", ascending=False)

    top_recommendations = recommendations[["Symbol", "Predicted_Return"]].head(top_n)

    print(f"✅ Generated top {top_n} stock recommendations.")
    return top_recommendations

print(train_ml_model())

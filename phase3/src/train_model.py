import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
import pickle


def train_ml_model(
    input_file=r"C:\Users\abc\StockApp\phase3\data\preprocessed\model_ready_data.csv",
    model_dir=r"C:\Users\abc\StockApp\phase3\model"
):
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"❌ Input file not found: {input_file}")

    df = pd.read_csv(input_file)
    print(" Dataset loaded successfully.")

   
    required_cols = [
        "PE_Ratio", "EPS", "ROE", "DebtToEquity", "Price",
        "YearHigh", "YearLow", "AvgVolume", "Volatality",
        "Current_price", "Future_price", "Volatality_index",
        "Quality_Score", "Growth_Score"
    ]

    available_features = [col for col in required_cols if col in df.columns]
    missing = set(required_cols) - set(available_features)
    if missing:
        print(f" Missing columns ignored: {missing}")

   
    if "Future_price" in df.columns and "Current_price" in df.columns:
        df["Target"] = df["Future_price"] / df["Current_price"]
    else:
        raise ValueError(" Missing columns for target creation: 'Future_price' or 'Current_price'")

  
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=available_features + ["Target"])
    print(f" Cleaned data shape: {df.shape}")

    X = df[available_features]
    y = df["Target"]

   
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

   
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)


    model = XGBRegressor(
        n_estimators=400,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )

    print(" Training the model...")
    model.fit(X_train, y_train)

    
    preds = model.predict(X_test)
    mse = mean_squared_error(y_test, preds)
    r2 = r2_score(y_test, preds)

    print(" Model Evaluation:")
    print(f"  - MSE: {mse:.6f}")
    print(f"  - R² Score: {r2:.4f}")

    cv_score = cross_val_score(model, X_scaled, y, cv=5, scoring="r2")
    print(f"  - Cross-validation R²: {cv_score.mean():.4f}")

    
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "stock_recommendor_model(2).pkl")
    scaler_path = os.path.join(model_dir, "scaler.pkl")

    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)

    print(f" Model and scaler saved successfully to: {model_dir}")

    return model, scaler, X.columns


def score_stocks(
    input_file=r"C:\Users\abc\StockApp\phase3\data\preprocessed\model_ready_data.csv",
    model_path=r"C:\Users\abc\StockApp\phase3\model\stock_recommendor_model.pkl",
    scaler_path=r"C:\Users\abc\StockApp\phase3\model\scaler.pkl",
    top_n=10
):

    if not os.path.exists(model_path):
        raise FileNotFoundError(f" Model not found at {model_path}. Run train_model.py first.")
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f" Scaler not found at {scaler_path}.")
    if not os.path.exists(input_file):
        raise FileNotFoundError(f" Input file not found: {input_file}")

    df = pd.read_csv(input_file)
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)

    feature_cols = [
        "PE_Ratio", "EPS", "ROE", "DebtToEquity", "Price",
        "YearHigh", "YearLow", "AvgVolume", "Volatality",
        "Current_price", "Future_price", "Volatality_index",
        "Quality_Score", "Growth_Score"
    ]
    available_features = [col for col in feature_cols if col in df.columns]

    X_scaled = scaler.transform(df[available_features])
    df["Predicted_Return"] = model.predict(X_scaled)

    recommendations = df.sort_values(by="Predicted_Return", ascending=False)
    top_recommendations = recommendations[["Symbol", "Predicted_Return"]].head(top_n)

    print(f" Generated top {top_n} stock recommendations.")
    return top_recommendations



if __name__ == "__main__":
    train_ml_model()

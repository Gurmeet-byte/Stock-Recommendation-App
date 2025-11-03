import os
import pandas as pd
from phase3.src.fetch_data import fetch_sp500_list, fetch_yahoo_data
from phase3.src.preprocess_data import preprocess_yahoo_data
from phase3.src.train_model import train_ml_model
from phase3.src.fetch_historical_prices import fetch_historical_data

def main():
    print("Fetching S&P 500 list...")
    sp500df = fetch_sp500_list()
    symbols = sp500df["Symbol"].tolist()

    print("Fetching Yahoo Finance data...")
    yahoo_df = fetch_yahoo_data(symbols, limit=100)
    
    print("Preprocessing data...")
    cleaned_df = preprocess_yahoo_data(
        r"C:\Users\abc\StockApp\phase3\data\raw\yahoo_fundamental_data.csv"
    )

    print("Fetching historical prices...")
    price_df = fetch_historical_data(symbols, months_ahead=6)

    print("Training model...")
    model, features = train_ml_model(
        r"C:\Users\abc\StockApp\phase3\data\preprocessed\cleaned_data.csv"
    )

    os.makedirs("phase8/models", exist_ok=True)
    import pickle
    with open("phase8/models/Stock_predictor.pkl", "wb") as f:
        pickle.dump(model, f)

    print("✅ Pipeline complete. Model saved to phase8/models/Stock_predictor.pkl")

if __name__ == "__main__":
    main()

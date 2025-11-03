import pandas as pd
import yfinance as yf
import os
from datetime import datetime
import pickle as pkl

def load_model():
    output_path = os.path.join("phase5", "models", "Stock_predictor.pkl")
    if not os.path.exists(output_path):
        raise FileNotFoundError("The file was not found")

    with open(output_path, "rb") as f:
        model = pkl.load(f)

    return model


def fetch_live_data(symbols):
    all_data = []
    for symbol in (symbols if isinstance(symbols[0], str) else symbols[0]):
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info

            hist = ticker.history(period='6mo')
            if hist.empty:
                print(f"No historical data available for {symbol}")
                continue

            volatility = hist["Close"].pct_change().std()

            sector = info.get("sector")
            if not sector:
                try:
                    sector = ticker.fast_info.get("sector", "Unknown")
                except Exception:
                    sector = "Unknown"

            stock_data = {
                "Symbol": symbol,
                "Sector": sector if sector else "Unknown",
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
                "Growth_Score": 0.6
            }

            all_data.append(stock_data)

        except Exception as e:
            print(f"Error fetching {symbol}: {e}")

    df = pd.DataFrame(all_data)
    print(" The Data is Downloaded")

    if "Sector" not in df.columns:
        df["Sector"] = "Unknown"

    return df


def predict(model, live_df):
    feature_cols = [
        "PE_Ratio", "EPS", "ROE", "DebtToEquity", "Price",
        "YearHigh", "YearLow", "AvgVolume", "Volatality",
        "Current_price", "Future_price", "Volatality_index",
        "Quality_Score", "Growth_Score"
    ]
    available_features = [f for f in feature_cols if f in live_df.columns]
    preds = model.predict(live_df[available_features])
    live_df['Predicted_Score'] = preds
    return live_df.sort_values(by='Predicted_Score', ascending=False)


def filter_by_user_preferences(df, budget, horizon, sector_pref, target_stock=None):
    df = df[df["Price"] <= budget]

    if sector_pref and sector_pref.lower() != "any":
        df = df[df["Sector"].str.lower().str.contains(sector_pref.lower(), na=False)]

    if horizon == "short":
        df = df.sort_values(by="Volatality", ascending=True)
    elif horizon == "long":
        df = df.sort_values(by="Growth_Score", ascending=False)

    if target_stock:
        target_ticker = yf.Ticker(target_stock)
        try:
            target_sector = target_ticker.info.get("sector", "").lower()
            if target_sector:
                df["similarity_boost"] = df["Sector"].str.lower().apply(lambda x: 1 if target_sector in x else 0)
                df["Predicted_Score"] += df["similarity_boost"] * 0.1
        except Exception as e:
            print(f"Could not fetch target stock sector: {e}")

    if df.empty:
        print(" No stocks match your criteria. Try 'any' for sector or raise your budget.")

    return df


if __name__ == "__main__":
    model = load_model()

    budget = float(input("Enter your budget per stock: "))
    horizon = input("Investment horizon (short/long): ").strip().lower()
    sector_pref = input("Preferred sector (or type 'any'): ").strip()
    target_stock = input("Target stock (optional, e.g. NVDA): ").strip() or None

    symbols = ["AAPL", "MSFT", "GOOGL", "TSLA", "AMZN", "NVDA", "META", "NFLX"]

    live_df = fetch_live_data(symbols)
    if live_df.empty:
        print("No live data fetched. Aborting.")
    else:
        preds_df = predict(model, live_df)
        filtered_df = filter_by_user_preferences(preds_df, budget, horizon, sector_pref, target_stock)

        if not filtered_df.empty:
            top_recommendations = filtered_df.head(5)
            print(" Top 5 Personalized Stock Recommendations:")
            cols_to_show = [col for col in ["Symbol", "Sector", "Price", "Predicted_Score"] if col in top_recommendations.columns]
            print(top_recommendations[cols_to_show])

            output_path = os.path.join("phase7", "data", "output", f"personalized_{datetime.today().date()}.csv")
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            top_recommendations.to_csv(output_path, index=False)
            print(f" Saved recommendations → {output_path}")

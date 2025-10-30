import os
import pandas as pd
import pickle
from sklearn.metrics.pairwise import cosine_similarity

def load_model():
    model_path = os.path.join("phase5", "models", "Stock_predictor.pkl")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"❌ Model not found at {model_path}")
    
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    print(" Model loaded successfully")
    return model


def get_user_input():
    print("\n📋 Enter your investment preferences:\n")

    try:
        budget = float(input(" Enter your budget per stock (e.g., 1500): ").strip())
    except ValueError:
        budget = None
        print(" Invalid input, skipping budget filter.")

    horizon = input(" Investment horizon (short-term / long-term): ").strip().lower()
    if horizon not in ["short-term", "long-term"]:
        horizon = "long-term"
        print(" Invalid input, defaulting to 'long-term'.")

    sector = input(" Preferred sector (optional, press Enter to skip): ").strip()
    if sector == "":
        sector = None

    target = input(" Target stock symbol (optional, press Enter to skip): ").strip().upper()
    if target == "":
        target = None

    return {
        "budget": budget,
        "investment_horizon": horizon,
        "sector_preference": sector,
        "target_stock": target
    }


def personalize_recommendations(model, input_file, user_input, top_n=10):
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"❌ Input file not found: {input_file}")

   
    df = pd.read_csv(input_file)
    print(f"📊 Loaded dataset with {len(df)} rows.")

    feature_cols = [
        "PE_Ratio", "EPS", "ROE", "DebtToEquity", "Price",
        "YearHigh", "YearLow", "AvgVolume", "Volatality",
        "Current_price", "Future_price", "Volatality_index",
        "Quality_Score", "Growth_Score"
    ]
    
    available_features = [col for col in feature_cols if col in df.columns]
    missing_features = [col for col in feature_cols if col not in df.columns]
    if missing_features:
        print(f" Warning: Missing features in dataset: {missing_features}")

    df["Predicted_Score"] = model.predict(df[available_features])

    if user_input.get("budget"):
        df = df[df["Price"] <= user_input["budget"]]
        print(f" Filtered stocks under ₹{user_input['budget']} → {len(df)} left")

    if user_input.get("sector_preference"):
        df = df[df["Sector"].str.contains(user_input["sector_preference"], case=False, na=False)]
        print(f" Filtered by sector '{user_input['sector_preference']}' → {len(df)} left")

    if user_input.get("investment_horizon") == "long-term":
        df["Personalized_Score"] = (
            0.5 * df["Quality_Score"] +
            0.3 * df["Growth_Score"] +
            0.2 * df["Predicted_Score"]
        )
    else:
        df["Personalized_Score"] = (
            0.5 * (1 - df["Volatality"]) +
            0.3 * df["Predicted_Score"] +
            0.2 * df["Growth_Score"]
        )

    if user_input.get("target_stock") and user_input["target_stock"] in df["Symbol"].values:
        target_vector = df[df["Symbol"] == user_input["target_stock"]][available_features].values
        df["Similarity_to_Target"] = cosine_similarity(df[available_features], target_vector).flatten()
        df["Final_Score"] = 0.7 * df["Personalized_Score"] + 0.3 * df["Similarity_to_Target"]
        print(f" Applied similarity matching to {user_input['target_stock']}")
    else:
        df["Final_Score"] = df["Personalized_Score"]

    recommendations = df.sort_values(by="Final_Score", ascending=False).head(top_n)
    print(f"\n Top {top_n} Personalized Recommendations:\n")
    print(recommendations[["Symbol", "Sector", "Price", "Final_Score"]].to_string(index=False))

    
    output_path = os.path.join("phase6", "data", "output", "personalized_recommendations.csv")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    recommendations.to_csv(output_path, index=False)
    print(f" Saved results to: {output_path}")

    return recommendations


if __name__ == "__main__":
    
    model = load_model()

   
    input_file = os.path.join("phase4", "data", "processed", "model_ready_data.csv")

    user_input = get_user_input()

    personalize_recommendations(model, input_file, user_input, top_n=10)

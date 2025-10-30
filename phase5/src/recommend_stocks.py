import pandas as pd 
import os 
import pickle  as pkl

def load_model():
    model_path=os.path.join("phase5","models","Stock_predictor.pkl")
    if not os.path.exists(model_path):
        return FileNotFoundError
    with open(model_path,"rb") as f:
        model=pkl.load(f)

    print("The Model is Loaded ")
    return model

def recommend_stock(model,input_file,top_n=10):
    if not os.path.exists(input_file):
        return FileNotFoundError
    
    df=pd.read_csv(input_file)
    print(f" Loaded dataset with {len(df)} rows.")

    feature_cols = [
        "PE_Ratio", "EPS", "ROE", "DebtToEquity", "Price",
        "YearHigh", "YearLow", "AvgVolume", "Volatality",
        "Current_price", "Future_price", "Volatality_index",
        "Quality_Score", "Growth_Score"
    ]

    available_features = [col for col in feature_cols if col in df.columns]
    missing_features = [col for col in feature_cols if col not in df.columns]

    if missing_features:
        print(f"⚠️ Warning: Missing features in dataset: {missing_features}")

    df["Predicted_Score"] = model.predict(df[available_features])

    recommendations = df.sort_values(by="Predicted_Score", ascending=False)

    top_recommendations = recommendations[["Symbol", "Predicted_Score"]].head(top_n)

    print(f"\n✅ Top {top_n} Recommended Stocks:")
    print(top_recommendations)

    output_path = os.path.join("phase5", "data", "output", "stock_recommendations.csv")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    top_recommendations.to_csv(output_path, index=False)
    print(f"\n💾 Recommendations saved to: {output_path}")

    return top_recommendations


if __name__ == "__main__":
    model = load_model()
    input_file = os.path.join("phase4", "data", "processed", "model_ready_data.csv")
    recommend_stock(model, input_file, top_n=10)
    print(f"The model type is {type(model)}")
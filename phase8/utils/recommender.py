import yfinance as yf

def filter_recommendations(df, budget, horizon, sector_pref, target_stock=None):
    df = df[df["Price"] <= budget]

    if sector_pref and sector_pref.lower() != "any":
        df = df[df["Sector"].str.lower().str.contains(sector_pref.lower(), na=False)]

    if horizon == "short":
        df = df.sort_values(by="Volatality", ascending=True)
    elif horizon == "long":
        df = df.sort_values(by="Growth_Score", ascending=False)

    if target_stock:
        try:
            target_sector = yf.Ticker(target_stock).info.get("sector", "").lower()
            if target_sector:
                df["similarity_boost"] = df["Sector"].str.lower().apply(
                    lambda x: 1 if target_sector in x else 0
                )
                df["Predicted_Score"] += df["similarity_boost"] * 0.1
        except Exception as e:
            print(f"Could not fetch target stock: {e}")

    return df

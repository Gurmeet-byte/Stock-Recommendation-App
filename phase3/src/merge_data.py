import pandas as pd
import os

def merge_datasets(
    fundamentals_path=r"C:\Users\abc\StockApp\phase3\data\preprocessed\cleaned_data.csv",
    prices_path=r"C:\Users\abc\StockApp\phase3\data\raw\historical_prices.csv",
    output_file=r"C:\Users\abc\StockApp\phase3\data\processed\merged_data.csv"
):
    print("🔗 Merging fundamental + historical data...")

    if not os.path.exists(fundamentals_path):
        raise FileNotFoundError(f" Fundamentals file not found: {fundamentals_path}")
    if not os.path.exists(prices_path):
        raise FileNotFoundError(f" Prices file not found: {prices_path}")

    fundamentals_df = pd.read_csv(fundamentals_path)
    prices_df = pd.read_csv(prices_path)

    print(f" Fundamentals: {fundamentals_df.shape[0]} rows, {fundamentals_df.shape[1]} columns")
    print(f" Prices: {prices_df.shape[0]} rows, {prices_df.shape[1]} columns")

    fundamentals_df['Symbol'] = fundamentals_df['Symbol'].str.upper()
    prices_df['Symbol'] = prices_df['Symbol'].str.upper()

    merged_df = pd.merge(fundamentals_df, prices_df, on='Symbol', how='inner')

    print(f" Merged dataset: {merged_df.shape[0]} rows, {merged_df.shape[1]} columns")

    merged_df.drop_duplicates(subset='Symbol', inplace=True)
    merged_df.dropna(subset=['Forward_Return'], inplace=True)

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    merged_df.to_csv(output_file, index=False)

    print(f" Saved merged dataset → {output_file}")
    return merged_df



if __name__ == "__main__":
    df = merge_datasets()
    print(df.head())

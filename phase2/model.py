import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
import warnings
import concurrent.futures
from typing import Optional

warnings.filterwarnings('ignore')

# Globals
scaler = StandardScaler()
company_to_ticker = {}

# ----------------------------
# Utility: company to ticker
# ----------------------------
def company_ticker_symbol():
    mapping = {
        # Technology
        'apple': 'AAPL', 'microsoft': 'MSFT', 'google': 'GOOGL', 'alphabet': 'GOOGL',
        'amazon': 'AMZN', 'tesla': 'TSLA', 'meta': 'META', 'facebook': 'META',
        'nvidia': 'NVDA', 'intel': 'INTC', 'amd': 'AMD', 'qualcomm': 'QCOM',
        'cisco': 'CSCO', 'oracle': 'ORCL', 'ibm': 'IBM', 'salesforce': 'CRM',
        'adobe': 'ADBE', 'netflix': 'NFLX', 'paypal': 'PYPL', 'square': 'SQ',
        # Healthcare
        'johnson&johnson': 'JNJ', 'johnson & johnson': 'JNJ', 'pfizer': 'PFE', 'merck': 'MRK', 'abbott': 'ABT',
        'thermo fisher': 'TMO', 'eli lilly': 'LLY', 'unitedhealth': 'UNH',
        'abbvie': 'ABBV', 'bristol myers': 'BMY', 'gilead': 'GILD', 'amgen': 'AMGN',
        'moderna': 'MRNA', 'biogen': 'BIIB', 'regeneron': 'REGN', 'vertex': 'VRTX',
        # Financial Services
        'jpmorgan': 'JPM', 'bank of america': 'BAC', 'wells fargo': 'WFC',
        'goldman sachs': 'GS', 'morgan stanley': 'MS', 'blackrock': 'BLK',
        'visa': 'V', 'mastercard': 'MA', 'american express': 'AXP',
        'charles schwab': 'SCHW', 'paypal': 'PYPL',
        # Consumer
        'walmart': 'WMT', 'target': 'TGT', 'costco': 'COST', 'home depot': 'HD',
        'lowes': 'LOW', 'mcdonalds': 'MCD', 'starbucks': 'SBUX', 'coca cola': 'KO',
        'pepsi': 'PEP', 'procter gamble': 'PG', 'unilever': 'UL', 'nestle': 'NSRGY',
        'nike': 'NKE', 'lululemon': 'LULU', 'disney': 'DIS', 'netflix': 'NFLX',
        # Energy & Industrial
        'exxon': 'XOM', 'chevron': 'CVX', 'conocophillips': 'COP', 'shell': 'SHEL',
        'next era energy': 'NEE', 'duke energy': 'DUK', 'southern company': 'SO',
        'boeing': 'BA', 'lockheed martin': 'LMT', 'raytheon': 'RTX', 'general electric': 'GE'
    }

    # Add small variations
    variations = {}
    for company, ticker in mapping.items():
        variations[company.replace(" ", "")] = ticker
        variations[company.replace("&", "and")] = ticker
        variations[company.replace("corporation", "corp")] = ticker
        variations[company.replace("incorporated", "inc")] = ticker
    mapping.update(variations)
    return mapping

# ----------------------------
# Convert user string -> ticker
# ----------------------------
def ticker_to_name(company_name: Optional[str]) -> Optional[str]:
    global company_to_ticker
    if not company_to_ticker:
        company_to_ticker = company_ticker_symbol()

    # If user skipped, return None
    if company_name is None:
        return None

    clean_name = company_name.lower().strip()
    if clean_name in company_to_ticker:
        return company_to_ticker[clean_name]

    # Try direct ticker-like input
    if len(clean_name) <= 5 and clean_name.isalpha():
        return clean_name.upper()

    # Try yfinance lookup - defensive
    try:
        yf_t = yf.Ticker(clean_name.upper())
        info = yf_t.info
        if info and 'symbol' in info:
            return info['symbol']
    except Exception:
        pass
    

    # fuzzy-ish match
    for company, ticker in company_to_ticker.items():
        if clean_name in company or company in clean_name:
            return ticker

    return company_name.upper()

# ----------------------------
# Download single stock profile (defensive)
# ----------------------------
def download_single_stock_data(ticker: str) -> Optional[dict]:
    try:
        if ticker is None:
            return None
        ticker = str(ticker).upper()
        # Get info (metadata)
        stock = yf.Ticker(ticker)
        info = stock.info or {}
        # If no info, treat as missing/delisted
        if not info:
            return None

        # Try to fetch history; use simple yf download for reliability
        try:
            hist = stock.history(period='6mo')
        except Exception:
            hist = pd.DataFrame()

        if hist is None or hist.empty or len(hist) < 10:
            # Not enough price data; still return profile but with missing price
            current_price = np.nan
            returns = pd.Series(dtype=float)
            volatility = np.nan
        else:
            current_price = hist['Close'].iloc[-1]
            returns = hist['Close'].pct_change().dropna()
            volatility = returns.std() * np.sqrt(252) if len(returns) > 0 else np.nan

        # Safely pull ratios from info with common keys
        stock_profile = {
            'ticker': ticker,
            'company_name': info.get('longName') or info.get('shortName') or ticker,
            'sector': info.get('sector', 'Unknown') or 'Unknown',
            'market_cap': info.get('marketCap', np.nan),
            'pe_ratio': info.get('trailingPE', np.nan),
            'pb_ratio': info.get('priceToBook', np.nan),
            'dividend_yield': info.get('dividendYield', np.nan),
            'forward_pe_ratio': info.get('forwardPE', np.nan),
            'margin_score': info.get('profitMargins', np.nan),  # profit margin
            'return_on_equity': info.get('returnOnEquity', np.nan),
            'return_on_assets': info.get('returnOnAssets', np.nan),
            'gross_margin': info.get('grossMargins', np.nan),
            'debt_to_equity': info.get('debtToEquity', np.nan),
            'current_ratio': info.get('currentRatio', np.nan),
            'beta': info.get('beta', np.nan),
            'volatility': volatility if not np.isnan(volatility) else 0.3,
            'current_price': current_price if not pd.isna(current_price) else np.nan
        }
        return stock_profile
    except Exception as e:
        # don't crash the whole loop on one ticker
        print(f"Error downloading {ticker}: {e}")
        return None

# ----------------------------
# Get S&P 500 list
# ----------------------------
def get_comprehensive_stock_data():
    try:
        url = 'https://datahub.io/core/s-and-p-500-companies/r/constituents.csv'
        df = pd.read_csv(url)
        return list(df[['Symbol', 'GICS Sector']].itertuples(index=False, name=None))
    except Exception as e:
        print(f"❌ Error fetching S&P 500 data: {e}")
        return []

# ----------------------------
# Download comparison stocks (parallel)
# ----------------------------
def download_comparison_stocks(stock_list=None, max_workers: int = 10, show_progress: bool = True) -> pd.DataFrame:
    if stock_list is None:
        stock_list = get_comprehensive_stock_data()
    if not stock_list:
        # fallback
        return download_comparison_stocks_fallback()

    # Use ThreadPoolExecutor to speed up lots of small yfinance calls
    stock_info = []
    successful = 0

    def worker(tuple_item):
        ticker = tuple_item[0]
        sector = tuple_item[1] if len(tuple_item) > 1 else 'Unknown'
        data = download_single_stock_data(ticker)
        if data:
            data['sector'] = sector or data.get('sector', 'Unknown')
            return data
        return None

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as exe:
        futures = {exe.submit(worker, t): t for t in stock_list}
        for i, fut in enumerate(concurrent.futures.as_completed(futures), 1):
            res = fut.result()
            if res:
                stock_info.append(res)
                successful += 1
            # Progress printed in blocks to avoid spamming
            if show_progress and i % 50 == 0:
                print(f"✅ Processed {i} tickers, downloaded {successful} valid profiles...")

    print(f"🎯 Successfully downloaded {successful} out of {len(stock_list)} tickers (attempted {len(stock_list)})")
    return pd.DataFrame(stock_info)

def download_comparison_stocks_fallback():
    predefined_tickers = [
        'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA', 'AVGO', 'ORCL', 'ADBE',
        'CSCO', 'CRM', 'INTC', 'AMD', 'QCOM', 'TXN', 'IBM', 'JNJ', 'UNH', 'LLY',
        'PFE', 'ABT', 'TMO', 'MRK', 'JPM', 'BAC', 'WFC', 'GS', 'MS', 'BLK',
        'WMT', 'TGT', 'COST', 'HD', 'LOW', 'MCD', 'SBUX', 'KO', 'PEP', 'PG',
        'XOM', 'CVX', 'COP', 'NEE', 'DUK', 'SO', 'BA', 'LMT', 'RTX'
    ]
    stock_info = []
    for t in predefined_tickers:
        data = download_single_stock_data(t)
        if data:
            stock_info.append(data)
    return pd.DataFrame(stock_info)

# ----------------------------
# Scoring (defensive + corrected)
# ----------------------------
def calculate_all_scores(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy().reset_index(drop=True)

    # Ensure expected columns exist
    expected_cols = [
        'pe_ratio', 'pb_ratio', 'dividend_yield', 'forward_pe_ratio', 'margin_score',
        'return_on_equity', 'return_on_assets', 'gross_margin',
        'debt_to_equity', 'current_ratio', 'beta', 'volatility', 'market_cap'
    ]
    for col in expected_cols:
        if col not in df.columns:
            df[col] = 0.0

    # Value score
    # Avoid division by zero by replacing zeros/infs with large default
    pe = df['pe_ratio'].replace([0, np.inf, -np.inf], np.nan).fillna(100.0)
    pb = df['pb_ratio'].replace([0, np.inf, -np.inf], np.nan).fillna(10.0)
    pe_score = (50.0 / pe).clip(0, 1)
    pb_score = (5.0 / pb).clip(0, 1)
    dividend_score = df['dividend_yield'].fillna(0).clip(0, 0.2) * 5  # 20% -> full score
    df['value_score'] = ((pe_score + pb_score + dividend_score) / 3.0).clip(0, 1)

    # Growth score
    fpe = df['forward_pe_ratio'].replace([0, np.inf, -np.inf], np.nan).fillna(100.0)
    forward_pe_score = (50.0 / fpe).clip(0, 1)
    profit_margin_score = df['margin_score'].fillna(0).clip(-1, 1)  # margins could be negative
    profit_margin_score = ((profit_margin_score + 1) / 2).clip(0, 1)  # normalize -1..1 to 0..1
    roe_score = df['return_on_equity'].fillna(0).clip(-1, 1)
    roe_score = ((roe_score + 1) / 2).clip(0, 1)
    df['growth_score'] = ((forward_pe_score + profit_margin_score + roe_score) / 3.0).clip(0, 1)

    # Quality score
    roa = df['return_on_assets'].fillna(0).clip(-1, 1)
    roa = ((roa + 1) / 2).clip(0, 1)
    gross_margin = df['gross_margin'].fillna(0).clip(-1, 1)
    gross_margin = ((gross_margin + 1) / 2).clip(0, 1)
    df['quality_score'] = ((roa + gross_margin) / 2.0).clip(0, 1)

    # Stability score
    debt_equity = df['debt_to_equity'].replace(0, np.nan).fillna(1.0)
    debt_score = (1.0 / (1.0 + debt_equity)).clip(0, 1)  # higher debt->lower score
    current_ratio = df['current_ratio'].fillna(1.5).clip(0, 10)
    liquidity_score = ((current_ratio - 1) / 4.0).clip(0, 1)  # scaled
    df['stability_score'] = ((debt_score + liquidity_score) / 2.0).clip(0, 1)

    # Risk score
    beta = df['beta'].fillna(1.0).clip(0.0, 3.0)
    beta_score = (beta / 3.0).clip(0, 1)
    volatility = df['volatility'].fillna(0.3).clip(0, 2.0)
    vol_score = (volatility / 2.0).clip(0, 1)
    df['risk_score'] = ((beta_score + vol_score) / 2.0).clip(0, 1)

    # Composite
    df['composite_score'] = (
        df['value_score'] * 0.25 +
        df['growth_score'] * 0.25 +
        df['quality_score'] * 0.20 +
        df['stability_score'] * 0.15 +
        (1.0 - df['risk_score']) * 0.15
    ).clip(0, 1)

    # Ensure columns exist for later steps
    for col in ['value_score', 'growth_score', 'quality_score', 'stability_score', 'risk_score', 'composite_score']:
        if col not in df.columns:
            df[col] = 0.0

    return df

# ----------------------------
# Build feature matrix
# ----------------------------
def build_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        raise ValueError("build_feature_matrix received empty DataFrame")

    # Create sector dummies safely
    sector_dummies = pd.get_dummies(df['sector'].fillna('Unknown'), prefix='sector')

    feature_columns = [
        'value_score', 'growth_score', 'quality_score', 'stability_score', 'risk_score',
        'composite_score', 'market_cap', 'pe_ratio', 'dividend_yield', 'volatility', 'beta'
    ]
    # Ensure they exist
    for col in feature_columns:
        if col not in df.columns:
            df[col] = 0.0

    numerical_feature = df[feature_columns].fillna(0.0)
    feature_matrix = pd.concat([numerical_feature, sector_dummies], axis=1)

    # Scale (fit_transform)
    feature_matrix_scaled = pd.DataFrame(
        scaler.fit_transform(feature_matrix),
        columns=feature_matrix.columns,
        index=feature_matrix.index
    )
    return feature_matrix_scaled

# ----------------------------
# User input (consistent keys)
# ----------------------------
def get_user_input():
    target_stock = input("Enter The Target Stock (if you want to otherwise Skip): ").strip()
    if target_stock == "":
        target_stock = None

    while True:
        try:
            budget = float(input('Enter Your Investment Budget: ').replace(",", "").strip())
            if budget <= 0:
                print('Budget must be > 0')
                continue
            break
        except ValueError:
            print("Enter a valid number for budget")

    risk_mapping = {'1': 'low', '2': 'medium', '3': 'high'}
    while True:
        rc = input("Enter risk capacity (1=low, 2=medium, 3=high): ").strip()
        if rc in risk_mapping:
            risk_capacity = risk_mapping[rc]
            break
        print("Enter 1, 2 or 3")

    sector_interested = input("Enter the Sector you are interested in (or press Enter for Any): ").strip()
    if sector_interested == "":
        sector_interested = "Any"

    while True:
        try:
            horizon = int(input("Enter the horizon of your investment (years): ").strip())
            if horizon <= 0:
                print("Horizon must be > 0")
                continue
            investment_horizon = horizon
            break
        except ValueError:
            print("Enter a valid integer for horizon")

    # similarity_focus: optional, defaults to 'balanced'
    similarity_focus = input("Similarity focus? (sector/growth/balanced) [balanced]: ").strip().lower()
    if similarity_focus not in ('sector', 'growth', 'balanced'):
        similarity_focus = 'balanced'

    return {
        'target_stock': target_stock,
        'budget': budget,
        'risk_capacity': risk_capacity,
        'sector_interested': sector_interested,
        'investment_horizon': investment_horizon,
        'similarity_focus': similarity_focus
    }

# ----------------------------
# Recommend similar stocks
# ----------------------------
def find_similar_stocks(
    target_stock_data: Optional[pd.Series],
    comparison_stocks: pd.DataFrame,
    user_inputs: dict,
    num_recommendations: int = 6  # fixed: we want top 5–6 only
):
    # Combine datasets
    if target_stock_data is not None:
        all_stocks = pd.concat([pd.DataFrame([target_stock_data]), comparison_stocks], ignore_index=True)
    else:
        all_stocks = comparison_stocks.copy()

    # Compute scores
    all_stocks = calculate_all_scores(all_stocks)

    # Build feature matrix
    feature_matrix = build_feature_matrix(all_stocks)

    # Build user vector
    if target_stock_data is not None:
        user_vector = feature_matrix.iloc[0].values
    else:
        user_vector = feature_matrix.mean().values
        rf = {'low': 0.9, 'medium': 1.0, 'high': 1.1}[user_inputs['risk_capacity']]
        user_vector = user_vector * rf

    # Calculate cosine similarity
    similarity_scores = cosine_similarity([user_vector], feature_matrix)[0]
    similar_indices = np.argsort(similarity_scores)[::-1]

    # Skip self
    if target_stock_data is not None:
        similar_indices = similar_indices[1: num_recommendations * 6]  # larger pool
    else:
        similar_indices = similar_indices[: num_recommendations * 6]

    recommendations = []
    risk_map = {"low": 0.3, "medium": 0.6, "high": 0.9}

    for idx in similar_indices:
        stock = all_stocks.iloc[idx]

        # Sector filter
        if user_inputs['sector_interested'].lower() != "any":
            if user_inputs['sector_interested'].lower() not in str(stock.get('sector', '')).lower():
                continue

        # Risk filter (slightly relaxed)
        if stock.get('risk_score', 1.0) > (risk_map[user_inputs['risk_capacity']] + 0.4):
            continue

        # Horizon filter (slightly relaxed)
        if user_inputs['investment_horizon'] < 3 and stock.get('volatility', 1.0) > 0.6:
            continue

        # Budget allocation
        allocation = user_inputs['budget'] / num_recommendations
        price = stock.get('current_price', np.nan)
        if pd.isna(price) or price == 0:
            continue
        shares = int(allocation // price)
        if shares <= 0:
            continue

        recommendations.append({
            'ticker': stock.get('ticker'),
            'company': stock.get('company_name'),
            'sector': stock.get('sector'),
            'price': price,
            'shares': shares,
            'allocation': round(allocation, 2),
            'risk_score': round(stock.get('risk_score', 0), 2),
            'similarity': round(float(similarity_scores[idx]), 3)
        })

        # Stop once we reach 6 recommendations
        if len(recommendations) >= num_recommendations:
            break

    # Fallback: if filters cut too many, fill with top-scoring ones
    if len(recommendations) < num_recommendations:
        extra_needed = num_recommendations - len(recommendations)
        fallback = (
            all_stocks.sort_values('composite_score', ascending=False)
            .head(extra_needed)
        )
        for _, s in fallback.iterrows():
            price = s.get('current_price', np.nan)
            if pd.isna(price) or price == 0:
                continue
            shares = int((user_inputs['budget'] / num_recommendations) // price)
            if shares <= 0:
                continue
            recommendations.append({
                'ticker': s.get('ticker'),
                'company': s.get('company_name'),
                'sector': s.get('sector'),
                'price': price,
                'shares': shares,
                'allocation': round(user_inputs['budget'] / num_recommendations, 2),
                'risk_score': round(s.get('risk_score', 0), 2),
                'similarity': 0.0
            })
            if len(recommendations) >= num_recommendations:
                break

    return recommendations


# ----------------------------
# Display
# ----------------------------
def display_recommendations(target_stock, recommendations, user_inputs):
    if target_stock is None:
        print("\n🎯 RECOMMENDATIONS (no specific target stock provided)")
    else:
        print(f"\n🎯 STOCKS SIMILAR TO {target_stock.get('ticker', 'UNKNOWN')}")
    print("=" * 80)
    print(f"Criteria: Budget ${user_inputs['budget']:,}, Risk {user_inputs['risk_capacity']}, Sector {user_inputs['sector_interested']}, Horizon {user_inputs['investment_horizon']}y")
    for i, rec in enumerate(recommendations, 1):
        print(f"\n#{i} {rec['ticker']} - {rec['company']}")
        print(f"   Sector: {rec['sector']}")
        print(f"   Price: {rec['price']}")
        print(f"   Allocation per stock: ${rec['allocation']:,} -> {rec['shares']} shares")
        print(f"   Risk score: {rec['risk_score']}  Similarity: {rec['similarity']}")

# ----------------------------
# Main
# ----------------------------
def main():
    global company_to_ticker
    company_to_ticker = company_ticker_symbol()

    user_inputs = get_user_input()
    target_ticker = ticker_to_name(user_inputs['target_stock'])

    target_stock_data = None
    if target_ticker is not None:
        print(f"Looking up target ticker: {target_ticker}")
        target_stock_data = download_single_stock_data(target_ticker)
        if target_stock_data is None:
            print("Warning: couldn't download target stock profile. Continuing without target.")

    print("Downloading comparison universe (this may take a bit)...")
    comparison_stocks = download_comparison_stocks()  # uses S&P list internally (parallel)

    if comparison_stocks is None or comparison_stocks.empty:
        print("No comparison stocks downloaded. Exiting.")
        return

    # If target exists, convert to DataFrame row for scoring
    if target_stock_data is not None:
        # Ensure dict -> Series that matches DataFrame columns for concat
        target_df = calculate_all_scores(pd.DataFrame([target_stock_data])).iloc[0]
        # find_similar_stocks expects Series-like for target_stock_data
        recommendations = find_similar_stocks(target_df, comparison_stocks, user_inputs)
    else:
        recommendations = find_similar_stocks(None, comparison_stocks, user_inputs)

    if not recommendations:
        print("No suitable recommendations found with current criteria (budget/filters). Try increasing budget or relaxing filters.")
    else:
        display_recommendations(target_stock_data or {}, recommendations, user_inputs)

if __name__ == "__main__":
    main()

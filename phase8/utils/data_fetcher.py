import pandas as pd
import yfinance as yf

def fetch_live_data(symbols):
    all_data = []
    for symbol in symbols:
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            hist = ticker.history(period='6mo')

            if hist.empty:
                continue

            volatility = hist["Close"].pct_change().std()

            stock_data = {
                "Symbol": symbol,
                "Sector": info.get("sector", "Unknown"),
                "PE_Ratio": info.get("trailingPE", 0),
                "EPS": info.get("trailingEps", 0),
                "ROE": (info.get("returnOnEquity", 0) or 0) * 100,
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
    return df

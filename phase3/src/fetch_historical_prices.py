# import pandas as pd
# import numpy as np 
# from datetime import datetime,timedelta
# import yfinance 
# import os 

# def fetch_historical_data(symbols,output_file=r'C:\Users\abc\StockApp\phase3\data\raw\historical_prices.csv',months_ahead=6):

#     all_data=[]
#     end_date=datetime.today()
#     start_date=end_date - timedelta(days=365*2)

#     print("Fetching real time historical data ")

#     for symbol in symbols:
#         try:
#             df=yfinance.download(symbol,start=start_date,end=end_date,progress=False)

#             df.columns = [col[1] if isinstance(col, tuple) else col for col in df.columns]

#             if df.empty:
#                 print("No Data for Symbol ")
#                 continue
#             df=df[['Close']].reset_index()
#             df['Symbol']=symbol

#             days_ahead=int(months_ahead*21)
#             df['Future_Close']=df['Close'].shift(-days_ahead)
#             df['Forward_Return'] = (df['Future_Close'] - df['Close']) / df['Close']
#             if len(df)>days_ahead:
#                last_row=df.iloc[-(days_ahead+1)]
#             all_data.append({
#                     "Symbol":symbol,
#                     'Current_price':last_row['Close'],
#                     "Future_price":last_row['Future_Close'],
#                     "Forward_Return":last_row['Forward_Return']
#                 })

#         except Exception as e:
#             print(f"Error {e}")

#     price_df=pd.DataFrame(all_data)
#     os.makedirs(os.path.dirname(output_file),exist_ok=True)
#     price_df.to_csv(output_file,index=False)
#     print(f"Saved Historical Price{output_file}")

#     return price_df



       
# print(fetch_historical_data(["AAPL"],months_ahead=6))
    







import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import yfinance as yf
import os

def fetch_historical_data(symbols, output_file=r'C:\Users\abc\StockApp\phase3\data\raw\historical_prices.csv', months_ahead=6):
    all_data = []
    end_date = datetime.today()
    start_date = end_date - timedelta(days=365 * 2)

    print("📈 Fetching real-time historical data...")

    # Ensure symbols is a list
    if isinstance(symbols, str):
        symbols = [symbols]

    for symbol in symbols:
        try:
            print(f"→ Fetching {symbol}...")
            df = yf.download(symbol, start=start_date, end=end_date, progress=False,auto_adjust=True)

            if df is None or df.empty:
                print(f"⚠️ No data for {symbol}")
                continue

            # ✅ Handle MultiIndex properly (new + old yfinance versions)
            if isinstance(df.columns, pd.MultiIndex):
                levels = list(df.columns.names)

                # NEW yfinance format: ('Price', 'Ticker')
                if levels == ['Price', 'Ticker']:
                    df = df.xs(symbol, axis=1, level='Ticker', drop_level=False)

                # OLD yfinance format: ('Ticker', 'Price')
                elif levels == ['Ticker', 'Price']:
                    df = df.xs(symbol, axis=1, level='Ticker', drop_level=False)

                # Drop top level if only one ticker
                elif len(df.columns.levels[0]) == 1:
                    df.columns = df.columns.droplevel(0)

            # After all that, normalize column names
            df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]

            if 'Close' not in df.columns:
                print(f"⚠️ 'Close' column missing for {symbol}. Columns found: {df.columns}")
                continue

            df = df[['Close']].reset_index()
            df['Symbol'] = symbol

            days_ahead = int(months_ahead * 21)
            df['Future_Close'] = df['Close'].shift(-days_ahead)
            df['Forward_Return'] = (df['Future_Close'] - df['Close']) / df['Close']

            if len(df) > days_ahead:
                last_row = df.iloc[-(days_ahead + 1)]
                all_data.append({
                    "Symbol": symbol,
                    "Current_price": last_row['Close'],
                    "Future_price": last_row['Future_Close'],
                    "Forward_Return": last_row['Forward_Return']
                })
            else:
                print(f"⚠️ Not enough data for {symbol}")

        except Exception as e:
            print(f"⚠️ Error fetching {symbol}: {e}")

    price_df = pd.DataFrame(all_data)
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    price_df.to_csv(output_file, index=False)
    print(f"💾 Saved historical prices → {output_file}")
    print(f"✅ Collected data for {len(price_df)} symbols")

    return price_df


# 🧪 Test
if __name__ == "__main__":
    print(fetch_historical_data(["AAPL", "MSFT", "NVDA"], months_ahead=6))


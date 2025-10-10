from flask import Flask, render_template, request
import yfinance as yf
import pandas as pd

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def dashboard():
    stock_data = None
    officers_data = None
    holders = None

    print(">>> Request Method:", request.method)

    if request.method == "POST":
        ticker_symbol = request.form.get('ticker')
        print(">>> Form submitted data:", request.form)
        print(">>> Ticker symbol extracted:", ticker_symbol)


        if ticker_symbol:
            ticker_symbol = ticker_symbol.upper()
            print(">>> Fetching data for:", ticker_symbol)

            try:
                ticker = yf.Ticker(ticker_symbol)
                info = ticker.info
                print(">>> Got info keys:", list(info.keys())[:10])  # show first few keys

                company_name = info.get('longName', 'N/A')
                sector = info.get('sector', 'N/A')

                # Officers
                officers = info.get('companyOfficers', [])
                if officers:
                    df = pd.DataFrame(officers)
                    ceo_df = df[df['title'].str.contains('CEO', case=False, na=False)]
                    officers_df = df[~df['title'].str.contains('CEO', case=False, na=False)].head(2)
                    final_df = pd.concat([ceo_df, officers_df])
                    officers_data = final_df[['name', 'title', 'age', 'totalPay']].to_dict(orient='records')
                    print(">>> Officers data:", officers_data)
                else:
                    print(">>> No officers data found")

                # Institutional Holders
                holders_df = ticker.institutional_holders
                if holders_df is not None and not holders_df.empty:
                    holders_df['pctHeld'] = (holders_df['pctHeld'] * 100).round(2)
                    holders_df['pctChange'] = (holders_df['pctChange'] * 100).round(2)
                    holders = holders_df[['Holder', 'pctHeld', 'Shares', 'Value', 'pctChange']].head(5).to_dict(orient='records')
                    print(">>> Holders found:", len(holders))
                else:
                    print(">>> No institutional holders found")

                stock_data = {"company_name": company_name, "sector": sector, "symbol": ticker_symbol}

            except Exception as e:
                print(">>> ERROR fetching ticker:", e)

    return render_template('index.html', stock_data=stock_data, officers_data=officers_data, holders=holders)

if __name__ == "__main__":
    app.run(debug=True)

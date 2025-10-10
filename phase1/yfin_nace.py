# import yfinance as yf
# import pandas as pd 

# data=yf.download("AAPL",start='2024-01-01',end='2025-10-10',progress=False,threads=False)

# print(data.tail(10))




#Getting the Basic info of the Company 

import yfinance as yf
import pandas as pd 
ticker=yf.Ticker('HOOD')
infor=ticker.info

print(f"AllTimeHigh={infor['allTimeHigh']}")
print(f"2. The business summary of the company----:{infor['longBusinessSummary']}")
print(f"Operatingmargins={infor['operatingMargins']}")
print(f"GrossProfits={infor['grossProfits']}")


officers=ticker.info.get('companyOfficers',[])
df=pd.DataFrame(officers)
ceo_df=df[df['title'].str.contains('CEO',case=False,na=False)]

officers_df=df[~df['title'].str.contains('CEO',case=False,na=False)].head(2)

final_df=pd.concat([ceo_df,officers_df])

print(final_df[['name','title','age','totalPay']])


print(ticker.recommendations)
print(ticker.recommendations_summary)
print(ticker.earnings_dates)

df=ticker.institutional_holders.copy()

df['pctHeld']=(df['pctHeld'] * 100).round(2)
df['pctChange']=(df['pctChange']*100).round(2)
print(df[['Holder', 'pctHeld', 'Shares', 'Value', 'pctChange']].head())

print(ticker.major_holders)
print(ticker.news)




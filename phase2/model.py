import pandas as pd
import numpy as np 
import yfinance as yf
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
import warnings

warnings.filterwarnings('ignore')
stock_data=None
feature_data=None
scaler=StandardScaler()
company_to_ticker={}

def company_ticker_symbol():
      
      mapping = {
        # Technology
        'apple': 'AAPL', 'microsoft': 'MSFT', 'google': 'GOOGL', 'alphabet': 'GOOGL',
        'amazon': 'AMZN', 'tesla': 'TSLA', 'meta': 'META', 'facebook': 'META',
        'nvidia': 'NVDA', 'intel': 'INTC', 'amd': 'AMD', 'qualcomm': 'QCOM',
        'cisco': 'CSCO', 'oracle': 'ORCL', 'ibm': 'IBM', 'salesforce': 'CRM',
        'adobe': 'ADBE', 'netflix': 'NFLX', 'paypal': 'PYPL', 'square': 'SQ',
        
        # Healthcare
        'johnson & johnson': 'JNJ', 'pfizer': 'PFE', 'merck': 'MRK', 'abbott': 'ABT',
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
      

      variations={}
      for company,ticker in mapping.items():
            variations[company.replace(" ","")]=ticker
            variations[company.replace("&","and")]=ticker
            variations[company.replace("corporation","corp")]=ticker
            variations[company.replace("incorporated","inc")]=ticker
            
      mapping.update(variations)
      return mapping                


def ticker_to_name(company_name):
    global company_to_ticker_map

    if not company_to_ticker_map:
        company_to_ticker_map=company_ticker_symbol()
        
        
    clean_name=company_name.lower().strip()
    if clean_name in company_to_ticker_map:
        return company_to_ticker_map[clean_name]
    
    try:
         search_result=yf.Ticker(clean_name.upper())
         info=search_result.info
         if info and 'symbol' in info:
              return info['symbol']
    except:
         pass
    
    for company,ticker in company_to_ticker_map.items():
        if clean_name in company or company in clean_name:
            return ticker
        
    return company_name.upper()


def downlaod_single_stock_data(ticker):
     try:
          stock=yf.Ticker(ticker)
          info=stock.info

          if not info:
               return None
          
          hist=stock.history(period='6mo')

          if hist.empty or len(hist)<30:
               return None

          returns=hist['Close'].pct_change().dropna()
          volatility=returns.std() *np.sqrt(256) if len(returns)>0 else 0.3 
          current_price=hist['Close'].iloc[-1]
          
          
          stock_profile = {
            'ticker': ticker,
            'company_name': info.get('longName', ticker),
            'sector': info.get('sector', 'Unknown'),
            'market_cap': info.get('marketCap', 0),
            'pe_ratio': info.get('trailingPE', np.nan),
            'volatility': volatility,
            'current_price': current_price,
        }
          return stock_profile
     except Exception as e:
          print(f'Error Downloading the data')
          return None



def get_chomprihansive_stock_data():
     url='https://datahub.io/core/s-and-p-500-companies/r/constituents.csv'
     df=pd.read_csv(url)
     return list(df[['Symbol', 'GICS Sector']].itertuples(index=False, name=None))

def calculate_all_scores(df):
     df=df.copy()

     pe_score=1/(df['pe_ratio'].replace([np.inf,-np.inf],100).clip(1,100)/50)
     pb_score=1/(df['pb_ratio'].replace([np.inf,-np.inf],10).clip(0.1,20)/5)
     dividend_score=df['dividend_yield'].fillna(0)*20
     df['value_score']=(pe_score+pb_score+dividend_score.clip(0,1))/3


     forward_pe_score=1/(df['forward_pe_ratio'].replace([np.inf,-np.inf],100).clip(1,100)/50)
     profit_margin_score=df['margin_score'].fillna(0).clip(0,0.5)*20
     roe_score=df['return_on_equity'].fillna(0).clip(0,1)
     df['groth_score']=(forward_pe_score+profit_margin_score+roe_score)/3


     if 'quality_score' not in df.columns:
          roa=df.get('return_on_assets',pd.Series(0)).fillna(0).clip(0,0.2)*5
          gross_margin=df.get('gross_margin',pd.Series(0)).fillna(0).clip(0,0.8)*1.25
          df['quality_score']=(roa+gross_margin)/2


     if 'stability_score' not in df.columns:
          debt_equity=df.get('debt_to_equity',pd.Series(0.5)).fillna(0.5)
          debt_score=(1+1/debt_equity)
          current_ratio=df.get('current_ratio',pd.Series(1.5)).fillna(1.5)
          liquidty_score=(current_ratio.clip(1,3)-1)/2
          df['stability_score']=(debt_score+liquidty_score)/2


     if 'risk_score' not in df.columns:
          beta=df.get('beta',pd.Series(0).fillna(1.0))
          beta_score=(beta.clip(0.5,2)-0.5)/1.5
          volatility=df.get('volatility',pd.Series(0.3).fillna(0.3))
          vol_score=volatility.clip(0,1)
          df['risk_score']=(beta_score+vol_score)/2

     df['composite_score']=(
          df['value_score']*0.25 +
          df['growth_score']*0.25 +
          df['quality_score']*0.20 +
          df['stability_score'] *0.15 +
          (1-df['risk_score'])*0.15
     )

     score_columns=['value_score','growth_score','quality_score','stability_score','risk_score']

     for col in score_columns:
          df[col]=df[col].clip(0,1)

     
     return df



def build_feature_matrix(Stock_data):
     sector_dummies=pd.get_dummies(stock_data['sector'],prefix='sector')

     feature_columns = [
    'value_score', 'growth_score', 'quality_score', 'stability_score', 'risk_score',
    'composite_score', 'market_cap', 'pe_ratio', 'dividend_yield', 'volatility', 'beta']

     numerical_featrue=stock_data[feature_columns].fillna(0)
     feature_matrix=pd.concat([numerical_featrue,sector_dummies],axis=1)

     feature_matrix=pd.DataFrame(
          scaler.fit_transform(feature_matrix),
          columns=feature_matrix.columns,
          index=feature_matrix.index
     )     

     return feature_matrix


def get_user_input(input_info):
     target_stock=input("Enter The Target Stock (if you want to otherwise Skip)")
     if target_stock=="":
          target_stock=None

     while True:
          try:
               budget=input('Enter Your Investment Budget').replace(",","")
               if budget<=0:
                    print('Enter the busget again')
                    continue

          except ValueError:
               print("Enter The Valid input for this")

     
     risk_mapping={
          '1':'low',
          '2': 'medium',
          '3' : 'high'

     }

     while True:
          risk_choice=input("Enter the Number 1 ,2 ,3 ").strip()
          if risk_choice in risk_mapping:
               risk_capacity=risk_mapping[risk_choice]
               break
          else:
               print("Enter the Valid number 1,2 or 3 ")

     sector_intrested=input("Enter the Sector Your are intrested in ")
     if sector_intrested=="":
          sector_intrested="Any"

     
     while True:
          try:
               horizon=input("Enter the horizon of your Investment")
               if horizon<=0:
                    print("IT should be greater than 1")
                    continue
               investment_horizon=horizon
               break
          except ValueError:
               print("Enter the Valid Number for this")
               
          











               

                     



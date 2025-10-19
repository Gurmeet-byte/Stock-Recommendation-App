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
    global company_to_ticker

    if not company_to_ticker:
        company_to_ticker=company_ticker_symbol()
        
        
        if company_name is None:
          return None
        
        
    clean_name=company_name.lower().strip()
    if clean_name in company_to_ticker:
        return company_to_ticker[clean_name]
    
    try:
         search_result=yf.Ticker(clean_name.upper())
         info=search_result.info
         if info and 'symbol' in info:
              return info['symbol']
    except:
         pass
    
    for company,ticker in company_to_ticker.items():
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


# def get_chomprihansive_stock_data():
#      url='https://datahub.io/core/s-and-p-500-companies/r/constituents.csv'
#      df=pd.read_csv(url)
#      return list(df[['Symbol', 'GICS Sector']].itertuples(index=False, name=None))

def download_comparison_stocks(stock_list=None):
    if stock_list==None:
        stock_list=get_comprehensive_stock_data()
    """Download data for comparison stocks using S&P 500 companies"""
    try:
        # Get the comprehensive stock list
        stock_list = get_comprehensive_stock_data()
        
        print(f"📊 Found {len(stock_list)} S&P 500 companies for comparison")
        
        stock_info = []
        successful_downloads = 0
        
        for stock_tuple in stock_list:
            # Assuming the tuple structure is (Symbol, GICS Sector)
            ticker = stock_tuple[0]  # Symbol
            sector = stock_tuple[1]  # GICS Sector
            
            # Download stock data
            data = downlaod_single_stock_data(ticker)
            
            if data:
                # Enhance the data with the sector from our comprehensive list
                data['sector'] = sector  # Use the sector from S&P 500 data
                stock_info.append(data)
                successful_downloads += 1
                
                # Progress indicator
                if successful_downloads % 50 == 0:
                    print(f"✅ Downloaded {successful_downloads} stocks...")
        
        print(f"🎯 Successfully downloaded {successful_downloads} out of {len(stock_list)} stocks")
        
        return pd.DataFrame(stock_info)
        
    except Exception as e:
        print(f"❌ Error downloading comparison stocks: {e}")
        # Fallback to original method if S&P 500 data fails
        print("🔄 Falling back to predefined stock universe...")
        return download_comparison_stocks_fallback()

def download_comparison_stocks_fallback():
    """Fallback method using predefined stock universe"""
    predefined_tickers = [
        # Technology
        'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA', 'AVGO', 'ORCL', 'ADBE',
        'CSCO', 'CRM', 'INTC', 'AMD', 'QCOM', 'TXN', 'IBM', 'NOW', 'UBER', 'SNOW',
        # Healthcare
        'JNJ', 'UNH', 'LLY', 'PFE', 'ABT', 'TMO', 'MRK', 'ABBV', 'DHR', 'BMY',
        # Financial Services
        'JPM', 'BAC', 'WFC', 'C', 'GS', 'MS', 'BLK', 'SCHW', 'AXP', 'V',
        # Consumer
        'WMT', 'TGT', 'COST', 'HD', 'LOW', 'MCD', 'SBUX', 'KO', 'PEP', 'PG',
        # Energy & Industrial
        'XOM', 'CVX', 'COP', 'NEE', 'DUK', 'SO', 'D', 'BA', 'LMT', 'RTX',
    ]
    
    stock_info = []
    for ticker in predefined_tickers:
        data = downlaod_single_stock_data(ticker)
        if data:
            stock_info.append(data)
    
    return pd.DataFrame(stock_info)

def get_comprehensive_stock_data():
    """Get S&P 500 companies data with sectors"""
    try:
        url = 'https://datahub.io/core/s-and-p-500-companies/r/constituents.csv'
        df = pd.read_csv(url)
        
        # Return list of tuples (Symbol, GICS Sector)
        return list(df[['Symbol', 'GICS Sector']].itertuples(index=False, name=None))
    
    except Exception as e:
        print(f"❌ Error fetching S&P 500 data: {e}")
        print("🔄 Using fallback stock list...")
        # Return an empty list to trigger fallback
        return []


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


def get_user_input():
     target_stock=input("Enter The Target Stock (if you want to otherwise Skip)").strip()
     if target_stock=="":
          target_stock=None

     while True:
          try:
               budget=float(input('Enter Your Investment Budget').replace(",","").strip())
               if budget<=0:
                    print('Enter the busget again')
                    
               break

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
               horizon=int(input("Enter the horizon of your Investment").strip())
               if horizon<=0:
                    print("IT should be greater than 1")
                    continue
               investment_horizon=horizon
               break
          except ValueError:
               print("Enter the Valid Number for this")

     return {
          'target_stock':target_stock,
          'budget':budget,
          'risk_choice':risk_choice,
          'sector_intrested':sector_intrested,
          'horizon':investment_horizon
     }
               
          
def create_user_profile_from_target_stock(target_stock_data, user_inputs):
    """Create user profile based on target stock characteristics"""
    
    
    if user_inputs['similarity_focus'] == 'sector':
        weights = [0.4, 0.2, 0.1, 0.2, 0.1, 0.3, 0.8, 0.1, 0.1, 0.1, 0.1]  
    elif user_inputs['similarity_focus'] == 'growth':
        weights = [0.3, 0.6, 0.4, 0.2, 0.2, 0.5, 0.2, 0.3, 0.4, 0.2, 0.2]  
    else:  
        weights = [0.2, 0.2, 0.2, 0.3, 0.8, 0.3, 0.1, 0.1, 0.1, 0.6, 0.6]  
    
    user_vector = [
        target_stock_data['value_score'] * weights[0],
        target_stock_data['growth_score'] * weights[1],
        target_stock_data['quality_score'] * weights[2],
        target_stock_data['stability_score'] * weights[3],
        target_stock_data['risk_score'] * weights[4],
        target_stock_data['composite_score'] * weights[5],
        np.log1p(target_stock_data['market_cap']) * weights[6],
        (1 / target_stock_data['pe_ratio'].clip(1, 100)) * weights[7] if not np.isnan(target_stock_data['pe_ratio']) else 0.5,
        target_stock_data['dividend_yield'] * weights[8],
        target_stock_data['volatility'] * weights[9],
        target_stock_data['beta'] * weights[10]
    ]
    
    return user_vector


def find_similar_stocks(target_stock_data, comparison_stocks, user_inputs, num_recommendations=8):
    """Find and recommend similar stocks based on user preferences."""

    # Combine target stock with comparison stocks
    if target_stock_data is not None:
        all_stocks = pd.concat([pd.DataFrame([target_stock_data]), comparison_stocks], ignore_index=True)
    else:
        all_stocks = comparison_stocks.copy()

    # Calculate all financial scores (assumes you’ve defined this)
    all_stocks = calculate_all_scores(all_stocks)

    # Build the feature matrix for similarity comparison
    feature_matrix = build_feature_matrix(all_stocks)

    # --- Create user vector ---
    if target_stock_data is not None:
        user_vector = feature_matrix.iloc[0].values
    else:
        # Generic vector based on average, adjusted by user’s risk capacity
        user_vector = feature_matrix.mean().values
        risk_factor = {'low': 0.9, 'medium': 1.0, 'high': 1.1}[user_inputs['risk_capacity']]
        user_vector *= risk_factor

    # --- Compute cosine similarity ---
    similarity_scores = cosine_similarity([user_vector], feature_matrix)[0]

    # --- Rank by similarity ---
    similar_indices = np.argsort(similarity_scores)[::-1]
    if target_stock_data is not None:
        similar_indices = similar_indices[1:num_recommendations * 2]
    else:
        similar_indices = similar_indices[:num_recommendations * 2]

    # --- Prepare recommendations ---
    recommendations = []
    for idx in similar_indices:
        stock = all_stocks.iloc[idx]

        # Sector filter
        if user_inputs['sector_interested'].lower() != "any":
            if user_inputs['sector_interested'].lower() not in stock['sector'].lower():
                continue

        # Risk filter
        risk_map = {"low": 0.3, "medium": 0.6, "high": 0.9}
        if stock['risk_score'] > risk_map[user_inputs['risk_capacity']] + 0.3:
            continue

        # Horizon filter (example rule)
        if user_inputs['investment_horizon'] < 3 and stock['volatility'] > 0.5:
            continue

        # Budget allocation and share count
        allocation = user_inputs['budget'] / num_recommendations
        shares = int(allocation // stock['current_price'])
        if shares <= 0:
            continue

        recommendations.append({
            'ticker': stock['ticker'],
            'company': stock['company_name'],
            'sector': stock['sector'],
            'price': stock['current_price'],
            'shares': shares,
            'allocation': round(allocation, 2),
            'risk_score': round(stock['risk_score'], 2),
            'similarity': round(similarity_scores[idx], 3)
        })

        if len(recommendations) >= num_recommendations:
            break

    return recommendations


def display_recommendations(target_stock, recommendations, user_inputs):
    """Display recommendations"""
    print(f"\n🎯 STOCKS SIMILAR TO {target_stock['ticker']}")
    print("="*80)
    
    # Display criteria
    print(f"\n📊 RECOMMENDATION CRITERIA:")
    print(f"   • Target Stock: {target_stock['company_name']} ({target_stock['ticker']})")
    # ... more criteria
    
    # Display each recommendation
    for rec in recommendations:
        print(f"\n#{rec['rank']} {rec['ticker']} - {rec['company']}")
        print(f"   📈 Sector: {rec['sector']}")
        print(f"   💰 Allocation: ${rec['allocation']:,.2f} ({rec['shares']} shares)")
        # ... more details
    
    # Display portfolio summary
    print(f"\n💰 PORTFOLIO SUMMARY:")
    
def main():
    """Main function for stock-focused recommendations"""
    # Initialize
    company_ticker_symbol()
    
    # Get user inputs
    user_inputs = get_user_input()
    
    # Convert company name to ticker
    target_ticker = ticker_to_name(user_inputs['target_stock'])
    
    # Download data
    target_stock_data = downlaod_single_stock_data(target_ticker)
    comparison_stocks = download_comparison_stocks(get_comprehensive_stock_data())
    
    # Calculate scores
    target_stock_data = calculate_all_scores(pd.DataFrame([target_stock_data])).iloc[0]
    
    # Find and display recommendations
    recommendations = find_similar_stocks(target_stock_data, comparison_stocks, user_inputs)
    display_recommendations(target_stock_data, recommendations, user_inputs)

if __name__ == "__main__":
    main()
               

                     



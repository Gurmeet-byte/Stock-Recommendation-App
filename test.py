import yfinance as yf

def get_company_sector(ticker_symbol):
    """
    Fetches the sector of a company using its stock ticker symbol.
    
    Args:
        ticker_symbol (str): The stock ticker symbol (e.g., 'JPM' for JPMorgan Chase).

    Returns:
        str: The sector name or an error message if not found.
    """
    try:
        # Create a Ticker object
        ticker_data = yf.Ticker(ticker_symbol)
        
        # Access the 'info' attribute to get fundamental data
        info = ticker_data.info
        
        # The sector information is usually in the 'sector' key
        sector = info.get('sector', 'Sector information not available')
        industry = info.get('industry', 'Industry information not available')
        
        return sector, industry

    except Exception as e:
        return f"An error occurred: {e}", "N/A"

# Example usage:
ticker_symbol_1 = 'CPRI'  # JPMorgan Chase & Co. (a major bank)
sector_1, industry_1 = get_company_sector(ticker_symbol_1)
print(f"Ticker: {ticker_symbol_1}")
print(f"Sector: {sector_1}")
print(f"Industry: {industry_1}")

print("-" * 20)

ticker_symbol_2 = 'AAL' # Microsoft Corp (a tech company)
sector_2, industry_2 = get_company_sector(ticker_symbol_2)
print(f"Ticker: {ticker_symbol_2}")
print(f"Sector: {sector_2}")
print(f"Industry: {industry_2}")

ticker_symbol_3 = 'AMT'  # JPMorgan Chase & Co. (a major bank)
sector_3, industry_3 = get_company_sector(ticker_symbol_1)
print(f"Ticker: {ticker_symbol_3}")
print(f"Sector: {sector_3}")
print(f"Industry: {industry_3}")

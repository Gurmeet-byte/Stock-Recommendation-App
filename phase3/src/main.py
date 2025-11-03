from src.fetch_data import fetch_sp500_list,fetch_yahoo_data
from src.preprocess_data import preprocess_yahoo_data
from src.train_model import train_ml_model,score_stocks
from src.fetch_historical_prices import fetch_historical_data





def main():
    sp500df=fetch_sp500_list()
    symbols=sp500df['Symbol'].tolist()
    yahoo_df=fetch_yahoo_data(symbols,limit=50)
    clean_df=preprocess_yahoo_data(r'C:\Users\abc\StockApp\phase3\data\raw\yahoo_fundamental_data.csv')

    # print("Data is fetched and cleaned")
    # model,features=train_ml_model(r"C:\Users\abc\StockApp\phase3\data\preprocessed\cleaned_data.csv")
    # recommendations=score_stocks(r"C:\Users\abc\StockApp\phase3\data\preprocessed\cleaned_data.csv")
    pricedf=fetch_historical_data(symbols,months_ahead=6)


    # print(recommendations)



if __name__=="__main__":
    main()



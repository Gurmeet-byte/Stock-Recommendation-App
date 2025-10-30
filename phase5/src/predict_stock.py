import pandas as pd
import os 
import pickle as pkl 


def load_model():
    model_path=os.path.join("phase5","models","Stock_predictor.pkl")
    with open(model_path,"rb") as f:
        model=pkl.load(f)
    
    print("The models is Loaded succesfully")

    return model

def predict_new_stock(model,stock_data):
    df=pd.DataFrame([stock_data])
    prediction=model.predict(df)[0]


    return prediction


if __name__ == "__main__":
    model = load_model()

    
    
    stock_input = {
    "PE_Ratio": 15.2,
    "EPS": 3.1,
    "ROE": 12.5,
    "DebtToEquity": 0.4,
    "Price": 250,
    "YearHigh": 290,
    "YearLow": 200,
    "AvgVolume": 5000000,
    "Volatality": 0.08,
    "Current_price": 250,
    "Future_price": 275,
    "Volatality_index": 0.07,
    "Quality_Score": 0.82,
    "Growth_Score": 0.76
}

    

prediction=predict_new_stock(model, stock_input)

if prediction<0.3:
    print("Reccomendation SELL")

elif prediction <0.6:
    print("Reccomendation HOLD")

else:
    print("Reccomendation BUY")

    




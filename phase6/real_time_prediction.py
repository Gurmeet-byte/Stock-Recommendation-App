import pandas as pd 
import yfinance as yf
import os 
from datetime import datetime
import pickle as pkl 

def load_model():
    output_path=os.path.join("phase5","models","Stock_predictor.pkl")
    if not os.path.exists(output_path):
        return FileNotFoundError
    
    with open(output_path,"rb") as f:
        model=pkl.load(f)

    return model

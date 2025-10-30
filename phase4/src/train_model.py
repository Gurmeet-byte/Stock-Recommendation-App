import pandas as pd 
import os 
import numpy as np 
from sklearn.metrics import mean_absolute_error,r2_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import pickle as pkl 

def train_model():
    input_path=os.path.join("phase4","data","processed","model_ready_data.csv")
    model_dir=os.path.join("phase4","models")
    os.makedirs(model_dir,exist_ok=True)
    model_path=os.path.join(model_dir,"Stock_predictor.pkl")


    df=pd.read_csv(input_path)
    print("Reading the CSV file ")

    if "Forward_Return" not in df.columns:
        return ValueError("Check the file ")
    
    df=df.dropna(subset=['Forward_Return'])

    Y=df["Forward_Return"]
    X=df.drop(columns=['Symbol',"Forward_Return"],errors="ignore")
    X=X.select_dtypes(include=[np.number]).fillna(0)

    X_train,X_test,Y_train,Y_test=train_test_split(
        X,Y,random_state=42,test_size=0.2
    )

    model=RandomForestRegressor(
        n_estimators=200, #the numebr of tress
        random_state=42, # Random number generator 
        max_depth=8, # maximum of how much the tress will go 
        min_samples_split=5  #minimum samples required to split a node 
    )
    model.fit(X_train,Y_train)
    y_pred=model.predict(X_test)
    mae=mean_absolute_error(Y_test,y_pred)
    r2=r2_score(Y_test,y_pred)

    print(f"The Mean absolute error {mae}")
    print(f"The R2_Score is {r2}")


    
    with open(model_path,"wb") as f:
        pkl.dump(model,f)

    return model,mae,r2

if __name__=="__main__":
    print(train_model())
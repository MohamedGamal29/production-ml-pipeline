from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd


# 1. Initialize the app
app = FastAPI(title="House price prediction API", description="Production API for ML Model")


# 2. Load the trained model 
print(" Loading ML Model...")
try:
    model = joblib.load('models/rf_model.pk1')
    print(" Model Loaded Successfully")
except Exception as e:
    print(f" Error loading model : {e}")


# 3. Define the input data structure
class HouseFeatures(BaseModel):
    Square_Feet: float
    Bedrooms: float
    Age_Years: float
    Price_per_SqFt: float

 
# 4. Create the API Endpoint (The web URL)
@app.post("/predict")
def predict_price(features: HouseFeatures):
    user_values = [[
        features.Square_Feet,
        features.Bedrooms,
        features.Age_Years,
        features.Price_per_SqFt
    ]]
    input_data = pd.DataFrame(user_values, columns=model.feature_names_in_)  
    prediction = model.predict(input_data)[0]
    return {
        "status": "success",
        "predicted_price_usd": round(prediction, 2)
    }

@app.get("/")
def home():
    return {"message": "Welcome to the ML production API. Go to /docs to test the model."}

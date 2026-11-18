from fastapi import FastAPI
from pathlib import Path
from pydantic import BaseModel
from Machine_Learning_model.Inference import preprocess_single_house
import pandas as pd
import joblib

MODEL_PATH = Path(__file__).resolve().parent / "house_price_model.joblib"

app = FastAPI(title="House Price Prediction API")

artifacts=joblib.load(MODEL_PATH)

class HouseFeatures(BaseModel):
    House_ID: int
    City: str
    Locality_Tier: str
    Bathrooms: int
    BHK: int
    Super_Area_sqft: float
    Carpet_Area_sqft: float
    Floor_No: int
    Total_Floors: int
    Property_Age_years: int
    Parking: int
    Furnishing: str
    Lift: int
    Gated_Society: int
    Distance_to_Metro_km: float
    Distance_to_CityCenter_km: float
    Nearby_School_km: float
    Nearby_Hospital_km: float
    Crime_Rate_Index: float
    Price_per_sqft_INR: int

@app.get("/")
def home():
    return {"status": "The API is running."}

@app.post("/predict")
def predict(house: HouseFeatures):

    df = pd.DataFrame([house.model_dump()])
    data = preprocess_single_house(df, artifacts)
    prediction = artifacts['model'].predict(data)
    return {'Predicted Price': round(float(prediction[0]), 2)}
    
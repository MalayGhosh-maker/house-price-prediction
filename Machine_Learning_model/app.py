from fastapi import FastAPI
from pydantic import BaseModel
from Machine_Learning_model.Inference import preprocess_single_house
import pandas as pd
import joblib

app = FastAPI(title="House Price Prediction API")

artifacts=joblib.load("house_price_model.joblib")

class HouseFeatures(BaseModel):
    Area: float
    Bedrooms: int
    Bathrooms: int
    Floors: int
    YearBuilt: int
    Location: str
    Condition: str
    Garage: str

@app.get("/")
def home():
    return {"status": "The API is running."}

@app.post("/predict")
def predict(house: HouseFeatures):

    df = pd.DataFrame([house.model_dump()])
    X_scaled = preprocess_single_house(df, artifacts)
    prediction = artifacts['model'].predict(X_scaled)
    return {'Predicted Price': round(float(prediction[0]), 2)}
    
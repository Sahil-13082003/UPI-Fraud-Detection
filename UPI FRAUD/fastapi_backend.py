from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

# Load models and encoders
xgb_model = joblib.load('xgb_model.sav')
type_encoder = joblib.load('type_encoder.sav')
nameOrig_encoder = joblib.load('nameOrig_encoder.sav')
nameDest_encoder = joblib.load('nameDest_encoder.sav')

# FastAPI instance
app = FastAPI()

# Data model to be sent from frontend (optional step if needed to specify the structure)
class TransactionData(BaseModel):
    step: int
    type: str
    amount: float
    nameOrig: str
    oldbalanceOrg: float
    newbalanceOrig: float
    nameDest: str
    oldbalanceDest: float
    newbalanceDest: float

# Endpoint for prediction
@app.post("/predict")
async def predict(data: TransactionData):
    try:
        # Encode features
        type_encoded = safe_transform(type_encoder, data.type.lower())
        nameOrig_encoded = safe_transform(nameOrig_encoder, data.nameOrig)
        nameDest_encoded = safe_transform(nameDest_encoder, data.nameDest)

        # Feature vector
        features = np.array([[data.step, type_encoded, data.amount, nameOrig_encoded,
                              data.oldbalanceOrg, data.newbalanceOrig, nameDest_encoded,
                              data.oldbalanceDest, data.newbalanceDest]])

        # Predict
        prediction = xgb_model.predict(features)[0]
        label = "Fraud" if prediction == 1 else "Not Fraud"
        return {"prediction": label}

    except Exception as e:
        return {"error": str(e)}

def safe_transform(encoder, label):
    try:
        return encoder.transform([label])[0]
    except ValueError:
        return -1  # Unknown label fallback

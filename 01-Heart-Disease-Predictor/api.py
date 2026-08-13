## IMPORTS ##

from fastapi import FastAPI # FastAPI is the web framework
from pydantic import BaseModel # Defines the structure of patient data we receive. Pydantic is a data validation lib. BaseModel lets us define the SHAPE of data we expect to receive.
import joblib # load our saved model
import numpy as np # for data manipulation
import pandas as pd

class PatientData(BaseModel): # Defines exactly what data the API expects and the type(int or float). FastAPI automatically validates incoming data.
    age: int
    sex: int
    cp: int
    trestbps: int
    chol: int
    fbs: int
    restecg: int
    thalach: int
    exang: int
    oldpeak: float
    slope: int
    ca: int
    thal: int

## Create FastAPI App ##

app = FastAPI()

## Load Model and Scaler ##
model = joblib.load('heart_disease_model.pkl')
scaler = joblib.load('scaler.pkl')

## Predict Endpoint ##

@app.post("/predict") # called a decorator. Tells FastAPI this fxn handles POST(sending data to the server) requests.
def predict(patient: PatientData): # fxn called predict. FastAPI auto receives and validates the incoming data. patient contains the 13 fields the doctor sent.

    # Convert patient data to DataFrame
    input_data = pd.DataFrame([[
        patient.age, patient.sex, patient.cp, patient.trestbps, patient.chol, patient.fbs, patient.restecg, patient.thalach, patient.exang, patient.oldpeak, patient.slope, patient.ca, patient.thal
    ]], columns=['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal' ])

    # Scale and predict
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1]

    return{
        "prediction": int(prediction),
        "risk": f"{probability:.1%}",
        "message": "Heart Disease Detected" if prediction == 1 else "No Heart Disease"
    }


if __name__ == "__main__": # Will only run if you execute this file directly.
    import uvicorn # uvicorn is the actual server that runs FastAPI.
    uvicorn.run(app, host="0.0.0.0", port=8000)



# Doctor fills in patient details
#             ↓
# Clicks "Predict" button
#             ↓
# App sends POST request to /predict with patient data
#             ↓
# FastAPI receives data → runs model → returns prediction
#             ↓
# Doctor sees result



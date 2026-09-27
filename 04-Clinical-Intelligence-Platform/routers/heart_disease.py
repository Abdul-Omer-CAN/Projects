## Imports

from fastapi import APIRouter
import numpy as np
from pydantic import BaseModel
import joblib
import pandas as pd
import shap
import os
import io
from fastapi.responses import StreamingResponse
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter

router = APIRouter()

# Patient Data model 

class PatientData(BaseModel):
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

# Load Model and Scaler

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model = joblib.load(os.path.join(BASE_DIR, 'heart_disease_model.pkl'))
scaler = joblib.load(os.path.join(BASE_DIR, 'scaler.pkl'))

FEATURE_NAMES = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']

explainer = shap.TreeExplainer(model)

# Shared prediction logic, used by both /predict and /report

def run_prediction(patient: PatientData):
    input_data = pd.DataFrame([[
        patient.age, patient.sex, patient.cp, patient.trestbps, patient.chol,
        patient.fbs, patient.restecg, patient.thalach, patient.exang,
        patient.oldpeak, patient.slope, patient.ca, patient.thal
    ]], columns=FEATURE_NAMES)
    
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][0]

    shap_output = explainer.shap_values(input_scaled)

    if isinstance(shap_output, list):
        shap_values = shap_output[0][0]
    elif isinstance(shap_output, np.ndarray) and shap_output.ndim ==3:
        shap_values = shap_output[0, :, 0]
    else:
        shap_values = -shap_output[0]

    explanation = sorted(
        zip(FEATURE_NAMES, shap_values),
        key=lambda x: abs(x[1]),
        reverse=True
    )[:5]

    explanation_list = [
        {
            "feature": name,
            "impact": round(float(value), 3),
            "direction": "increased risk" if value > 0 else "decreased risk"
        }
        for name, value in explanation
    ]

    return {
        "prediction": int(prediction),
        "risk": f"{probability:.1%}",
        "message": "No Heart Disease" if prediction == 1 else "Heart Disease Detected",
        "explanation": explanation_list
    }

# Predict Endpoint

@router.post("/predict")
def predict(patient: PatientData):
    return run_prediction(patient)

# PDF Report Endpoint

@router.post("/report")
def generate_report(patient: PatientData):
    result = run_prediction(patient)

    buffer = io.BytesIO() 
    c = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter
    y = height -50 

    c.setFont("Helvetica-Bold", 16)
    c.drawString(50, y, "Clinical Intelligence Platform - Patient Report")
    y -= 30

    c.setFont("Helvetica", 12)
    c.drawString(50, y, f"Prediction: {result['message']}")
    y -= 20
    c.drawString(50, y, f"Risk: {result['risk']}")
    y -= 30

    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, y, "Patient Inputs:")
    y -= 20
    c.setFont("Helvetica", 11)
    for field, value in patient.model_dump().items():
        c.drawString(60, y, f"{field}: {value}")
        y -= 15

    y -= 15
    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, y, "Top Contributing Factors:")
    y -= 20
    c.setFont("Helvetica", 11)
    for item in result["explanation"]:
        c.drawString(60, y, f"{item['feature']}: {item['direction']} (impact: {item['impact']})")
        y -= 15

    c.save()
    buffer.seek(0)

    return StreamingResponse(
        buffer, 
        media_type="application/pdf",
        headers={"Content-Disposition": "attachment; filename=patient_report.pdf"}
    )


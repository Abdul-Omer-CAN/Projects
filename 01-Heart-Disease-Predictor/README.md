# Heart Disease Predictor 🫀

A machine learning API that predicts heart disease risk using XGBoost with 80% accuracy.

## What it does
- Takes 13 patient features as input
- Predicts heart disease risk (0 = No Disease, 1 = Disease)
- Returns prediction + risk percentage
- Processeds via FastAPI REST API

## Tech Stack
- Python
- XGBoost
- scikit-learn
- FastAPI
- uvicorn
- pandas
- joblib

## How to Run

### 1. Install dependencies
pip install -r requirements.txt

### 2. Train the model
python train.py

### 3. Run the API
python api.py

### 4. Test the API
Go to: http://localhost:8500/docs

## API Endpoint
POST /predict

### Example Input
{
    "age": 55,
    "sex": 1,
    "cp": 2,
    "trestbps": 140,
    "chol": 250,
    "fbs": 0,
    "restecg": 1,
    "thalach": 150,
    "exang": 0,
    "oldpeak": 2.3,
    "slope": 0,
    "ca": 0,
    "thal": 1
}

### Example Output
{
    "prediction": 1,
    "risk": "83.8%",
    "message": "Heart Disease DETECTED"
}

## Dataset
UCI Heart Disease Dataset - 303 patients, 13 features

## Author
Abdulrehman Omer | github.com/Abdul-Omer-CAN
## Imports ##

import numpy as np
import pandas as pd
import joblib # to load our saved model and scaler

## Load Model and Scaler ##

model = joblib.load('heart_disease_model.pkl') # load trained model
scaler = joblib.load('scaler.pkl') # load scaler
print("Model and scaler loaded!")

## Predict Function ##

# Define a fxn that takes all 13 patient features as inputs.
def predict_heart_disease(age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal):

    # Convert patient inputs into a numpy array. double brackets mean [[]] a 2d array model expects 2d.
    input_data = pd.DataFrame([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]],
                              columns =['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']) 

    # Scale the input using same scaler from training.
    input_scaled = scaler.transform(input_data) 

    # Make prediction
    prediction = model.predict(input_scaled)[0] # [0] means just the first patient.
    probability = model.predict_proba(input_scaled)[0][1] # predict_proba means returns probabilities for each class. [0] means first patient & [1] means probability of class 1 (heart disease). grab patient 1's data of heart disease.

    # Return result
    if prediction == 1: # if model predicted heart disease then return warning message.
        return f"Heart Disease DETECTED - Risk: {probability:.1%}"
    else: # if not then return the message below.
        return f"No Heart Disease - Risk: {probability:.1%}"

    
## Test with a sample patient ##

result = predict_heart_disease(age=30, sex=0, cp=2, trestbps=110, chol=170, fbs=0, restecg=0, thalach=175, exang=0, oldpeak=0.0, slope=2, ca=0, thal=3)
print(result)

# We can improve results for underrepresented groups by using SMOTE aka Synthetic Minority Oversampling Technique. 
# Creates synthetic samples for underrepresented groups and balances the dataset. Improves predictions for minority groups.
# In our sample patient age=30 & sex=0(female combo) is underrepresented or doesnt exist in our database hence model defaults to high risk.

## Imports ##

import pandas as pd # Load and manipulate data
import numpy as np # Mathematical operations
from sklearn.model_selection import train_test_split # split data into train/test
from sklearn.preprocessing import StandardScaler # scale features to same range
from sklearn.metrics import accuracy_score # measure model accuracy
from sklearn.metrics import classification_report # detailed precision/recall/f1-score
from xgboost import XGBClassifier # our main ML model
import  joblib # Save and Load trained model


## Load Dataset ##

url = "https://raw.githubusercontent.com/sharmaroshan/Heart-UCI-Dataset/master/heart.csv"
df = pd.read_csv(url)

# We are checking: Did the data load correctly? How big is it? What columns do we have?

print("Shape:", df.shape) # Shows how many rows and columns are in the dataset will show -> Shape: (patients, features)
print("Columns:", df.columns.tolist()) # Shows the names of all the columns. So we know what features we are working with.
print(df.head()) # Shows the first 5 rows of the dataset. To check the data has been loaded correctly.

## Data Exploration ##

print("\nMissing values:") # Check if any data is missing # LABEL
print(df.isnull().sum()) # Count missing values per column. We need zero 

print("\nTarget distribution:") # Check how many sick vs healthy patients. # LABEL
print(df['target'].value_counts()) # 0 = no disease, 1 = disease - should be around balanced, so that model learns fairly.\

print("\nBasic statistics:") # Get mean, min, max and std for each column. # LABEL
print(df.describe()) # Quick overview of all numerical features

## Feature and Target Split ##

X = df.drop('target', axis=1) # features - remove target column from the dataframe. The inputs into our model. axis=1 means drop the target column.
y = df['target'] # target - what we want to predict

print("Features shape:", X.shape) # should be (303, 13) aka 303 rows and 13 columns. A dataframe -> rows x column.
print("Target shape:", y.shape) # should be (303,) aka y should have 303 values. this is called series. not a dataframe. will only show us 0 or 1 whether patient has disease or not. Series -> just rows.
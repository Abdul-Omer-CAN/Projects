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



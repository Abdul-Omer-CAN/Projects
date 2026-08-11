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


##  Train/Test split ##

X_train, X_test, y_train, y_test = train_test_split(  # X -> features & y -> testing
    X, y,
    test_size=0.2,  # 20% is for testing and 80% for training
    random_state=42 # ensures same split everytime we run
)

print("Training set size:", X_train.shape) # will be 80% of the total patients (242, 13) # .shape shows (rows,columns)
print("Testing set size:", X_test.shape) # will be 20% of the total patients (61,13)

## Feature Scaling ##

scaler = StandardScaler()   # Create scaler object contains the scaler
X_train = scaler.fit_transform(X_train) # learn the mean & std from training data(fit) and scale(_transform ). Gives us a value around 0    
X_test = scaler.transform(X_test) # scale test data using same mean/std from training.
# mean is always zero and in a normal distribution values fall within 3 standard deviations from mean.
print("Scaling complete!")

## Train XGBoost Model(prebuilt ML algo) ##

model = XGBClassifier( # creates an XGBoost classification model and stores it in model
    n_estimators=100, #number of trees to build. each tree learns from the one before it. More trees=better accuracy but also will take more time.
    max_depth=4,  # how deep each tree grows. each tree can go 4 levels deep. It prevents overfitting. 4 level deep means itll ask 4 questions. 20 will be too deep and 1 will be too shallow.
    learning_rate=0.1, # how fast the model learns or how much each tree corrects the previous trees mistakes. 0.1 means take small careful steps. 1 is too high. 0.1 is sweet spot.
    random_state=42 # reproducibility. same result each time you run it.
)

model.fit(X_train, y_train) # train the model on the training data. Finds patterns between bp, age etc.
print("Model training complete!")


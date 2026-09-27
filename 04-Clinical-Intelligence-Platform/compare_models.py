## Imports ##

import pandas as pd # Load and manipulate data
import numpy as np # Mathematical operations
from sklearn.model_selection import train_test_split # split data into train/test
from sklearn.preprocessing import StandardScaler # scale features to same range
from sklearn.metrics import accuracy_score # measure model accuracy
from sklearn.metrics import classification_report # detailed precision/recall/f1-score
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier # our main ML model
import  joblib # Save and Load trained model
import mlflow
import mlflow.xgboost
import mlflow.sklearn


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
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

mlflow.set_experiment("heart-disease-automl-comparison")

## Define the candidates ##

candidates = {
    "XGBoost": XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.14, random_state=42),
    "RandomForest": RandomForestClassifier(n_estimators=100, max_depth=3, random_state=42),
    "LogisticRegression":LogisticRegression(max_iter=1000, random_state=42)
}

results = []

## Train and evaluate each candidate ##

for name, candidate_model in candidates.items():
    with mlflow.start_run(run_name=name):

        candidate_model.fit(X_train_scaled, y_train)
        y_pred = candidate_model.predict(X_test_scaled)

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        mlflow.log_param("model_type", name)
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)

        if name == "XGBoost":
            mlflow.xgboost.log_model(candidate_model, name="model")
        else: mlflow.sklearn.log_model(candidate_model, name="model", skops_trusted_types=["sklearn.tree._tree.Tree"])

        print(f"\n{name}")
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall:  {recall:.4f}")
        print(f"  F1 Score:  {f1:.4f}")

        results.append({
            "name": name,
            "model": candidate_model,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1
        }) 

    ## Pick the winner ##

best_result = max(results, key=lambda r: r["f1_score"])

print("\n" + "="*40)
print("Comparison Summary")
print("="*40)
for r in sorted(results, key=lambda r: r["f1_score"], reverse=True):
    marker = " <- WINNER" if r["name"] == best_result["name"] else ""
    print(f"{r['name']:20s} f1={r['f1_score']:.4f} accuracy={r['accuracy']:.4f}{marker}")

print(f"\nSelected model: {best_result['name']}")
print("Reason: highest F1 score, meaning the best balance between catching actual heart disease cases (recall) and not raising too many false alarms (precision).")


## Save Model ##

joblib.dump(best_result["model"], 'heart_disease_model.pkl')
joblib.dump(scaler, 'scaler.pkl') # Save Scaler too, needed for predictions. It contains the mean and std it learnt from training data. MUST use same scaler for new predictions otherwise scaling will be wrong.
print("Model and scaler saved!") # .pkl is pickle file type. It is python's way of saving ANY Python object to disk.    

